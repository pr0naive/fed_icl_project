# Code Walkthrough

A file-by-file tour of the codebase, explaining what each module does, the design decisions behind it, and the non-obvious behaviours a future reader (or a future me) needs to know.

The companion files are `concepts.md` (the conceptual reference) and `glossary.md` (one-line definitions and the variable map).

## Repository layout

fed_icl_project/
|-- config.py                  Parameters; env-variable overridable.
|-- data.py                    AG News loading; Dirichlet partition; train/test split discipline.
|-- llm.py                     Ollama interface; prompt construction; label parser with fallback logging.
|-- federation.py              FedICLClient and FedICLServer; selection and ordering strategies.
|-- main.py                    Entry point; baselines; Fed-ICL loop; held-out evaluation; JSON output.
|-- requirements.txt
|-- README.md
|-- glossary.md
|-- code_walkthrough.md        (this file)
|-- lab_notebook.md            Running log of decisions and results.
|-- methodology_checklist.md   Pre-flight checks before each run.
|-- results_*.json             Run outputs, auto-named from the config.

Throughout this document, code variable names appear in `CODE_FONT`. The same names appear in `glossary.md` with one-line definitions; this file gives the *why* behind each.

---

## config.py

Two responsibilities: hold experimental parameters with sensible defaults, and allow each parameter to be overridden from the shell via an environment variable. The pattern is:

```python
MODEL_NAME      = os.environ.get("FED_ICL_MODEL", "mistral")
NUM_CLIENTS     = int(os.environ.get("FED_ICL_CLIENTS", 3))
NUM_ROUNDS      = int(os.environ.get("FED_ICL_ROUNDS", 6))
DIRICHLET_ALPHA = float(os.environ.get("FED_ICL_ALPHA", 0.5))
NUM_SHOTS       = int(os.environ.get("FED_ICL_K", 3))
SELECTION_STRATEGY = os.environ.get("FED_ICL_SEL", "random")
ORDER_STRATEGY     = os.environ.get("FED_ICL_ORDER", "original")
NUM_SERVER_QUERIES = int(os.environ.get("FED_ICL_Q", 100))
EVAL_SIZE          = int(os.environ.get("FED_ICL_EVAL", 100))
SEED               = int(os.environ.get("FED_ICL_SEED", 42))
```

The reason. A sweep across orderings, alphas, and seeds becomes a shell loop, not a series of source edits. Each results file is fully described by the launch command. A run started yesterday can be reproduced today by recovering its config from the JSON it wrote.

`OLLAMA_HOST`, `TEMPERATURE`, and `MAX_TOKENS` are also here. Temperature is fixed at 0.0 for determinism; `MAX_TOKENS` is 10 because the model is asked to emit one label.

---

## data.py

Three concerns: load AG News, define the label space, partition data across clients.

### Loading

`_load_ag_news(num_examples=350, seed=SEED)` pulls a subsample from the AG News training split. The size is small (350 of 120,000) because each run does roughly 3,600 model calls on the M4, and a larger pool would either inflate per-run time or push the pool size beyond what is practical for ICL. The subsample is fixed by `seed`, so the same 350 examples appear in every run.

`_load_ag_news_test(num_examples=EVAL_SIZE, seed=SEED+1)` pulls a separate subsample from AG News' test split, used exclusively for held-out evaluation. Independent seeding (`seed+1`) means changing the training subsample does not silently change the held-out set.

The 350-row pool is partitioned by `prepare_experiment()`:

```python
data = RAW_DATA.copy()
np.random.shuffle(data)
eval_set       = _load_ag_news_test(EVAL_SIZE, SEED)
server_queries = data[:NUM_SERVER_QUERIES]
client_pool    = data[NUM_SERVER_QUERIES:]
client_datasets = partition_data_dirichlet(client_pool, NUM_CLIENTS, DIRICHLET_ALPHA)
```

At the defaults: 350 train pool → 100 server queries + 250 client pool. The client pool is then Dirichlet-partitioned across the three clients. With α=0.5 on the reference run, partition sizes were 113, 80, 57.

### Label space

`LABEL_SPACE = ["world", "sports", "business", "science"]`. The order matters because the parser's fallback returns `LABEL_SPACE[0]` ("world") when no label can be extracted. This is documented in `llm.py`.

### Dirichlet partition

`partition_data_dirichlet(data, num_clients, alpha)` allocates each class across the clients independently:

```python
for c in range(num_classes):
    class_indices = np.where(labels == c)[0]
    np.random.shuffle(class_indices)
    proportions = np.random.dirichlet([alpha] * num_clients)
    counts = (proportions * len(class_indices)).astype(int)
    ...
```

The `.astype(int)` truncates, which means the integer counts can sum to fewer examples than the class actually has. The leftover examples must go somewhere. The previous code dumped them all on client 0:

```python
counts[0] += len(class_indices) - counts.sum()    # OLD: biased client 0
```

This was a methodological bug. Because the local-only baseline uses client 0, client 0 was systematically larger than its Dirichlet share, and that inflated the local-only accuracy.

The current code distributes the residual round-robin from a random starting client:

```python
deficit = len(class_indices) - int(counts.sum())
if deficit > 0:
    start = np.random.randint(num_clients)
    for offset in range(deficit):
        counts[(start + offset) % num_clients] += 1
```

The fix moved local-only on llama3 by roughly six percentage points in the reference run.

---

## llm.py

Three concerns: talk to Ollama, build the ICL prompt, parse the model's response into a label.

### Prompt construction

`build_icl_prompt(examples, query_text)` produces strings like:

Classify the following news headline into exactly one of these categories:
"world", "sports", "business", or "science". Reply with only the category
label, nothing else.
Headline: "Markets rally on jobs data."
Category: business
Headline: "Federer wins fifth title."
Category: sports
Headline: "Conflict in eastern Europe escalates."
Category:

The instruction is explicit ("Reply with only the category label, nothing else") because llama3 in particular has a tendency to add commentary if not constrained. Combined with `MAX_TOKENS=10`, the model emits short responses in practice. The reference run's parse fallback rate of 0.1% confirms this.

### Ollama call

`query_ollama(prompt, model, max_retries=2)` posts to the Ollama HTTP API. Two non-obvious behaviours:

- **Retry on transient errors.** Previously, a single failed call returned an empty string, which the parser silently turned into a phantom "world" prediction. The current code retries up to two extra times with a 3-second sleep between attempts. This addresses the HTTP 500 / orphaned-runner failure observed earlier.
- **Connection errors re-raised**, not retried. If Ollama is not running at all, the run should fail fast rather than spend a minute retrying.

### Label parsing

`parse_label(raw_response)` is the file's most subtle function. It does two things: try to extract a label, and log when it cannot.

The matching rule:

```python
text = raw_response.lower().strip().strip(".,!\"'")
tokens = [t for t in re.split(r"[\s.,!?:;]+", text) if t]

if tokens and tokens[0] in LABEL_SPACE:
    return tokens[0]
for tok in tokens:
    if tok in LABEL_SPACE:
        return tok
```

Token-level matching, not substring matching. The previous rule was `if label in text`, which matched "world" inside any sentence containing the word, including sports stories about world cups. That rule misclassified verbose model outputs in a way that correlated with ordering strategy (poor orderings make models more verbose), which would have confounded the ordering experiments.

The fallback:

```python
_PARSE_STATS["fallback"] += 1
if len(_PARSE_STATS["examples"]) < 30:
    _PARSE_STATS["examples"].append(raw_response[:120])
return LABEL_SPACE[0]
```

When no label is found anywhere, the parser still returns "world", but it counts the fallback and stores up to 30 sample raw responses for inspection. `get_parse_stats()` exposes the totals, and `main.py` writes them into the results JSON. If the fallback rate ever exceeds 10% in a condition, the accuracy for that condition is flagged as parser-influenced. On the reference run the rate was 0.083% (rounded to 0.1%).

---

## federation.py

Two classes: `FedICLServer` and `FedICLClient`. The selection and ordering machinery lives on the client. Aggregation lives on the server.

### FedICLClient

`select_examples(query_text, pool, n)` picks K examples from a pool. The two implemented rules:

```python
if SELECTION_STRATEGY == "similarity":
    query_words = set(query_text.lower().split())
    scores = []
    for text, label in pool:
        example_words = set(text.lower().split())
        scores.append(len(query_words & example_words))
    top_indices = np.argsort(scores)[-n:]
    selected = [pool[i] for i in top_indices]
    np.random.shuffle(selected)        # ← critical
    return selected
```

The shuffle at the end is essential. `np.argsort(scores)[-n:]` returns the top-n indices sorted ascending by score, which means without the shuffle the returned list is already in similarity-ascending order. Under `ORDER_STRATEGY="original"`, this would mean the control condition is actually similarity-ascending, contaminating every ordering comparison under similarity selection. The shuffle makes "original" a genuine control.

The random selection branch is straightforward `np.random.choice(len(pool), size=n, replace=False)`.

`order_examples(examples, query_text)` arranges the K selected examples. The six rules:

- **`original`**: returns `list(examples)`. Returns a copy, not the input, so the caller's list is not mutated.
- **`similarity_ascending` / `similarity_descending`**: sorts by lexical overlap with the query.
- **`label_grouped`**: sorts by label, so all examples of one class appear consecutively.
- **`label_alternating`**: pops one example from each label's bucket in turn until all buckets are empty.
- **`random_shuffle`**: a fresh `np.random.shuffle` each call.

All six produce a new list rather than mutating the input. The previous version used `examples.sort(...)`, which mutated the caller's list. Defensive design; prevents bugs in any caller that reuses an example list.

`relabel_local_data(global_context)` is the step where the client re-labels its own examples using the global context as demonstrations. This is the protocol's main mechanism for letting the global state shape what each client thinks its examples are.

`predict_server_queries(server_queries, global_context)` produces the client's per-query predictions for the round. The candidate pool here is `self.local_data + self.relabelled_data`. The relabelled set carries the global signal; the local set provides the client's private contribution.

### FedICLServer

The server holds three things: the server queries (queries with hidden ground-truth labels), the global context (the server's current best-guess labels for those queries), and the aggregation function.

`__init__` populates the global context with a uniformly random label for each query:

```python
self.global_context = [(text, np.random.choice(LABEL_SPACE)) for text, _ in server_queries]
```

This gives the protocol a deliberately bad starting point (round 0 accuracy near 25% for the 4-class task), against which subsequent round accuracy is measured.

`aggregate_predictions(all_predictions)` updates the global context by majority vote per query. Implementation uses `Counter(votes).most_common(1)[0][0]`. Ties are broken by insertion order, which is the order of `votes`, which is the client iteration order. This is documented but not parameterised; in practice with three clients on four labels, ties are uncommon enough not to dominate the result.

`evaluate_context()` computes round accuracy by comparing the global context to ground truth.

---

## main.py

The orchestration file. Five responsibilities: smoke-check Ollama, run baselines, run Fed-ICL, run held-out evaluation, write the results JSON.

### Baselines

`run_baseline_zero_shot(eval_set)` calls `predict_with_icl([], text)` for each example. No demonstrations, no ordering, no federation. Pure model behaviour on the test split.

`run_baseline_local_only(client_datasets, eval_set)` constructs a `FedICLClient` from `client_datasets[0]` and calls its `select_examples` and `order_examples` methods, exactly like Fed-ICL does. Whatever `SELECTION_STRATEGY` and `ORDER_STRATEGY` are set to, the local-only baseline applies them. Previously this function had its own inline `np.random.choice` for selection and no ordering, which conflated federation and ordering effects in any cross-condition comparison. The current version isolates the federation effect cleanly.

### Fed-ICL loop

`run_fed_icl(server_queries, client_datasets, eval_set)` is the main loop. The structure mirrors the protocol described in `concepts.md`:

```python
init_eval = server.evaluate_context()                # round 0
for t in range(1, NUM_ROUNDS + 1):
    global_context = server.get_global_context()
    all_predictions = []
    for client in clients:
        client.relabel_local_data(global_context)
        preds = client.predict_server_queries(server_queries, global_context)
        all_predictions.append(preds)
    server.aggregate_predictions(all_predictions)
    eval_result = server.evaluate_context()
    results["rounds"].append({...})
```

After the rounds, the held-out evaluation block builds a `FedICLClient` from the final global context and scores each test example through the same select-and-order pipeline. Previously this block used inline random selection with no ordering, which meant held-out accuracy was not comparable across ordering conditions. The current version makes the held-out set a clean reflection of the deployed behaviour.

### Results

The output dict carries the full config (every parameter that determined the run), the round-by-round accuracy trajectory, both baselines, the held-out accuracy, the parse statistics, and the total time. The filename is generated from the config:

```python
out_path = (
    f"results_{MODEL_NAME}"
    f"_alpha{DIRICHLET_ALPHA}"
    f"_K{NUM_SHOTS}"
    f"_T{NUM_ROUNDS}"
    f"_seed{SEED}"
    f"_order-{ORDER_STRATEGY}.json"
)
```

Two consequences: runs with different parameters never overwrite each other, and a `ls results_*.json` is a self-documenting inventory of every experiment that has been run.

### The summary print

`print_summary(results)` produces the terminal banner that ends each run, including the round-by-round bar chart, the baselines, the held-out accuracy, the federation gain over local-only, and the parse fallback rate. The federation-gain line is the single most useful piece of output for telling at a glance whether the run produced a meaningful result.

---

## The lifecycle of a single run

What happens, in order, when you execute:

```bash
FED_ICL_MODEL=llama3 FED_ICL_ALPHA=0.5 FED_ICL_K=3 \
FED_ICL_ORDER=original FED_ICL_SEED=42 python main.py
```

1. `config.py` reads the environment variables, falling back to defaults for anything not set.
2. `data.py` loads the 350-row AG News train subsample (seed 42) and the 100-row test subsample (seed 43).
3. `data.py` shuffles, slices off the first 100 as server queries, leaves 250 as client pool, and Dirichlet-partitions the pool across three clients.
4. `llm.py` checks Ollama is reachable and the model is loaded.
5. `main.py` runs the zero-shot baseline on the test split.
6. `main.py` runs the local-only baseline using Client 0, with the configured selection and ordering applied.
7. `main.py` instantiates the server (with random initial global context) and the three clients.
8. For each of T=6 rounds: each client relabels its local data, predicts the 100 server queries, the server aggregates by majority vote, and round accuracy is recorded.
9. `main.py` runs the held-out evaluation on the AG News test split, using the final global context with the same selection and ordering applied.
10. The summary is printed; the JSON is written; the program exits.

Total model calls at the defaults: 100 zero-shot + 100 local-only + (3 clients × (mean local pool ≈ 83) × 6 rounds) + (3 clients × 100 server queries × 6 rounds) + 100 held-out ≈ 3,600. At roughly 2 seconds per call on the M4, this is about 2 to 2.5 hours.

---

## Things that look like bugs but are not

A reader new to the code is likely to flag these. They are intentional.

**Round 0 accuracy is roughly 25%.** This is the deliberately random initialisation of the global context. The actual learning happens in rounds 1 through T.

**Client sizes are very uneven.** With α=0.5 on a small pool, the Dirichlet partition can give one client three times more data than another. This is the heterogeneity the protocol is designed to handle.

**Local-only accuracy can equal zero-shot accuracy.** With short news headlines, K=3 demonstrations sometimes do not add information beyond the task instruction. This is a feature of the dataset, not a bug in the baseline.

**The parse fallback returns "world".** Returning the first label in `LABEL_SPACE` is a deliberate choice; it is logged so the rate can be reported alongside accuracy. The threshold for flagging a condition as parser-influenced is 10%.

**`np.random.shuffle` of the eval set uses the global seed.** All randomness in the run is seeded from the single `SEED` value, so the same seed produces byte-identical results across re-runs.

---

## Things that are not in the code yet but are on the roadmap

**Embedding-based KATE-style similarity selection.** The current selection uses lexical overlap. SBERT (sentence-transformers `all-MiniLM-L6-v2`) would be a faithful replication of Liu et al. (2022). The proposal acknowledges this divergence honestly rather than overclaiming.

**Per-class accuracy breakdown.** Currently only overall accuracy is reported. For ordering experiments (especially label-grouped and label-alternating), per-class accuracy will be more informative because Zhao et al.'s biases are per-class effects.

**Multi-seed sweep wrapper.** The code already supports per-run seed overrides via `FED_ICL_SEED`, but the proposal's "three seeds per cell" requires a shell loop to be written. Trivial but not yet committed.

**Confidence intervals via paired bootstrap.** The methodology section commits to this; the code does not yet compute it. A small post-processing script over the multi-seed JSONs will produce the intervals.