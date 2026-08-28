# Code Walkthrough

A file-by-file tour of the codebase, explaining what each module does, the design decisions behind it, and the non-obvious behaviours a future reader (or a future me) needs to know. The companion file is `glossary.md`, which gives one-line definitions and the variable map; this file gives the why behind each.

Throughout this document, code variable names appear in `CODE_FONT`. The same names appear in `glossary.md`.

## Repository layout

```
fed_icl_project/
├── config.py                  Parameters; env-variable overridable.
├── data.py                    Dataset loading; Dirichlet partition; regime and split discipline.
├── llm.py                     Ollama interface; prompt construction; label parser with fallback logging.
├── federation.py              FedICLClient and FedICLServer; selection and ordering strategies.
├── main.py                    Entry point; baselines; Fed-ICL loop; held-out evaluation; JSON output.
├── results/                   Per-seed result JSONs behind the reported tables.
│   └── archive/               Superseded and pre-protocol runs, kept for provenance.
├── logs/                      Pilot run logs.
├── docs/
│   ├── code_walkthrough.md    (this file)
│   ├── glossary.md            One-line definitions and the variable map.
│   ├── lab_notebook.md        Dated running log of decisions and results.
│   └── methodology_checklist.md  Pre-flight checks before each run.
├── README.md
├── REPRODUCE.md               How to verify a result cell.
├── requirements.txt
└── .gitignore
```

The five source modules stay at the repository root because their imports are flat (`from config import ...`) and `main.py` writes output relative to the working directory.

---

## config.py

Two responsibilities: hold experimental parameters with defaults, and allow each to be overridden from the shell via an environment variable. The pattern:

```python
MODEL_NAME         = os.environ.get("FED_ICL_MODEL", "mistral")
NUM_CLIENTS        = int(os.environ.get("FED_ICL_CLIENTS", 3))
NUM_ROUNDS         = int(os.environ.get("FED_ICL_ROUNDS", 6))
DIRICHLET_ALPHA    = float(os.environ.get("FED_ICL_ALPHA", 0.5))
NUM_SHOTS          = int(os.environ.get("FED_ICL_K", 3))
SELECTION_STRATEGY = os.environ.get("FED_ICL_SEL", "similarity_embedding")
ORDER_STRATEGY     = os.environ.get("FED_ICL_ORDER", "original")
NUM_SERVER_QUERIES = int(os.environ.get("FED_ICL_Q", 100))
EVAL_SIZE          = int(os.environ.get("FED_ICL_EVAL", 1000))
EVAL_SEED          = int(os.environ.get("FED_ICL_EVAL_SEED", 12345))
SEED               = int(os.environ.get("FED_ICL_SEED", 42))
```

The reason: a sweep across orderings, alphas, and seeds becomes a shell loop, not a series of source edits, and each results file is fully described by its launch command.

Two defaults are worth calling out because they do not match the reported experiments. `DATA_REGIME` defaults to `canonical`, but every reported run uses `canonical_full`. `FILTER_LOCAL_DATA` defaults to off, but every reported run has it on. So bare `python main.py` runs an unfiltered canonical frame; the reported runs set `FED_ICL_REGIME=canonical_full FED_ICL_FILTER=1`. `OLLAMA_HOST`, `TEMPERATURE` (0.0 for determinism), and `MAX_TOKENS` (small, since the model emits one label) are also here.

<!-- VERIFY: this block reflects config.py as of the last read. If you change
     the in-file defaults for DATA_REGIME or FILTER_LOCAL_DATA to match the
     reported setup, update this paragraph accordingly. -->

---

## data.py

Three concerns: load the dataset, define the label space, partition data across clients.

### Loading and regime

The reported runs use the `canonical_full` regime: the full training split is the pool and query frame, and the held-out evaluation set is drawn disjointly from the test split. A client pool of `CLIENT_POOL_SIZE` (250) examples is sampled and partitioned across the clients; `NUM_SERVER_QUERIES` (100) server queries and the `EVAL_SIZE` (1000) held-out set are drawn from the test split. The held-out set is class-balanced and seeded by `EVAL_SEED` (12345), fixed and independent of the partition `SEED`, so changing the partition never changes the evaluation sample.

The `DATASET` switch selects AG News (four classes) or DBpedia (14 classes); the label space and class count follow from it.

<!-- VERIFY: data.py was rewritten for canonical_full and DBpedia. Confirm the
     exact loader/function names and the pool/query/eval split against the
     current data.py before relying on the specifics above. -->

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

The `.astype(int)` truncates, so the integer counts can sum to fewer examples than the class actually has. The leftover must go somewhere. The previous code dumped it all on client 0:

```python
counts[0] += len(class_indices) - counts.sum()    # OLD: biased client 0
```

This was a methodological bug: because the local-only baseline is built from a single client, systematically enlarging one client inflated the local-only accuracy. The current code distributes the residual round-robin from a random starting client:

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

Three concerns: talk to Ollama, build the ICL prompt, parse the response into a label.

### Prompt construction

`build_icl_prompt(examples, query_text)` wraps the K demonstrations and the query in the fixed template. The instruction is explicit ("Reply with only the category label, nothing else") because some models add commentary otherwise. Combined with a small `MAX_TOKENS`, responses stay short in practice.

### Ollama call

`query_ollama(prompt, model, max_retries=2)` posts to the Ollama HTTP API. Two non-obvious behaviours:

- **Retry on transient errors.** Previously a single failed call returned an empty string, which the parser silently turned into a phantom prediction. The current code retries up to two extra times with a short sleep, addressing the HTTP 500 / orphaned-runner failure observed earlier.
- **Connection errors re-raised**, not retried. If Ollama is not running at all, the run fails fast rather than spending a minute retrying.

### Label parsing

`parse_label(raw_response)` extracts a label and logs when it cannot. The matching is token-level, not substring:

```python
text = raw_response.lower().strip().strip(".,!\"'")
tokens = [t for t in re.split(r"[\s.,!?:;]+", text) if t]

if tokens and tokens[0] in LABEL_SPACE:
    return tokens[0]
for tok in tokens:
    if tok in LABEL_SPACE:
        return tok
```

The previous rule was `if label in text`, which matched a label word inside any sentence containing it, and misclassified verbose outputs in a way that correlated with ordering strategy (poor orderings make models more verbose), confounding the ordering experiments. When no label is found anywhere, the parser returns `LABEL_SPACE[0]`, counts the fallback, and stores sample raw responses for inspection. `main.py` writes the totals into the results JSON, and any condition whose fallback rate exceeds 10% is flagged as parser-influenced.

---

## federation.py

Two classes: `FedICLServer` and `FedICLClient`. Selection and ordering live on the client; aggregation lives on the server.

### FedICLClient

`select_examples(query_text, pool, n)` picks K examples. The default is embedding kNN; lexical and random are ablation arms. The lexical branch:

```python
if SELECTION_STRATEGY == "similarity":
    query_words = set(query_text.lower().split())
    scores = [len(query_words & set(text.lower().split())) for text, label in pool]
    top_indices = np.argsort(scores)[-n:]
    selected = [pool[i] for i in top_indices]
    np.random.shuffle(selected)        # ← critical
    return selected
```

The shuffle at the end is essential. `np.argsort(scores)[-n:]` returns the top-n indices sorted ascending by score, so without the shuffle the returned list is already in similarity-ascending order, which would make the `original` control secretly similarity-ascending and contaminate every ordering comparison. The shuffle makes `original` a genuine control. The `similarity_embedding` default selects by cosine similarity in paraphrase-MiniLM-L6-v2 space (sentence-transformers) and shuffles the selected set for the same reason. The `random` branch is a straightforward `np.random.choice(len(pool), size=n, replace=False)`.

`order_examples(examples, query_text)` arranges the K selected examples per `ORDER_STRATEGY`. All six rules (`original`, `similarity_ascending`, `similarity_descending`, `label_grouped`, `label_alternating`, `random_shuffle`) return a new list rather than mutating the input; a previous version used in-place `examples.sort(...)`, which mutated the caller's list. Ordering similarity is lexical word overlap, deliberately distinct from the embedding similarity used for selection.

`relabel_local_data(global_context)` re-labels the client's own examples using the global context as demonstrations, the protocol's mechanism for letting the shared state shape each client's view of its data. `predict_server_queries(server_queries, global_context)` produces the client's per-query predictions; under the `fed_icl` variant the candidate pool is `self.local_data + self.relabelled_data`, and under `fed_icl_free` it is the relabelled set only.

### FedICLServer

The server holds the server queries, the global context (its current best-guess labels for those queries), and the aggregation function. `__init__` seeds the global context with a uniformly random label per query, a deliberately bad round-0 start (accuracy near chance) against which subsequent rounds are measured. `aggregate_predictions(all_predictions)` updates the context by per-query majority vote (`Counter(votes).most_common(1)`), ties broken by client iteration order. `evaluate_context()` computes round accuracy against ground truth.

---

## main.py

The orchestration file: smoke-check Ollama, run baselines, run Fed-ICL, run held-out evaluation, write the JSON.

### Baselines

`run_baseline_zero_shot(eval_set)` predicts each held-out example with no demonstrations. `run_baseline_local_only(client_datasets, eval_set)` builds a client from one partition and scores the held-out set through the same select-and-order-and-filter pipeline as Fed-ICL, so whatever `SELECTION_STRATEGY`, `ORDER_STRATEGY`, and filtering are set to, the local-only baseline applies them. A previous version used inline random selection with no ordering, which conflated federation and ordering effects; the current version isolates the federation effect on identical held-out data.

### Fed-ICL loop

`run_fed_icl(server_queries, client_datasets, eval_set)` is the main loop:

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
    results["rounds"].append({...})
```

After the rounds, the held-out block builds a client from the final global context and scores each held-out example through the same select-and-order pipeline, so held-out accuracy is comparable across ordering conditions.

### Results

The output dict carries the full config, the round-by-round trajectory, both baselines, the held-out accuracy, the parse statistics, and the total time. The filename is generated from the config so runs never overwrite each other and `ls results_*.json` is a self-documenting inventory, for example:

```
results_{MODEL_NAME}_variant-{FED_VARIANT}_alpha{DIRICHLET_ALPHA}_K{NUM_SHOTS}_T{NUM_ROUNDS}_pool{CLIENT_POOL_SIZE}_regime-{DATA_REGIME}_filter{0|1}_C{FILTER_C}_{DATASET}_seed{SEED}_order-{ORDER_STRATEGY}.json
```

<!-- VERIFY: confirm the exact field order in the filename against main.py; the
     committed result filenames in results/ are the ground truth. -->

---

## The lifecycle of a single run

For a reported (canonical, filtered) run:

```bash
FED_ICL_REGIME=canonical_full FED_ICL_FILTER=1 \
FED_ICL_MODEL=llama3 FED_ICL_ALPHA=0.5 FED_ICL_K=3 \
FED_ICL_FILTER_C=3 FED_ICL_SEED=42 FED_ICL_ORDER=original python main.py
```

1. `config.py` reads the environment, falling back to defaults for anything unset.
2. `data.py` loads the dataset, samples the client pool, and Dirichlet-partitions it across three clients; the server queries and the class-balanced held-out set (n=1000, `EVAL_SEED`) are drawn from the test split.
3. `llm.py` checks Ollama is reachable and the model is loaded.
4. `main.py` runs the zero-shot baseline on the held-out set.
5. `main.py` runs the local-only baseline through the configured select-order-filter pipeline, scored on the held-out set.
6. `main.py` instantiates the server (random initial context) and the clients.
7. For each of T=6 rounds: each client relabels its local data, predicts the server queries, and the server aggregates by majority vote.
8. `main.py` runs the held-out evaluation with the final global context through the same pipeline.
9. The summary prints, the JSON is written, the program exits.

Run time depends on model, pool, and filtering. As a rough guide, a single run is on the order of 10 to 20 minutes on an NVIDIA L4 and a few hours on a MacBook. The reported study is five seeds across three models and several conditions, so a full re-run is a multi-day job; the committed `results/` exists so verification does not require it.

---

## Things that look like bugs but are not

- **Round 0 accuracy near chance.** The global context is deliberately randomly initialised; learning happens in rounds 1 through T.
- **Very uneven client sizes.** With small α the Dirichlet partition can give one client several times more data than another. This is the heterogeneity the protocol is meant to handle.
- **Local-only accuracy near zero-shot.** With short inputs, K=3 demonstrations sometimes add little beyond the task instruction. A property of the data, not a bug.
- **The parse fallback returns the first label.** A deliberate, logged choice, so the fallback rate can be reported alongside accuracy; conditions above 10% are flagged.
- **All randomness flows from `SEED`, except evaluation.** Partition-side randomness is seeded from `SEED`; the held-out set is seeded separately by `EVAL_SEED` so it stays fixed across partition seeds.

---

## Design decisions that shaped the results

Three corrections during the project moved the numbers materially and are worth knowing when reading old result files:

- **Embedding selection became the default.** Early runs used lexical overlap; the reported pipeline uses kNN in paraphrase-MiniLM-L6-v2 space, the paper-faithful method. Lexical and random remain as ablation arms.
- **The residual-allocation fix** (above) removed a bias in the local-only baseline.
- **The held-out, decoupled, class-balanced evaluation** replaced an earlier n=100 set coupled to the partition seed, and federation gain was redefined as a held-out comparison rather than the in-loop shared-query number, which had overstated generalisation by about 9 percentage points.

All reported findings are five-seed (seeds 3, 7, 13, 42, 99); single-seed runs were shown during the project to be misleading in both directions.
