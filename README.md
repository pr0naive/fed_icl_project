# Fed-ICL Replication

Replication of *Federated In-Context Learning* (Wang et al., ICML 2025) for an
MSc dissertation investigating how the **ordering of in-context examples**
affects convergence and final accuracy in federated settings, and whether that
effect interacts with data heterogeneity.

Runs on a MacBook Air M4 via Ollama with a 4-class news topic classification
task (world / sports / business / science) loaded from HuggingFace AG News.

## Status

This is a research codebase, not a finished product. The most recent baseline
on AG News (n=100, α=0.5, K=3 clients, llama3) shows:

- Zero-shot baseline: 72%
- Local-only baseline: 80%
- Fed-ICL (after 6 rounds): 79%
- Held-out test set: 80%
- Federation gain over random init: +59 percentage points

Federation exceeds zero-shot by ~7pp and roughly matches local-only at this
heterogeneity level. The next planned experiments are a heterogeneity sweep
(α ∈ {0.05, 0.5, 10.0}) and an ordering sweep across the five strategies
implemented in `federation.py`.

A previous run on a synthetic 152-example dataset showed task saturation
(zero-shot at 100% — no headroom for federation). That result is preserved as
`results_synthetic_saturation_evidence.json` for reference.

## Quick start

```bash
# 1. Make sure Ollama is running and the model is pulled
ollama serve &
ollama pull llama3

# 2. Set up Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run
python main.py
```

A full default run takes roughly 2–3 hours on an M-series Mac. Results land
in `results.json`.

## What it does

`main.py` runs three things in order, all on the same data split:

1. **Zero-shot baseline** — no examples in the prompt. Establishes how well
   the LLM solves the task without any in-context guidance.
2. **Local-only baseline** — one client, no federation. Establishes whether
   federation adds anything over a single client with its own data.
3. **Fed-ICL** — full Algorithm 1 from Wang et al. (2025): K=3 simulated
   clients, T=6 rounds of relabelling and majority-vote aggregation.

After Fed-ICL completes, the final global context is used as the example pool
for a held-out evaluation set.

## Files

| File | Role |
|---|---|
| `config.py` | All experimental knobs in one place. Change values here, run again. |
| `data.py` | AG News loader, label space, Dirichlet partitioning, train/server/eval split. |
| `llm.py` | Ollama interface: prompt construction, model call, label parsing, readiness check. |
| `federation.py` | `FedICLClient` and `FedICLServer` classes. The five ordering strategies live in `FedICLClient.order_examples`. |
| `main.py` | Runs baselines and Fed-ICL, prints per-round accuracy, writes `results.json`. |
| `lab_notebook.md` | Chronological log of decisions, results, and reasoning. |
| `methodology_checklist.md` | Pre-flight check for every experimental run. |
| `requirements.txt` | numpy, requests, datasets. |

## Algorithm 1 ↔ code mapping

| Paper line | Code |
|---|---|
| For t = 1 to T | `for t in range(1, NUM_ROUNDS + 1)` in `main.py` |
| Lines 3–4 (relabel local data) | `client.relabel_local_data(global_context)` |
| Lines 5–6 (predict server queries) | `client.predict_server_queries(server_queries, global_context)` |
| Line 7 (majority-vote aggregation) | `server.aggregate_predictions(all_client_predictions)` |
| Line 8 (evaluate global context) | `server.evaluate_context()` |

## Configuration

All settings live in `config.py`:

| Parameter | Default | What it controls |
|---|---|---|
| `MODEL_NAME` | `"llama3"` | Ollama model. Also tested: mistral, phi. |
| `NUM_CLIENTS` | `3` | K — simulated clients. |
| `NUM_ROUNDS` | `6` | T — federation rounds. |
| `DIRICHLET_ALPHA` | `0.5` | Heterogeneity. 0.05 = extreme non-IID, 10.0 = nearly IID. |
| `NUM_SHOTS` | `3` | In-context examples per prompt. |
| `SELECTION_STRATEGY` | `"random"` | Or `"similarity"` (word-overlap-based). |
| `ORDER_STRATEGY` | `"original"` | The dissertation focus — see below. |
| `NUM_SERVER_QUERIES` | `100` | Size of the global context C. |
| `EVAL_SIZE` | `100` | Held-out test set, never seen during federation. |
| `SEED` | `42` | Fixed for reproducibility. Vary later for robustness checks. |

## Ordering strategies (the dissertation focus)

The research question is whether the order of in-context examples affects
federated convergence, and whether that interaction depends on data
heterogeneity. Five orderings are implemented in `FedICLClient.order_examples`:

| `ORDER_STRATEGY` | What it does |
|---|---|
| `"original"` | Examples in selection order (control). |
| `"similarity_ascending"` | Least similar first; most similar example sits closest to the query. |
| `"similarity_descending"` | Most similar first; tests whether early examples set context. |
| `"label_grouped"` | All world examples, then all sports, etc. Tests class clustering. |
| `"label_alternating"` | Round-robin across labels: world, sports, business, science, ... |
| `"random_shuffle"` | Different order each run; quantifies the variance ordering introduces. |

## Running experiment variants

Change one line in `config.py`, run `python main.py`, then rename
`results.json` before the next run so it isn't overwritten.

```bash
# After each run:
mv results.json results_alpha005.json   # or whatever describes that run
```

A full first-pass sweep is roughly:

```text
1. DIRICHLET_ALPHA = 0.05, 0.5, 10.0    # heterogeneity sweep    (3 runs)
2. NUM_SHOTS       = 1, 3, 5            # shot-count sweep       (3 runs)
3. ORDER_STRATEGY  = all five           # ordering sweep         (5 runs)
4. (best ordering) × alpha 0.05, 10.0   # interaction probe      (2 runs)
```

## Roadmap

Short-term:
- Heterogeneity sweep across α ∈ {0.05, 0.5, 10.0}
- Ordering comparison across all five strategies
- Interaction analysis (best ordering at each α extreme)
- GPU server access for faster iteration and smaller-model probes

Medium-term:
- SBERT-based similarity selection (KATE-style) as an alternative to word overlap
- Heterogeneous LLMs across clients (already supported via the `model` argument
  on `FedICLClient`)
- Multiple seeds per condition once the methodology is locked in

## References

- Wang, X., Wu, Z., Sun, L., et al. *Federated In-Context Learning.* ICML 2025.
- McMahan, H. B., et al. *Communication-Efficient Learning of Deep Networks
  from Decentralized Data.* AISTATS 2017.
- Liu, J., et al. *What Makes Good In-Context Examples for GPT-3?* DeeLIO
  2022 (KATE).
- Brown, T., et al. *Language Models are Few-Shot Learners.* NeurIPS 2020.
- Min, S., et al. *Rethinking the Role of Demonstrations: What Makes
  In-Context Learning Work?* EMNLP 2022.
