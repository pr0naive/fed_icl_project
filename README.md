# Fed-ICL: Demonstration Ordering in Federated In-Context Learning

Code and results for an MSc dissertation at the University of Exeter,
*Collaborative In-Context Learning in Federated Settings: The Effect of
In-Context Example Ordering under Data Heterogeneity*. The project builds
a faithful implementation of the Fed-ICL pipeline of Wang et al. (ICML
2025) and extends it to study how the **ordering** of in-context
demonstrations interacts with client **data heterogeneity**, evaluated on
held-out data across three open models and five random seeds.

The primary task is AG News four-class topic classification, with DBpedia
(14-class) as a difficulty extension. An MMLU extension is in progress on
the `mmlu-extension` branch and is not part of the results reported here.

**Supervisor**: Dr. Rui Jin (R.Jin1@exeter.ac.uk)
**Paper replicated**: Wang et al., *Federated In-Context Learning:
Iterative Refinement for Improved Answer Quality*, ICML 2025
(arXiv:2506.07440).

## Headline findings

All gains below are **held-out** accuracy (fixed, class-balanced, n=1000)
minus the **filtered local-only** baseline on the same evaluation set,
averaged over five seeds. Both conditions use the identical select-and-order
pipeline, so the comparison isolates federation.

- Federation does not reliably beat a strong local baseline. On AG News at
  the canonical filtered setup it helps in 6 of 15 model-by-seed conditions
  and hurts in 9; per-model mean held-out gain is phi3 -1.4pp, mistral
  -2.5pp, llama3 +0.9pp.
- The partition seed is the dominant source of variance. At C=10, mistral
  spans +6.3 to -10.6pp across seeds, far larger than any effect of model
  or filter breadth.
- Demonstration ordering has no reliable effect across three models and
  five seeds. A small most-similar-last (recency) lean appears on the
  similarity axis for mistral and phi3 but reverses on individual seeds
  and is within seed noise.
- Ordering does not interact with heterogeneity. The ascending-minus-
  descending gap does not systematically change with the Dirichlet alpha.
- Single-seed runs were misleading in both directions: an apparent
  federation gain and an apparent ordering effect both dissolved once
  multi-seeded. Five-seed discipline is the methodological backbone of the
  project, and the honest result is a rigorously characterised null.

## Repository structure

```
fed_icl_project/
├── config.py  data.py  federation.py  llm.py  main.py   # the pipeline
├── results/          # per-seed result JSONs + all_results_flat.csv
│   └── archive/      # superseded / pre-protocol runs, kept for provenance
├── logs/             # pilot run logs
├── docs/             # walkthrough, glossary, lab notebook, checklist
├── README.md  REPRODUCE.md  requirements.txt  .gitignore
```

Source modules stay at the repository root because their imports are flat
(`from config import ...`) and `main.py` writes output relative to the
working directory.

## Documentation map

| File | What it covers |
|---|---|
| [`docs/code_walkthrough.md`](docs/code_walkthrough.md) | File-by-file tour of the codebase and the lifecycle of a single run. |
| [`docs/glossary.md`](docs/glossary.md) | Definitions for every term and code variable, plus the paper-notation-to-code mapping. |
| [`docs/lab_notebook.md`](docs/lab_notebook.md) | Chronological research log: decisions, results, corrections. Read as a dated history, not as current documentation. |
| [`docs/methodology_checklist.md`](docs/methodology_checklist.md) | Pre-flight checks to run before launching an experiment. |

## Quick start

```bash
# 1. Ollama running with the models pulled
ollama serve &
ollama pull mistral   # and llama3, phi3

# 2. Python environment (3.10-3.12; 3.14 breaks bundled pip)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run. NOTE: bare defaults are the canonical, UNFILTERED regime, which
#    is NOT the setup the dissertation reports. See "Reproducing" below.
python main.py
```

The output JSON is written to the working directory, auto-named from the
run configuration so parallel runs cannot overwrite each other:

```
results_{model}_variant-{variant}_alpha{α}_K{clients}_T{rounds}_pool{pool}_regime-{regime}_filter{0/1}_C{C}_{dataset}_seed{seed}_order-{order}.json
```

Move finished runs into `results/` (tracked); stray result files left at
the repo root are ignored by `.gitignore`.

## Configuring a run

Every parameter can be set in `config.py` or overridden from the shell via
its `FED_ICL_*` environment variable, so sweeps are shell loops rather than
source edits:

```bash
# One run with non-default settings
FED_ICL_MODEL=llama3 FED_ICL_ALPHA=0.05 FED_ICL_ORDER=similarity_ascending python main.py

# Sweep orderings at one alpha
for o in original similarity_ascending similarity_descending label_grouped label_alternating random_shuffle; do
  FED_ICL_ORDER=$o python main.py
done
```

| `config.py` | Env override | Default | Controls |
|---|---|---|---|
| `MODEL_NAME` | `FED_ICL_MODEL` | `mistral` | Ollama model (`mistral`, `llama3`, `phi3`). |
| `DATASET` | `FED_ICL_DATASET` | `agnews` | `agnews` or `dbpedia` (14-class; `canonical_full` only). |
| `FED_VARIANT` | `FED_ICL_VARIANT` | `fed_icl` | `fed_icl` (union pool) or `fed_icl_free` (relabelled only; falls back to local data if that pool is empty). |
| `DATA_REGIME` | `FED_ICL_REGIME` | `canonical` | Reported runs use `canonical_full` (see Reproducing). |
| `NUM_CLIENTS` | `FED_ICL_CLIENTS` | `3` | Simulated clients (N). |
| `NUM_ROUNDS` | `FED_ICL_ROUNDS` | `6` | Federation rounds (T). |
| `DIRICHLET_ALPHA` | `FED_ICL_ALPHA` | `0.5` | Heterogeneity; smaller is more non-IID. Swept over {0.05, 0.5, 10.0}. |
| `NUM_SHOTS` | `FED_ICL_K` | `3` | Demonstrations per prompt (K). |
| `SELECTION_STRATEGY` | `FED_ICL_SEL` | `similarity_embedding` | kNN with paraphrase-MiniLM-L6-v2. `similarity` (lexical) and `random` are ablation arms. |
| `ORDER_STRATEGY` | `FED_ICL_ORDER` | `original` | One of six orderings (below). |
| `FILTER_LOCAL_DATA` | `FED_ICL_FILTER` | `0` (off) | Set to `1` to apply the paper's kNN filter. Reported runs use `1`. |
| `FILTER_C` | `FED_ICL_FILTER_C` | `NUM_SHOTS` (3) | Filter breadth (C in Algorithm 2). Swept over {3, 5, 10}. |
| `CLIENT_POOL_SIZE` | `FED_ICL_POOL_SIZE` | `250` | Total client pool across all clients (not per client). |
| `NUM_SERVER_QUERIES` | `FED_ICL_Q` | `100` | Shared server query set. |
| `EVAL_SIZE` | `FED_ICL_EVAL` | `1000` | Held-out evaluation set, class-balanced. |
| `EVAL_SEED` | `FED_ICL_EVAL_SEED` | `12345` | Fixed and decoupled from `SEED`, so the eval sample never varies with the partition. |
| `SEED` | `FED_ICL_SEED` | `42` | Partition seed. The five-seed grid uses {3, 7, 13, 42, 99}. |
| `TEMPERATURE` | (in-file) | `0.0` | Deterministic decoding. |

Secondary flags exist for the alternative regimes and dataset capping:
`FED_ICL_STRAT_QUERIES`, `FED_ICL_POOL_CAP`, `FED_ICL_SPLIT_SEED`,
`FED_ICL_TEST_FRAC`, `FED_ICL_QUERY_SEED`, `FED_ICL_HOST`.

**Reproducing the reported (canonical, filtered) setup.** The in-file
defaults are the older unfiltered `canonical` regime, so bare `python
main.py` does not reproduce the dissertation runs. The reported runs set
the regime and turn filtering on:

```bash
FED_ICL_REGIME=canonical_full FED_ICL_FILTER=1 \
FED_ICL_MODEL=mistral FED_ICL_ALPHA=0.5 FED_ICL_SEED=42 python main.py
```

The exact settings behind any reported run are recorded in that result
file's `config` block, so the reliable way to reproduce a specific cell is
to match its config. Two things worth reconciling in `config.py` itself:
the default `DATA_REGIME` is `canonical` while every reported run is
`canonical_full` (and the inline comment documents only `canonical` and
`split8020`, not `canonical_full`), and `FILTER_LOCAL_DATA` defaults off
while every reported run has it on. Either change the defaults to match the
reported setup or state the override set explicitly, so a marker running
defaults does not get different numbers.

## What the code does

`main.py` runs, on the same data split:

1. **Zero-shot baseline.** No demonstrations. How well the model does the
   task unaided. Deterministic at temperature 0, computed once per model.
2. **Local-only baseline.** A single client using only its own pool, no
   federation, through the same select-and-order-and-filter pipeline as
   Fed-ICL, then scored on the held-out set. The decisive test of whether
   federation earns its complexity.
3. **Fed-ICL.** The iterative protocol (Wang et al., 2025): clients
   relabel local data using the shared global context, predict the shared
   server queries, the server aggregates by majority vote, repeat for T
   rounds. The default `fed_icl` variant conditions on the union of
   original and relabelled data; the `fed_icl_free` variant uses the
   relabelled pool only.
4. **Held-out evaluation.** The final global context is the ICL pool for a
   fixed, class-balanced held-out set (n=1000) drawn from the dataset's
   test split with `EVAL_SEED`, never seen during federation. This is the
   headline metric.

Each run writes a JSON with the full config, the round-by-round trajectory
on the shared queries, the held-out accuracy, the baselines, and the parse
fallback rate.

## Ordering strategies (the dissertation focus)

Six orderings are implemented in `federation.py`. Selection and ordering
are decoupled by design: `select_examples` picks K demonstrations,
`order_examples` arranges them, and the selected set is shuffled before
ordering so order is the only thing that varies. Similarity ordering uses
lexical word overlap, distinct from the embedding cosine used for selection.

| `ORDER_STRATEGY` | Behaviour |
|---|---|
| `original` | Selection order after the pre-ordering shuffle (control). |
| `similarity_ascending` | Least similar first; most similar closest to the query. |
| `similarity_descending` | Most similar first. |
| `label_grouped` | Sorted by label, each class consecutive. |
| `label_alternating` | Round-robin across labels. |
| `random_shuffle` | Fresh random order each call; quantifies ordering variance. |

## Evaluation protocol

The evaluation went through two corrections that the reported numbers
depend on, both documented in the lab notebook:

- **Held-out, decoupled, class-balanced.** The eval set is drawn from the
  dataset's canonical test split with a fixed `EVAL_SEED=12345`,
  independent of the partition seed, at n=1000 with 250 examples per class.
  Earlier runs coupled the eval sample to the partition seed at n=100,
  which conflated partition structure with evaluation-sampling luck.
- **Federation gain is a held-out comparison.** Gain is defined as
  held-out federated accuracy minus held-out local-only accuracy, both at
  n=1000. An earlier definition subtracted the local-only baseline from the
  in-loop shared-query accuracy (R6), which overstated generalisation by
  about 9pp because R6 is an in-sample fit number.

## Reproducing the dissertation results

The `results/` folder holds the result JSON for every run behind the
reported tables, and each file records its full configuration and numbers.
Verification is a single-cell re-run: run one condition through `main.py`
and confirm the JSON it produces matches the committed one for the same
configuration. The reported runs use the `canonical_full` regime with
filtering on, which are not the in-file defaults, so overrides are needed:

```bash
FED_ICL_REGIME=canonical_full FED_ICL_FILTER=1 \
FED_ICL_MODEL=mistral FED_ICL_ALPHA=0.5 FED_ICL_K=3 \
FED_ICL_FILTER_C=3 FED_ICL_SEED=42 FED_ICL_ORDER=original \
python main.py
```

See [`REPRODUCE.md`](REPRODUCE.md) for prerequisites (environment, Ollama,
models) and how to reproduce other cells by matching a committed filename.

## Paper-to-code mapping

| Paper | Code | Meaning |
|---|---|---|
| N | `NUM_CLIENTS` | Clients. |
| K | `NUM_SHOTS` | Demonstrations per prompt. |
| T | `NUM_ROUNDS` | Federation rounds. |
| α | `DIRICHLET_ALPHA` | Dirichlet concentration. |
| C | `FILTER_C` | Filter breadth (Algorithm 2). |
| n | `EVAL_SIZE` | Held-out evaluation size. |

Full Algorithm 1 mapping is in `docs/code_walkthrough.md`.

## Reproducibility notes

- Every result file records the full config and the parse fallback rate.
- Seeds are fixed and threaded through partition, selection, and ordering;
  `EVAL_SEED` is separate so the held-out sample is constant across runs.
- All reported findings are five-seed (seeds 3, 7, 13, 42, 99).
- Output filenames encode the config so runs cannot overwrite each other.
- `docs/methodology_checklist.md` is the pre-flight checklist.

## Project status

All experiments reported in the dissertation are complete. The results in
`results/` are the five-seed, held-out, canonical-filtered runs behind the
tables and figures. `results/archive/` holds superseded and pre-protocol
runs, kept for provenance. The MMLU extension continues on the
`mmlu-extension` branch and is not part of the submitted results.

## References

- Wang, R., et al. *Federated In-Context Learning: Iterative Refinement for Improved Answer Quality.* ICML 2025 (arXiv:2506.07440).
- McMahan, H. B., et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
- Liu, J., et al. *What Makes Good In-Context Examples for GPT-3?* DeeLIO 2022 (KATE).
- Lu, Y., et al. *Fantastically Ordered Prompts and Where to Find Them.* ACL 2022.
- Zhao, Z., et al. *Calibrate Before Use: Improving Few-Shot Performance of Language Models.* ICML 2021.
- Brown, T., et al. *Language Models are Few-Shot Learners.* NeurIPS 2020.
- Min, S., et al. *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* EMNLP 2022.
- Hsu, T.-M. H., Qi, H., Brown, M. *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification.* arXiv:1909.06335, 2019.
- Li, T., et al. *Federated Optimization in Heterogeneous Networks.* MLSys 2020.
- Zhang, X., Zhao, J., LeCun, Y. *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015 (AG News).
