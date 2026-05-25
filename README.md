# Fed-ICL Replication

Replication and extension of *Federated In-Context Learning* (Wang et al., ICML 2025) for an MSc dissertation at the University of Exeter. The dissertation investigates how the **ordering of in-context examples** affects convergence and final accuracy in federated settings, and whether that effect interacts with data heterogeneity.

The codebase runs locally on a MacBook Air M4 via Ollama, on a four-class news topic classification task (AG News, loaded from HuggingFace).

**Supervisor**: Dr. Rui Jin (R.Jin1@exeter.ac.uk)
**Repository**: <https://github.com/pr0naive/fed_icl_project>

## Documentation map

This README is a launchpad. Detailed documentation lives in dedicated files so each can be updated independently as the project evolves.

| File | What it covers |
|---|---|
| [`code_walkthrough.md`](code_walkthrough.md) | File-by-file tour of the codebase, design decisions behind each module, and the lifecycle of a single run. |
| [`glossary.md`](glossary.md) | One-line definitions for every term and code variable, plus the paper-notation-to-code mapping. |
| [`lab_notebook.md`](lab_notebook.md) | Chronological log of decisions, results, and reasoning. The canonical source for current experimental numbers. |
| [`methodology_checklist.md`](methodology_checklist.md) | Pre-flight checks to run before launching an experiment. |

For headline experimental numbers, see the most recent entries in `lab_notebook.md`. They are not duplicated here because they change.

## Quick start

```bash
# 1. Ollama running with the model pulled
ollama serve &
ollama pull llama3   # or mistral, phi

# 2. Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run with defaults
python main.py
```

A default run takes roughly 2-3 hours on an M-series Mac. The output JSON is written to `results_{model}_alpha{α}_K{K}_T{T}_seed{S}_order-{order}.json` in the repo root; the filename is auto-generated from the run configuration so parallel runs cannot overwrite each other.

## Configuring a run

Every experimental parameter can be set either by editing `config.py` or by overriding it from the shell. The override pattern means sweeps are shell loops, not source edits:

```bash
# Single run with non-default settings
FED_ICL_MODEL=mistral FED_ICL_ALPHA=0.05 FED_ICL_K=5 \
FED_ICL_ORDER=label_grouped FED_ICL_SEED=42 python main.py

# Sweep over seeds
for s in 41 42 43; do
  FED_ICL_SEED=$s python main.py
done

# Sweep over orderings at one alpha
for o in original similarity_ascending similarity_descending label_grouped label_alternating random_shuffle; do
  FED_ICL_ORDER=$o python main.py
done
```

| Parameter | Env override | Default | Controls |
|---|---|---|---|
| `MODEL_NAME` | `FED_ICL_MODEL` | `mistral` | Ollama model identifier. |
| `NUM_CLIENTS` | `FED_ICL_CLIENTS` | `3` | Number of simulated clients. |
| `NUM_ROUNDS` | `FED_ICL_ROUNDS` | `6` | Federation rounds (T in the paper). |
| `DIRICHLET_ALPHA` | `FED_ICL_ALPHA` | `0.5` | Heterogeneity. Smaller = more non-IID. |
| `NUM_SHOTS` | `FED_ICL_K` | `3` | Demonstrations per prompt (K in the paper). |
| `SELECTION_STRATEGY` | `FED_ICL_SEL` | `random` | `random` or `similarity`. |
| `ORDER_STRATEGY` | `FED_ICL_ORDER` | `original` | One of six orderings (see below). |
| `NUM_SERVER_QUERIES` | `FED_ICL_Q` | `100` | Size of the shared server query set. |
| `EVAL_SIZE` | `FED_ICL_EVAL` | `100` | Held-out evaluation pool, drawn from the AG News test split. |
| `SEED` | `FED_ICL_SEED` | `42` | Random seed. Three seeds per cell is the methodology target. |

For the meaning of each parameter, see `glossary.md`. For the design reasoning behind the defaults, see `concepts.md`.

## What the code does

`main.py` runs four things in order, all on the same data split:

1. **Zero-shot baseline.** No demonstrations in the prompt. Establishes how well the LLM solves the task without any in-context guidance.
2. **Local-only baseline.** Client 0 with its own local pool only, no federation, using the same select-and-order pipeline as Fed-ICL. The decisive test of whether federation justifies its complexity.
3. **Fed-ICL.** The full Fed-ICL-Free protocol (Wang et al., 2025): clients relabel local data using the global context, predict shared server queries, the server aggregates by majority vote, repeat T times.
4. **Held-out evaluation.** The final global context is used as the example pool for an evaluation set drawn from the AG News test split, with the same selection and ordering rules.

Each run writes a JSON containing the full config, the round-by-round accuracy trajectory, the baselines, the held-out accuracy, and the parse fallback rate.

## Ordering strategies (the dissertation focus)

The research question is whether the order of in-context examples affects federated convergence, and whether that effect interacts with data heterogeneity. Six orderings are implemented in `federation.py`:

| `ORDER_STRATEGY` | Behaviour |
|---|---|
| `original` | Examples in selection order (control after the post-selection shuffle fix). |
| `similarity_ascending` | Least similar first; most similar example closest to the query. |
| `similarity_descending` | Most similar first. |
| `label_grouped` | Examples sorted by label, so all of one class appear consecutively. |
| `label_alternating` | Round-robin across labels. |
| `random_shuffle` | Fresh random order each call; quantifies ordering-induced variance. |

The selection rule and the ordering rule are decoupled by design. `select_examples` picks K examples; `order_examples` arranges them. Both are methods on `FedICLClient` in `federation.py`. See `concepts.md` section 4 for the rationale.

## Paper-to-code mapping

| Paper notation | Code identifier | Meaning |
|---|---|---|
| N | `NUM_CLIENTS` | Number of clients. |
| K | `NUM_SHOTS` | Demonstrations per prompt. |
| T | `NUM_ROUNDS` | Federation rounds. |
| α | `DIRICHLET_ALPHA` | Dirichlet concentration. |
| n | `EVAL_SIZE` | Evaluation pool size. |

Full Algorithm 1 mapping, including the relabel and aggregation steps, is in `code_walkthrough.md`.

## Reproducibility

- Every result file records the full config (model, α, K, T, selection, ordering, seed, evaluation pool size) and the parse fallback rate from that run.
- Random seeds are fixed and threaded explicitly through the data partition, selection, and ordering steps.
- Output filenames are auto-generated from the config so two runs with different parameters cannot overwrite each other.
- A `methodology_checklist.md` lists the pre-flight checks (Ollama up, model pulled, expected wall-clock, free disk for the JSON) to run before each experiment.

## Project status

The current experimental focus is establishing reference numbers across base models (llama3, mistral, phi) on the tightened pipeline before launching the full ordering and heterogeneity sweep. See `lab_notebook.md` for the latest results. The dissertation deadline is 8 August 2026.

## References

- Wang, R., et al. *Federated In-Context Learning: Iterative Refinement for Improved Answer Quality.* ICML 2025.
- McMahan, H. B., et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
- Liu, J., et al. *What Makes Good In-Context Examples for GPT-3?* DeeLIO 2022 (KATE).
- Lu, Y., et al. *Fantastically Ordered Prompts and Where to Find Them.* ACL 2022.
- Zhao, Z., et al. *Calibrate Before Use: Improving Few-Shot Performance of Language Models.* ICML 2021.
- Brown, T., et al. *Language Models are Few-Shot Learners.* NeurIPS 2020.
- Min, S., et al. *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?* EMNLP 2022.
- Hsu, T.-M. H., Qi, H., Brown, M. *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification.* arXiv:1909.06335, 2019.
- Li, T., et al. *Federated Optimization in Heterogeneous Networks.* MLSys 2020.
- Zhang, X., Zhao, J., LeCun, Y. *Character-level Convolutional Networks for Text Classification.* NeurIPS 2015 (AG News dataset).