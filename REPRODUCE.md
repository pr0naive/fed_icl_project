# Reproducing the results

The `results/` folder contains the result JSON for every run behind the
reported tables, and each file records its full configuration and its
numbers. To verify the work, re-run a single cell of the pipeline and check
that the JSON it produces matches the committed one for the same
configuration. This has been verified end to end (see section 2).

> Note on scope. The reported findings are five-seed grids (seeds 3, 7, 13,
> 42, 99) across three models and several conditions, on the order of a
> hundred runs at roughly 10 to 20 minutes each on an NVIDIA L4. Re-running
> the full study is a multi-day job. The committed `results/` exists so
> verification does not require it: you reproduce one cell and compare,
> rather than re-running everything.

---

## 1. Prerequisites

### Clone the repository

```bash
git clone https://github.com/pr0naive/fed_icl_project.git
cd fed_icl_project
```

### Python environment

Use Python 3.10 to 3.13. Python 3.14 ships a broken bundled pip inside
virtual environments and will fail on install, so if your default `python3`
is 3.14 (check with `python3 --version`), create the venv with an explicit
older interpreter:

```bash
python3.12 -m venv venv      # or python3.11 / python3.13; anything 3.10-3.13
source venv/bin/activate
python --version             # confirm it is 3.10-3.13
pip install -r requirements.txt
```

The verified run used Python 3.13. `requirements.txt` installs current
compatible versions, and a fresh install of the latest packages reproduced
the committed numbers exactly in testing. For provenance, the exact package
set from the verified run environment (Linux with CUDA) is recorded in
`requirements.lock.txt`; note that file is platform-specific and is a record
rather than a cross-platform install target.

### Data

No manual data download is needed. The datasets load through the
HuggingFace `datasets` library on first run (`fancyzhx/ag_news`, and
DBpedia for the extension), both public, no token required. The first run
downloads and caches them, so it needs internet access and a little disk
for the HuggingFace cache.

### Ollama and models

Install Ollama, start it, and pull the three models:

```bash
ollama serve &
ollama pull mistral      # 4.4 GB
ollama pull llama3       # 4.7 GB
ollama pull phi3         # 2.2 GB
```

Runs use deterministic decoding (`TEMPERATURE = 0.0`). A GPU is strongly
preferred: on an NVIDIA L4 a single run is roughly 10 to 20 minutes; on a
CPU-only machine it is several hours. The verified run used these three
models served by a user-launched Ollama.

---

## 2. Reproduce and verify one cell

Each `python main.py` writes one result JSON to the working directory,
auto-named from the run configuration. The reported experiments use the
`canonical_full` regime with filtering on, which are **not** the in-file
defaults, so the environment overrides below are required.

Run one condition:

```bash
FED_ICL_REGIME=canonical_full FED_ICL_FILTER=1 \
FED_ICL_MODEL=mistral FED_ICL_ALPHA=0.5 FED_ICL_K=3 \
FED_ICL_FILTER_C=3 FED_ICL_SEED=42 FED_ICL_ORDER=original \
python main.py
```

This exact command, run from a fresh clone, reproduces the committed result
to the decimal: held-out accuracy **71.5%**, local-only baseline **68.2%**,
federation gain **+3.3pp**, matching
`results/results_mistral_variant-fed_icl_alpha0.5_K3_T6_pool250_regime-canonical_full_filter1_C3_agnews_seed42_order-original.json`.
If your run produces those numbers, the pipeline is verified. To confirm
against the committed file directly:

```bash
python -c "import json; \
new=json.load(open('results_mistral_variant-fed_icl_alpha0.5_K3_T6_pool250_regime-canonical_full_filter1_C3_agnews_seed42_order-original.json')); \
old=json.load(open('results/results_mistral_variant-fed_icl_alpha0.5_K3_T6_pool250_regime-canonical_full_filter1_C3_agnews_seed42_order-original.json')); \
print('held-out new/old:', new['eval_accuracy'], old['eval_accuracy']); \
print('local    new/old:', new['baselines']['local_only'], old['baselines']['local_only'])"
```

To reproduce a different cell, change the overrides to match a committed
filename, which encodes regime, filter, C, alpha, dataset, variant, seed,
and order. The other experimental arms vary `FED_ICL_FILTER_C` over {3, 5,
10} for filter breadth, `FED_ICL_ALPHA` over {0.05, 0.5, 10.0} for
heterogeneity, `FED_ICL_DATASET=dbpedia` for the DBpedia extension,
`FED_ICL_ORDER` over the six orderings for the ordering study, and
`FED_ICL_VARIANT=fed_icl_free` for the variant comparison.

---

## Notes

- Result filenames encode the full configuration, so runs never overwrite
  each other and every file is self-describing.
- The evaluation set is fixed and class-balanced (n=1000, `EVAL_SEED=12345`)
  and decoupled from the partition seed, so held-out numbers are comparable
  across seeds.
- Reported findings are five-seed; single-seed runs are not representative
  and were shown during the project to be misleading in both directions.
