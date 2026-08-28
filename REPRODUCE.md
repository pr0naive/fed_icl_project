# Reproducing the results

The `results/` folder contains the result JSON for every run behind the
reported tables, and each file records its full configuration and its
numbers. To verify the work, re-run a single cell of the pipeline and check
that the JSON it produces matches the committed one for the same
configuration.

> Note on scope. The reported findings are five-seed grids (seeds 3, 7, 13,
> 42, 99) across three models and several conditions, on the order of a
> hundred runs at roughly 10-20 minutes each on an NVIDIA L4 and 2-3 hours
> each on a MacBook. Re-running the full study is a multi-day job. The
> committed `results/` exists so verification does not require it: you
> reproduce one cell and compare, rather than re-running everything.

---

## 1. Prerequisites

### Clone the repository

```bash
git clone https://github.com/pr0naive/fed_icl_project.git
cd fed_icl_project
```

### Python environment

Use Python 3.10 to 3.12. Python 3.14 ships a broken bundled pip inside
virtual environments and will fail on install.

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

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
ollama pull mistral
ollama pull llama3
ollama pull phi3
```

The three models together are several GB on disk. Runs use deterministic
decoding (`TEMPERATURE = 0.0`). A GPU is strongly preferred: on an NVIDIA
L4 a single run is roughly 10-20 minutes depending on model; on a MacBook
it is 2-3 hours.

<!-- TEST PASS: confirm the exact Ollama version and the pulled model
     sizes you used, and drop them in here so the assessor matches your
     environment. -->

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

Then compare the JSON it produces against the committed file in `results/`
for the same configuration: the `config` block should match, and the
held-out accuracy and baselines should reproduce.

<!-- TEST PASS: run this and confirm it produces a JSON whose config block
     and numbers match the committed canonical_full mistral seed-42
     original-order file. Add any flags your launch scripts set that are
     missing here (e.g. FED_ICL_STRAT_QUERIES, FED_ICL_POOL_CAP). The
     committed result file's config block is the ground truth. -->

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
