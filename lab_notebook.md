# Fed-ICL Lab Notebook

A running log of decisions, results, and reasoning for the dissertation
"Collaborative In-Context Learning in Federated Settings".

Newest entries at the bottom.

---

## Project framing (early April 2026)

**Research question** (agreed with Dr. Jin):
*How does the ordering of in-context examples affect convergence and final
accuracy in federated in-context learning, and does this effect interact
with data heterogeneity?*

Direction chosen: optimisation, not vulnerability. Centred on example
selection strategies; what to select, ordering, batch size.

Five papers read as foundation: Fed-ICL (Wang et al., ICML 2025),
FedAvg (McMahan et al., 2017), KATE / What Makes Good ICL Examples
(Liu et al., 2022), Language Models are Few-Shot Learners
(Brown et al., 2020), Rethinking the Role of Demonstrations
(Min et al., 2022).

Negative results agreed to be valid; rigour of analysis prioritised.

---

## Initial implementation (early April)

Built a 7-file Python codebase replicating Algorithm 1 from Wang et al.:
`config.py`, `data.py`, `llm.py`, `federation.py`, `main.py`,
`requirements.txt`, `README.md`.

Core components:
- Dirichlet partitioning across K=3 simulated clients
- Iterative rounds of client relabelling and server-side majority-vote
  aggregation
- Five ordering strategies in `federation.py` controlled via
  `ORDER_STRATEGY` in `config.py`: `original`, `similarity_ascending`,
  `similarity_descending`, `label_grouped`, `label_alternating`,
  `random_shuffle`

Run locally on M4 MacBook Air via Ollama (llama3, mistral, phi available).

Two early bugs: `LABEL_SPACE` ImportError (resolved by importing from
`data.py` rather than `config.py`); `ZeroDivisionError` when extreme
Dirichlet alpha values left a client with zero examples.

---

## First result: binary sentiment task, saturation (early April)

Ran the implementation on a binary sentiment classification task
(positive/negative).

**Result:** llama3 zero-shot accuracy was 95–100% from the start.
Federation could not improve on this, no headroom.

**Interpretation:** Task was too easy for llama3. Federation cannot
demonstrate value when the baseline is already at ceiling.

**Action taken:** Switched task to 4-class news topic classification
(world / sports / business / science) using a hand-written 152-example
synthetic dataset (38 per class).

**Why:** A 4-class task is harder than binary. The hypothesis was that
the increased class count would create the headroom federation needs to
demonstrate gain.

---

## Second result: synthetic 4-class, also saturated (early April)

Ran with the synthetic 152-example news dataset.

**Result:** Round 0 (random init): 16.7%. Round 1 onwards: 100%.
Held-out test set: 95%. Zero-shot baseline: 100%. Local-only: 100%.

**Interpretation:** llama3 also saturates the synthetic 4-class task.
Two factors compound: (1) llama3 is a strong instruction-tuned model
that handles news classification trivially, (2) the synthetic data was
hand-written with clear category signals, making it cleaner and easier
than realistic news.

This is preserved as `results_synthetic_saturation_evidence.json` in the
repo as evidence for the methodological observation that capable LLMs
saturate common ICL benchmarks.

**Action taken:** Prepared progress slides for Dr. Jin documenting the
saturation finding, then had to step away from the project for ~3 weeks
due to emergency travel.

---

## Resumption (28 April 2026)

Returned to the project after a 3-week gap. Reviewed Dr. Jin's 10 April
email, which gave guidance on five points:

1. Switch to a HuggingFace benchmark dataset
2. GPU access via mcrugcomp03.ex.ac.uk (SSH) or University GPU VMs
3. Fix random seed as constant for now; vary later for robustness
4. Focus next: switch dataset, establish solid baseline
5. Set up GitHub repo and share link

---

## GitHub setup (28 April)

Created private repository, then made public for ease of access on the
GPU server. Initial commit: codebase, README, .gitignore, and the
synthetic saturation evidence JSON.

`.gitignore` configured to exclude `venv/`, `__pycache__/`, `.DS_Store`,
`.continue/`, and `results*.json` , but with an explicit exception for
`results_synthetic_saturation_evidence.json`, which is preserved as
evidence of the saturation finding.

Required cleanup: initial `git add .` had captured `__pycache__/`,
`.DS_Store`, and `.continue/` files because the staging happened before
`.gitignore` was finalised. Resolved with `git rm -r --cached . && git add .`
to re-apply ignore rules.

Repo URL: `https://github.com/pr0naive/fed_icl_project`

---

## AG News dataset switch (28 April)

Replaced the synthetic 152-example dataset in `data.py` with HuggingFace
AG News (loaded via the `datasets` library). Changes:

- Added `_AG_NEWS_LABEL_MAP` to convert AG News integer labels (0-3) to
  the existing string label space (world/sports/business/science).
  AG News class 3 is officially "Sci/Tech" but mapped to "science" to
  preserve the existing prompt template.
- Added `_load_ag_news(num_examples, seed)` to load and deterministically
  subsample.
- Replaced the `RAW_DATA = [...]` literal with
  `RAW_DATA = _load_ag_news(num_examples=200, seed=SEED)`.

Everything else (`LABEL_SPACE`, `partition_data_dirichlet`,
`prepare_experiment`) unchanged.

**Why this dataset:**
- Standard benchmark in the ICL literature, comparable to other work
- Same 4-class structure as the synthetic data, minimal disruption
- Real news headlines are noisier and more ambiguous than hand-written
  examples, expected to provide headroom

---

## Environment fix: Python 3.14 → 3.12 (28 April)

Initial venv was created with Python 3.14 (newest stable). pip itself
was broken inside the venv: `ModuleNotFoundError: No module named
'pip._vendor.rich.markup'`. Cause: bundled pip in 3.14 venv missing
internal modules.

**Action taken:** Rebuilt venv with Python 3.12 (more mature for ML
tooling). Resolved.

**Why:** Python 3.14 is too new, many ML packages, including the
bundled pip, haven't fully caught up. 3.10–3.12 is the stable range.

---

## First AG News result (28 April, evening)

First run on AG News, default config:
- N=200 total examples, EVAL_SIZE=40, NUM_SERVER_QUERIES=30
- 3 clients, 6 rounds, alpha=0.5, 3 shots, llama3
- Runtime: 48 minutes

**Results:**
| Metric | Value |
|---|---|
| Zero-shot baseline | 95.0% (19/20) |
| Local-only baseline | 80.0% (16/20) |
| Round 0 (random init) | 33.3% |
| Rounds 1-4 | 76.7% |
| Rounds 5-6 | 80.0% |
| Held-out test set | 75.0% (15/20) |

**Federation gain over random init: +46.7%.**
**Federation accuracy *below* zero-shot baseline: -15.0%.**

**Interpretation:**
This is more interesting than expected. AG News is partially saturated
(zero-shot at 95%) but a clear federation curve is visible, random
init at 33% climbs monotonically to 80%. However, federation converges
*below* zero-shot, not above it.

The mechanism appears to be: llama3 already classifies news headlines
well without examples. Adding examples drawn from a Dirichlet-skewed
client pool (Client 0 has 25 world / 10 sports / 12 business / 22 science
out of 69 total) introduces a class-distribution bias. Three randomly
selected examples might be three "world" headlines, which causes the
model to over-weight that class for the next prediction. Adding biased
examples is worse than no examples for a model that's already strong on
the task.

This is consistent with Min et al. (2022): when base models are
strong, examples function more as distribution priors than as
demonstrations.

**Why this matters for the dissertation:**
This isn't a saturation failure, it's potentially a finding in its own
right. "Federated ICL fails to exceed zero-shot performance for capable
LLMs on standard benchmarks" is a defensible result with a clear
mechanism. The interaction with heterogeneity (alpha) becomes the
research question: at what level of heterogeneity does federation start
to add or subtract value over zero-shot?

**Eval set sizes are small.** 30 server queries and 20 test examples
means single-example flips move accuracy by 3–5 percentage points.
Ordering effects are typically smaller than this. Need to scale up.

---

## Action: scaling up evaluation (28 April, night)

Changes:
- `config.py`: `NUM_SERVER_QUERIES` 30 → 100, `EVAL_SIZE` 40 → 100
- `data.py`: `_load_ag_news(num_examples=...)` 200 → 350

Expected runtime: ~2–2.5 hours per run on the Mac (3× the previous run).

**Why:** to reduce per-example noise floor below the size of effects
we expect to measure (ordering differences typically 2–10 pp).

[Run in progress]

---

## GPU server access, pending (28 April)

Attempted SSH to `mcrugcomp03.ex.ac.uk` after VPN setup. Authentication
fails immediately after password entry: `Connection closed by port 22`.
Likely cause: account exists but isn't provisioned for shell access on
this specific server.

Emailed Dr. Jin to verify provisioning. Awaiting response.

**Workaround:** Local Mac runs are workable for the experiment sweep
(~2 hours per run × 12 runs = ~24 hours total, doable in two overnight
batches). GPU server becomes more useful when testing smaller models
(mistral, phi) to deliberately introduce headroom.

---

## Open questions / next steps

- Does the zero-shot vs Fed-ICL gap hold at n=100 evaluation?
- Does federation gain (over random init, not over zero-shot) vary with
  alpha?
- Do the five ordering strategies produce measurable differences at
  this eval size?
- If saturation persists with llama3, does federation behave differently
  with a weaker model (mistral 7B, phi 3.8B)?


## First scaled run, 100 server queries, baselines still on 20 (29 April, ~12:30 AM)

Ran with `NUM_SERVER_QUERIES=100`, `EVAL_SIZE=100`, total examples=350.

**Runtime: 6,642 seconds = 110 minutes** (≈2× the previous run, slightly
under prediction). Fed-ICL evaluation now over 100 server queries.

**Results:**
| Metric | Value | n |
|---|---|---|
| Zero-shot baseline | 75.0% | 20 |
| Local-only baseline | 80.0% | 20 |
| Round 0 (random init) | 25.0% | 100 |
| Round 1 | 76.0% | 100 |
| Round 2 | 79.0% | 100 |
| Round 3 | 79.0% | 100 |
| Round 4 | 80.0% | 100 |
| Round 5 | 76.0% | 100 |
| Round 6 | 78.0% | 100 |
| Held-out test | 75.0% | 20 |

Federation gain over random init: +53.0 pp (25% → 78%).

**Critical observation:** baselines and test set are still on 20
examples, `main.py` has hardcoded `eval_set[:20]` slices. Only the
Fed-ICL round-by-round evaluation actually scaled to 100. So
zero-shot/local-only/test numbers remain noisy.

Zero-shot dropped from 95% to 75% between runs despite both being on 20
examples, suggests the original 95% was an outlier from favourable
sampling. Fed-ICL convergence (78%) is much more reliable being based
on 100 examples.

The Dirichlet partition was sharply skewed this time: Client 0 had 0
sports, Client 1 had 30 sports out of 66, Client 2 had only 13 total
examples. Federation still converged consistently, encouraging
indication that majority voting handles severe heterogeneity at α=0.5.

**Action:** Removed the hardcoded `[:20]` in `main.py` so baselines run
on the full eval set. Re-run baseline before drawing any conclusion
about whether Fed-ICL exceeds, equals, or underperforms zero-shot.

**Why:** Cannot compare 100-example Fed-ICL to 20-example baseline.
Single example flips on n=20 move accuracy by 5 pp, larger than any
real effect we'd expect to detect. Baselines must be at the same scale.

## Corrected baseline, all metrics on n=100 (29 April, ~6:30 AM)

After fixing the hardcoded `eval_set[:20]` slice in `main.py`, re-ran the
full pipeline with all baselines and the held-out test on the same eval
set as Fed-ICL. Same Dirichlet partition as the previous run (Client 0:
71 examples, Client 1: 66, Client 2: 13).

**Results:**
| Metric | Value | n |
|---|---|---|
| Zero-shot baseline | 72.0% | 100 |
| Local-only baseline | 80.0% | 100 |
| Round 0 (random init) | 20.0% | 100 |
| Round 1 | 78.0% | 100 |
| Round 2 | 78.0% | 100 |
| Round 3 | 75.0% | 100 |
| Round 4 | 79.0% | 100 |
| Round 5 | 76.0% | 100 |
| Round 6 | 79.0% | 100 |
| Held-out test | 80.0% | 100 |

Federation gain over random init: +59.0 pp (20% → 79%).
Federation vs zero-shot: +7.0 pp.
Federation vs local-only: −1.0 pp (within noise).
Runtime: 165 minutes (vs 110 min previous; baselines now n=100 each).

**Interpretation:**
With proper sample-size symmetry, the picture is now defensible. Three
observations:

1. *Fed-ICL exceeds zero-shot by a real, measurable margin.* +7 pp on
   n=100 is well above the noise floor (~±2 pp at this scale).
   Federation is not a no-op.

2. *Fed-ICL roughly matches local-only at α=0.5.* This is consistent
   with the partition: local-only effectively uses Client 0's pool
   (71 examples covering all four classes), which is the strongest
   single client. Federation can only equal, not exceed, the best
   client when that client already has enough balanced data to do well
   alone.

3. *The held-out test set (80%) generalises from the server queries
   (79%) without overfitting.* The global context learned from the
   30-query iterative refinement transfers cleanly to unseen examples.

Yesterday's apparent "Fed-ICL underperforms zero-shot" finding was
entirely an artefact of the n=20 baselines. The original 95% zero-shot
on n=30 was favourable sampling on a small set; on n=100 the same
baseline is 72%. This is a useful reminder of how much sample size
matters at the scales we're working at.

**The research question becomes sharper:**
Federation's value should grow with heterogeneity. At α=0.5, no single
client is so weak that federation has obvious headroom over local-only.
At α=0.05, the partition would be extremely skewed and local-only
should suffer badly, federation should clearly win. At α=10.0, the
partition would be nearly uniform and the three metrics should converge.
The α-sweep is the next experiment.

**Action taken:** committed `results_agnews_baseline_n100.json` as a
preserved reference result in the repo (whitelisted in `.gitignore`).
This is the baseline against which all sweep runs will be compared.

---

## Methodological checklist added to repo (29 April)

Created `methodology_checklist.md` as a pre-flight check for every
experimental run, prompted by the n=20 issue caught yesterday. Items:

- [ ] Are all baselines (zero-shot, local-only) evaluated on the same n
      as Fed-ICL?
- [ ] Is the held-out test set evaluated on the same n?
- [ ] Is the random seed fixed and recorded?
- [ ] Is the Dirichlet partition reproducible from the seed?
- [ ] Has the result file been renamed before the next run?
- [ ] Has the run's purpose been logged in the lab notebook?

The single most important check is the first one: comparing
differently-sized baselines was what nearly hid a real federation gain
yesterday. Easy to miss; hard to detect after the fact.

---

## Open questions for the next meeting

- Should I include a fourth client at α=0.05 to make the heterogeneity
  more extreme, or stick with K=3 across all sweep configurations for
  consistency?
- For the ordering sweep, should I run all five strategies at α=0.5,
  or wait for the heterogeneity sweep to finish and then run ordering
  at the most informative α?
- The result here uses random selection. Would Dr. Jin prefer
  similarity-based selection (KATE-style) as the sweep's default, or
  is random selection the better baseline to vary ordering against?

## Mistral 7B run on AG News (3 May)

Same configuration as the llama3 baseline (α=0.5, K=3, T=6, 3 shots,
random selection, n=100 across all metrics, seed 42).

**Results:**
| Metric | mistral | llama3 (reference) |
|---|---|---|
| Zero-shot | 75% | 72% |
| Local-only | 78% | 80% |
| Round 6 (Fed-ICL) | 81% | 79% |
| Held-out test | 77% | 80% |
| Fed-ICL gain over zero-shot | +6 pp | +7 pp |
| Fed-ICL gain over local-only | +3 pp | −1 pp |
| Runtime | 8,322s | 9,932s |

**Observations:**
1. Mistral and llama3 are comparable in overall capability on this task, 
   all metrics within 3 pp of each other. The 7-8B open-instruct tier
   appears to perform similarly on AG News.

2. Federation's relative advantage is more visible with mistral:
   Fed-ICL exceeds local-only by 3 pp here, where llama3 had Fed-ICL
   matching local-only. Suggests federation's contribution becomes
   measurable when the single-client baseline is slightly weaker, even
   if overall capability is comparable.

3. Round-by-round stability is similar to llama3, convergence within
   one round, then a 4-pp band across the remaining rounds. No learning
   or degradation across rounds.

4. Mistral runs about 16% faster than llama3 (139 vs 165 min).

**Caveat:** A single comparison at one heterogeneity setting (α=0.5)
isn't enough to establish that federation's advantage scales with
weakness. Phi run pending.


## Methodology fix before proposal meeting (18 May)

**Context.** Draft proposal review with Dr. Jin scheduled for 18 May. A pre-meeting code review against the proposal claims surfaced several issues that would have invalidated or contaminated the upcoming ordering sweep. This session closes those gaps. No experimental conclusions change from these edits alone; numbers shift slightly because the parser is now strict and the baselines use the same pipeline as Fed-ICL.

### Methodology fixes

**Held-out evaluation now drawn from the AG News test split.**
`eval_set` previously came from a slice of the same 350-row training subsample as `server_queries` and `client_pool`. Within-run disjointness was preserved, but "held-out" in conventional ML usage means held out from the dataset's test split. `data.py` now loads 100 rows from `ag_news` `split="test"` with an independent seed (`SEED + 1`) so test sampling does not couple to training sampling.

**Dirichlet partition residual is round-robin, not dumped on client 0.**
Previously `counts[0] += len(class_indices) - counts.sum()` placed all rounding leftover on client 0. Because `run_baseline_local_only` uses client 0, this systematically inflated the local-only baseline's data pool. The residual is now distributed round-robin starting from a randomly chosen client index. Impact is largest at small alpha where rounding matters most.

**Local-only and held-out evaluation use the same select-and-order pipeline as Fed-ICL.**
Both previously did inline random selection with no ordering applied. This meant ordering experiments would have compared Fed-ICL-with-ordering against local-only-without-ordering, conflating federation and ordering effects. Both code paths now instantiate a `FedICLClient` and call its `select_examples` and `order_examples` methods. Federation-gain numbers now isolate the federation effect at a chosen ordering.

**`parse_label` switched from substring to token matching.**
Previous rule `if label in text` matched "world" inside any sentence containing the word, including sports headlines mentioning a world cup. Verbose model outputs were misclassified in a way that correlated with ordering strategy (poor orderings produce more verbose outputs from llama3). New rule splits on whitespace and punctuation and matches the first token that equals a label.

**Parse fallback rate is now logged.**
When no label matches, the parser still returns `LABEL_SPACE[0]` (preserved to avoid breaking the eval pipeline), but `_PARSE_STATS` now counts every fallback and stores up to 30 raw responses for inspection. `get_parse_stats()` is called from `main.py` and the summary is written into `results["parse_stats"]`. Smoke test on llama3 showed fallback rate of 0.1% (3 / 2,250), which means the strict parser is not over-rejecting valid outputs.

**Ollama retry with short backoff.**
`query_ollama` previously caught general exceptions and returned an empty string, which became a phantom "world" prediction via the parser fallback. Now retries up to two extra attempts with 3-second sleeps before giving up. Addresses the HTTP 500 / orphaned-runner failure mode observed previously.

**Selection branch shuffles after similarity selection.**
`np.argsort(scores)[-n:]` returns top-n indices sorted ascending by score, so under `SELECTION_STRATEGY="similarity"` the returned list was already similarity-ascending before `order_examples` saw it. `ORDER_STRATEGY="original"` was therefore not a true control under similarity selection. A shuffle inside the similarity branch makes ordering the sole controller of order.

**`order_examples` no longer mutates its input list.**
In-place `.sort()` replaced with `sorted(...)` returning a new list. Defensive; prevents surprising bugs as the codebase grows.

### Reproducibility

**All experimental parameters overridable via `FED_ICL_*` environment variables.**
`config.py` reads each parameter from `os.environ` with the literal value as fallback. Sweeps launch as a single shell command:

    for s in 41 42 43; do
      FED_ICL_SEED=$s FED_ICL_ORDER=label_grouped python main.py
    done

Source no longer needs to be edited between runs.

**Result files now record every parameter that produced them.**
Added `order_strategy`, `seed`, `num_server_queries`, `eval_size`, and `parse_stats` to `results["config"]`. Output filename auto-generated from the config:

    results_{model}_alpha{α}_K{K}_T{T}_seed{S}_order-{order}.json

Two runs with different parameters can never overwrite each other.

### Smoke test (smoke, not science)

Command: `FED_ICL_EVAL=10 FED_ICL_Q=10 FED_ICL_MODEL=llama3 python main.py`
Wall-clock: 5,644 s (≈ 94 min).

Signals:
- Parse fallback rate 0.1% (3 / 2,250 calls). The parser fix works and llama3 is emitting clean labels almost always.
- Auto-named output produced at `results_llama3_alpha0.5_K3_T6_seed42_order-original.json`.
- All new config fields written into the JSON as expected.
- Dirichlet partition under new residual code: client 0 = 30, client 1 = 216, client 2 = 94. Skew is genuine at α = 0.5 with a ~340-row training pool, not a bug. Worth flagging tomorrow as evidence that the heterogeneity sweep across α ∈ {0.05, 0.5, 10.0} will produce qualitatively different problems at each level.

Accuracy numbers at n = 10 are not meaningful (one example moves the metric by 10 percentage points) and are not retained as evidence. Real run uses `EVAL_SIZE = 100` and `NUM_SERVER_QUERIES = 100`.

### Commits planned for this session

1. **Code only.** Single commit containing all the edits above. No result changes.