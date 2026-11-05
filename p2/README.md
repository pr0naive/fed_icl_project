# Paper 2: Fed-ICL in Wang et al.'s MMLU setting

Clean re-implementation of Fed-ICL (Wang et al., ICML 2025) for the follow-up
paper, served with vLLM. Runs from the repo root: `python -m p2.run ...`.

## Files

| File | What it does |
|---|---|
| `config.py` | Every setting, with the paper's defaults. Deviations are marked. |
| `data_mmlu.py` | 114 fixed server queries (2 per subject); Dirichlet-over-subjects client split. |
| `retrieval.py` | kNN with paraphrase-MiniLM-L6-v2; Algorithm 2 local filtering. |
| `prompts.py` | Chat-format few-shot prompts (fixes the "Here are my answers" bug). |
| `backend.py` | vLLM client: concurrent, seeded requests; answer constrained to A-D; logs top log-probs. |
| `protocol.py` | Fed-ICL loop and arms: zeroshot, gt, ub, fedicl, fedicl_free, solo_free. |
| `run.py` | Runs arms for one seed; writes `results_p2/*.json` and `*.trace.jsonl.gz`. |

## Matches the paper

Llama-3.1-8B-Instruct, 3 clients, 6 rounds, 5 examples, temperature 0.1, 114
server queries, kNN retrieval with paraphrase-MiniLM-L6-v2, Algorithm 2
filtering, majority vote, Dirichlet over subjects, accuracy on server queries.

## Known deviations or unknowns (asked the authors)

- Answers constrained to A-D (`P2_CONSTRAIN=0` turns it off).
- Exact prompt text not published.
- Line 7 "conditions on D_i and D_i^k": default retrieves 5 from each (`P2_PROMPT_POOL=union` for 5 from the union).
- Initial server labels: random (`P2_INIT=zeroshot` alternative).
- Demonstration order: least to most similar.
- Held-out evaluation is our addition (`P2_EVAL`, off by default).

## The new control

`solo_free`: one client holding all client questions runs the Fed-ICL-Free loop.
No federation and no labels. If it matches Fed-ICL, the gain is not from federation.

## Tests

`python -m pytest -q tests_p2` runs everything offline with a fake LLM.
