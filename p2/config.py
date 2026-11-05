"""
Paper-2 configuration: Fed-ICL in Wang et al.'s (ICML 2025) MMLU setting.

Every default below mirrors the paper unless marked DEVIATION. Each value can be
overridden with an environment variable (P2_*) or a command-line flag in run.py.

Paper settings mirrored here:
  model               Llama-3.1-8B-Instruct (rebuttal Table 1 backbone)
  clients             3
  rounds              6
  context examples    5 (C)
  temperature         0.1
  server queries      2 per MMLU subject = 114
  retrieval           kNN in paraphrase-MiniLM-L6-v2 space (App. C.1)
  local filtering     Algorithm 2: keep C nearest local items per server query
  aggregation         majority vote for classification (Sec. 5, Eq. 5)
  heterogeneity       Dirichlet over subjects, alpha in {0.001, 1, 100}
  metric              accuracy of the aggregated answers on the server queries
"""

import os
from dataclasses import dataclass, field, asdict


def _env(name, default):
    return os.environ.get(name, default)


@dataclass
class Config:
    # Inference backend (vLLM, OpenAI-compatible server)
    model: str = _env("P2_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    base_url: str = _env("P2_BASE_URL", "http://localhost:8000/v1")
    temperature: float = float(_env("P2_TEMPERATURE", "0.1"))
    # DEVIATION (documented): constrain the reply to A/B/C/D so no answer is
    # lost to formatting. Set P2_CONSTRAIN=0 for free generation plus parsing.
    constrain: bool = _env("P2_CONSTRAIN", "1") == "1"
    max_tokens: int = int(_env("P2_MAX_TOKENS", "8"))
    concurrency: int = int(_env("P2_CONCURRENCY", "64"))
    top_logprobs: int = int(_env("P2_TOP_LOGPROBS", "5"))

    # Federation (Wang et al. defaults)
    num_clients: int = int(_env("P2_CLIENTS", "3"))
    num_rounds: int = int(_env("P2_ROUNDS", "6"))
    num_shots: int = int(_env("P2_SHOTS", "5"))
    alpha: float = float(_env("P2_ALPHA", "1.0"))
    seed: int = int(_env("P2_SEED", "42"))

    # Algorithm 1 line 7 conditions on D_i and D_i^k. "both" retrieves C items
    # from each set (literal reading, 2C demonstrations); "union" retrieves C
    # from the concatenation. The paper's exact prompt is not published.
    prompt_pool: str = _env("P2_PROMPT_POOL", "both")
    # Algorithm 1 line 1 initialises server labels; the paper does not say how.
    # "random" (uniform letter) is our default; "zeroshot" is an alternative.
    init: str = _env("P2_INIT", "random")
    use_filter: bool = _env("P2_FILTER", "1") == "1"

    # Data
    queries_per_subject: int = int(_env("P2_QPS", "2"))
    query_seed: int = int(_env("P2_QUERY_SEED", "12345"))
    # Held-out evaluation is OUR addition (the paper scores server queries only).
    # 0 disables it; when on, items are disjoint from queries and client pools.
    eval_size: int = int(_env("P2_EVAL", "0"))
    eval_seed: int = int(_env("P2_EVAL_SEED", "54321"))

    # Retrieval
    embedder: str = _env("P2_EMBEDDER", "paraphrase-MiniLM-L6-v2")  # "fake" for tests

    results_dir: str = _env("P2_RESULTS", "results_p2")

    def to_dict(self):
        return asdict(self)
