"""
Run one or more arms for one seed and save results.

Example (calibration, seed 42):
    python -m p2.run --arms zeroshot,gt,fedicl,fedicl_free,solo_free --seed 42

Writes results_p2/<arm>_alpha<a>_seed<s>.json (summary + per-round log) and
results_p2/<arm>_alpha<a>_seed<s>.trace.jsonl.gz (every LLM call).
"""

import argparse
import gzip
import json
import os
import time

from p2.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="zeroshot,gt,fedicl,fedicl_free,solo_free")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--fake", action="store_true", help="offline test with fake backend/data")
    args = ap.parse_args()

    cfg = Config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.alpha is not None:
        cfg.alpha = args.alpha

    from p2.data_mmlu import prepare
    from p2.protocol import Runner
    from p2.retrieval import Retriever

    if args.fake:
        from tests_p2.fakes import FakeBackend, fake_items
        cfg.embedder = "fake"
        items = fake_items()
        backend = FakeBackend(items)
    else:
        from p2.backend import VLLMBackend
        items = None
        backend = VLLMBackend(cfg)
        print("Server models:", backend.health())

    queries, clients, eval_set, info = prepare(cfg, items)
    print(f"Data: {info}")
    runner = Runner(cfg, backend, Retriever(cfg.embedder))
    os.makedirs(cfg.results_dir, exist_ok=True)

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        t0 = time.time()
        runner.trace = []
        if hasattr(backend, "stats"):
            backend.stats = {"calls": 0, "unparseable": 0, "errors": 0}
        print(f"\n=== {arm} | alpha={cfg.alpha} seed={cfg.seed} ===", flush=True)
        out = runner.run_arm(arm, queries, clients, eval_set)
        out["arm"] = arm
        out["config"] = cfg.to_dict()
        out["data"] = info
        out["backend"] = {"constrain_mode": getattr(backend, "constrain_mode", "fake"),
                          **getattr(backend, "stats", {})}
        out["seconds"] = round(time.time() - t0, 1)
        stem = os.path.join(cfg.results_dir, f"{arm}_alpha{cfg.alpha}_seed{cfg.seed}")
        with open(stem + ".json", "w") as f:
            json.dump(out, f, indent=1)
        with gzip.open(stem + ".trace.jsonl.gz", "wt") as f:
            for rec in runner.trace:
                f.write(json.dumps(rec) + "\n")
        print(f"=== {arm}: {out['correct']}/{len(queries)} = {out['accuracy']:.3f} "
              f"({out['seconds']}s) -> {stem}.json", flush=True)


if __name__ == "__main__":
    main()
