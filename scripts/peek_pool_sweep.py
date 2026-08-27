"""Quick look at pool-size sweep partial results."""
import json
import re
from pathlib import Path
from collections import defaultdict

PATTERN = re.compile(
    r"results_(?P<model>[a-z0-9]+)_alpha(?P<alpha>[0-9.]+)_K(?P<K>\d+)_T(?P<T>\d+)"
    r"_pool(?P<pool>\d+)_seed(?P<seed>\d+)_order-(?P<order>[a-z_]+)\.json"
)

# Also catch the older filename pattern (multi-seed at pool=250)
PATTERN_OLD = re.compile(
    r"results_(?P<model>[a-z0-9]+)_alpha(?P<alpha>[0-9.]+)_K(?P<K>\d+)_T(?P<T>\d+)"
    r"_seed(?P<seed>\d+)_order-(?P<order>[a-z_]+)\.json"
)

rows = []
for f in sorted(Path(".").glob("results_*alpha0.5_K3_T6*.json")):
    m = PATTERN.match(f.name)
    if m:
        pool = int(m.group("pool"))
    else:
        m = PATTERN_OLD.match(f.name)
        if not m:
            continue
        pool = 250  # implicit default in older runs
    d = json.load(open(f))
    base = d.get("baselines", {})
    r6 = d["rounds"][-1]["accuracy"]
    local = base.get("local_only")
    rows.append({
        "model": m.group("model"),
        "seed": int(m.group("seed")),
        "pool": pool,
        "local_only": local,
        "fed_icl_r6": r6,
        "gain": (r6 - local) if (r6 is not None and local is not None) else None,
    })

# Group by (model, pool), average across seeds where available
grouped = defaultdict(list)
for r in rows:
    grouped[(r["model"], r["pool"])].append(r["gain"] * 100)

print(f"{'Model':<8} {'Pool':>5} {'n':>3} {'Gain mean ± std':>18}   Per-seed gains")
print("-" * 75)
for model in ["phi3", "mistral", "llama3"]:
    for pool in sorted({r["pool"] for r in rows if r["model"] == model}):
        gains = grouped[(model, pool)]
        mean = sum(gains) / len(gains)
        if len(gains) > 1:
            sd = (sum((g - mean) ** 2 for g in gains) / (len(gains) - 1)) ** 0.5
            summary = f"{mean:+5.1f} ± {sd:4.1f}pp"
        else:
            summary = f"{mean:+5.1f} (n=1)  "
        per_seed = ", ".join(f"{g:+.0f}" for g in gains)
        print(f"{model:<8} {pool:>5} {len(gains):>3}  {summary:>17}   [{per_seed}]")