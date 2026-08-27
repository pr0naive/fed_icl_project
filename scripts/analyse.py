"""
analyse.py
==========
Extract a summary table from Fed-ICL result JSON files in the current directory.
Handles both the older filename pattern (no pool tag, treated as pool=250) and
the new pool-sweep pattern (pool=N in the filename).

Usage (from the directory containing the result files):
    python3 analyse.py
"""
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict

# New filename pattern (with pool)
PATTERN_NEW = re.compile(
    r"results_(?P<model>[a-z0-9]+)_alpha(?P<alpha>[0-9.]+)_K(?P<K>\d+)_T(?P<T>\d+)"
    r"_pool(?P<pool>\d+)_seed(?P<seed>\d+)_order-(?P<order>[a-z_]+)\.json"
)

# Older pattern (no pool tag, implicit default 250)
PATTERN_OLD = re.compile(
    r"results_(?P<model>[a-z0-9]+)_alpha(?P<alpha>[0-9.]+)_K(?P<K>\d+)_T(?P<T>\d+)"
    r"_seed(?P<seed>\d+)_order-(?P<order>[a-z_]+)\.json"
)

DEFAULT_POOL = 250
MODELS_ORDER = ["phi3", "mistral", "llama3"]


def load_rows(results_dir="results"):
    """Load every matching result JSON in results_dir into a list of dicts."""
    rows = []
    for f in sorted(Path(results_dir).glob("results_*alpha0.5_K3_T6*.json")):
        m = PATTERN_NEW.match(f.name)
        pool = None
        if m:
            pool = int(m.group("pool"))
        else:
            m = PATTERN_OLD.match(f.name)
            if not m:
                continue
            pool = DEFAULT_POOL
        d = json.load(open(f))
        base = d.get("baselines", {})
        rounds_acc = [r["accuracy"] for r in d.get("rounds", [])]
        r6 = rounds_acc[-1] if rounds_acc else None
        local = base.get("local_only")
        rows.append({
            "filename": f.name,
            "model": m.group("model"),
            "seed": int(m.group("seed")),
            "pool": pool,
            "zero_shot": base.get("zero_shot"),
            "local_only": local,
            "fed_icl_r6": r6,
            "held_out": d.get("eval_accuracy"),
            "gain": (r6 - local) if (r6 is not None and local is not None) else None,
            "rounds": rounds_acc,
            "parse_fallback": d.get("parse_stats", {}).get("fallback_rate"),
        })
    return rows


def fmt_pp(v, prec=1):
    if v is None:
        return "    n/a"
    return f"{v * 100:+{prec + 4}.{prec}f}pp" if abs(v) < 1 else f"{v:+{prec + 4}.{prec}f}pp"


def msd(values):
    """Mean and std of a list of values, robust to None and small n."""
    vs = [v for v in values if v is not None]
    if not vs:
        return None, None
    mean = statistics.mean(vs)
    sd = statistics.stdev(vs) if len(vs) > 1 else 0.0
    return mean, sd


def print_per_condition(rows):
    """Per-condition table: every (model, pool, seed) row."""
    print("\n" + "=" * 78)
    print("PER-CONDITION RESULTS  (alpha=0.5, K=3, T=6, n=100)")
    print("=" * 78)
    print(f"{'Model':<8} {'Pool':>5} {'Seed':>5}  {'ZS':>6} {'Local':>6} {'R6':>6} {'HO':>6}   {'Gain':>7}")
    print("-" * 78)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            MODELS_ORDER.index(r["model"]) if r["model"] in MODELS_ORDER else 99,
            r["pool"],
            r["seed"],
        ),
    )

    last_model = None
    last_pool = None
    for r in rows_sorted:
        if r["model"] != last_model:
            print()
        elif r["pool"] != last_pool:
            pass  # no blank line within model, only between models
        last_model, last_pool = r["model"], r["pool"]

        fmt = lambda v: f"{v*100:5.1f}%" if v is not None else "    --"
        gain_str = f"{r['gain']*100:+6.1f}pp" if r["gain"] is not None else "      --"
        print(
            f"{r['model']:<8} {r['pool']:>5} {r['seed']:>5}  "
            f"{fmt(r['zero_shot']):>6} {fmt(r['local_only']):>6} "
            f"{fmt(r['fed_icl_r6']):>6} {fmt(r['held_out']):>6}   "
            f"{gain_str:>7}"
        )


def print_per_model_per_pool(rows):
    """Aggregate across seeds: per (model, pool), mean and std."""
    print("\n" + "=" * 78)
    print("PER-MODEL × PER-POOL SUMMARY  (mean ± std across seeds)")
    print("=" * 78)
    print(f"{'Model':<8} {'Pool':>5} {'n':>3}  {'Local m±s':>14}  {'R6 m±s':>14}  {'Gain m±s':>16}")
    print("-" * 78)

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["pool"])].append(r)

    for model in MODELS_ORDER:
        pools = sorted({p for m, p in grouped.keys() if m == model})
        for pool in pools:
            entries = grouped[(model, pool)]
            n = len(entries)
            lo_m, lo_s = msd([e["local_only"] for e in entries])
            r6_m, r6_s = msd([e["fed_icl_r6"] for e in entries])
            gn_m, gn_s = msd([e["gain"] for e in entries])
            lo_str = f"{lo_m*100:5.1f} ± {lo_s*100:4.1f}" if lo_m is not None else "    n/a    "
            r6_str = f"{r6_m*100:5.1f} ± {r6_s*100:4.1f}" if r6_m is not None else "    n/a    "
            gn_str = f"{gn_m*100:+5.1f} ± {gn_s*100:4.1f}pp" if gn_m is not None else "    n/a    "
            print(f"{model:<8} {pool:>5} {n:>3}  {lo_str:>14}  {r6_str:>14}  {gn_str:>16}")
        print()


def print_per_pool_aggregated(rows):
    """Aggregate across both seeds and models: per pool, the average gain."""
    print("\n" + "=" * 78)
    print("PER-POOL SUMMARY  (across all models and seeds)")
    print("=" * 78)
    print(f"{'Pool':>5} {'n':>3}   {'Gain m±s':>15}   Notes")
    print("-" * 78)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["pool"]].append(r)

    for pool in sorted(grouped.keys()):
        entries = grouped[pool]
        n = len(entries)
        gn_m, gn_s = msd([e["gain"] for e in entries])
        gn_str = f"{gn_m*100:+5.1f} ± {gn_s*100:4.1f}pp" if gn_m is not None else "    n/a    "
        models_present = sorted({e["model"] for e in entries})
        print(f"{pool:>5} {n:>3}   {gn_str:>15}   models: {', '.join(models_present)}")


def print_negative_cases(rows):
    """Conditions where federation hurt (gain < 0)."""
    print("\n" + "=" * 78)
    print("CONDITIONS WHERE FEDERATION HURT  (gain < 0)")
    print("=" * 78)
    print(f"{'Model':<8} {'Pool':>5} {'Seed':>5}  {'Local':>6} {'R6':>6}   {'Gain':>7}")
    print("-" * 78)

    neg = [r for r in rows if r["gain"] is not None and r["gain"] < 0]
    if not neg:
        print("  None.")
        return
    neg.sort(key=lambda r: r["gain"])
    for r in neg:
        fmt = lambda v: f"{v*100:5.1f}%" if v is not None else "    --"
        print(
            f"{r['model']:<8} {r['pool']:>5} {r['seed']:>5}  "
            f"{fmt(r['local_only']):>6} {fmt(r['fed_icl_r6']):>6}   "
            f"{r['gain']*100:+6.1f}pp"
        )
    print(f"\nTotal: {len(neg)} of {len(rows)} conditions ({100*len(neg)/len(rows):.0f}%)")


def print_best_conditions(rows, top_n=5):
    """Top-N conditions by gain."""
    print("\n" + "=" * 78)
    print(f"TOP {top_n} CONDITIONS BY FEDERATION GAIN")
    print("=" * 78)
    print(f"{'Model':<8} {'Pool':>5} {'Seed':>5}  {'Local':>6} {'R6':>6}   {'Gain':>7}")
    print("-" * 78)

    pos = sorted(
        [r for r in rows if r["gain"] is not None],
        key=lambda r: -r["gain"],
    )[:top_n]
    for r in pos:
        fmt = lambda v: f"{v*100:5.1f}%" if v is not None else "    --"
        print(
            f"{r['model']:<8} {r['pool']:>5} {r['seed']:>5}  "
            f"{fmt(r['local_only']):>6} {fmt(r['fed_icl_r6']):>6}   "
            f"{r['gain']*100:+6.1f}pp"
        )


def print_verdict_stats(rows):
    """Verdict-supporting statistics per (model, pool): lower bound and CV.

    Lower bound (mean - std) > 0 is the 'federation reliably helps' criterion:
    the federation gain stays positive even one std below the observed mean.
    CV (std / |mean|) normalises noise against effect size, useful when
    comparing cells with very different mean gains.
    """
    print("\n" + "=" * 78)
    print("VERDICT-SUPPORTING STATISTICS  (per model × pool)")
    print("  *** marks 'federation reliably helps' (mean - std > 0)")
    print("=" * 78)
    print(f"{'Model':<8} {'Pool':>5} {'n':>3}  {'Mean':>8}  {'Std':>5}  {'Mean-Std':>9}  {'CV':>5}")
    print("-" * 78)

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["pool"])].append(r)

    reliable = []
    for model in MODELS_ORDER:
        pools = sorted({p for m, p in grouped.keys() if m == model})
        for pool in pools:
            entries = grouped[(model, pool)]
            n = len(entries)
            gn_m, gn_s = msd([e["gain"] for e in entries])
            if gn_m is None or n < 2:
                print(f"{model:<8} {pool:>5} {n:>3}  (n<2, no std)")
                continue
            mean_pp = gn_m * 100
            std_pp = gn_s * 100
            lower = mean_pp - std_pp
            cv = abs(std_pp / mean_pp) if mean_pp != 0 else float("inf")
            cv_str = f"{cv:5.2f}" if cv != float("inf") else "  inf"
            marker = " ***" if lower > 0 else ""
            print(
                f"{model:<8} {pool:>5} {n:>3}  {mean_pp:+6.1f}pp  {std_pp:5.1f}  "
                f"{lower:+6.1f}pp  {cv_str}{marker}"
            )
            if lower > 0:
                reliable.append((model, pool, mean_pp, std_pp, lower))
        print()

    print("Cells where federation reliably helps (mean - std > 0):")
    print("-" * 78)
    if reliable:
        for m, p, mean, std, lb in reliable:
            print(f"  {m} pool={p}: mean={mean:+.1f}pp std={std:.1f}pp lower={lb:+.1f}pp")
    else:
        print("  None across the observed seeds.")


def print_rank_consistency(rows):
    """Paired rank consistency across seeds per model, with Kendall's W.

    For each model, ranks pool sizes by gain within each seed (1 = best).
    If the same pool consistently ranks best across seeds, that is stronger
    evidence than absolute gain values alone, because it sidesteps the
    std-estimation problem at small n. Kendall's W summarises agreement
    on a 0 (no agreement) to 1 (perfect agreement) scale.
    """
    print("\n" + "=" * 78)
    print("RANK CONSISTENCY ACROSS SEEDS  (1 = best gain at that seed)")
    print("=" * 78)

    grouped = defaultdict(dict)  # (model, pool) -> {seed: gain}
    for r in rows:
        if r["gain"] is not None:
            grouped[(r["model"], r["pool"])][r["seed"]] = r["gain"]

    for model in MODELS_ORDER:
        pools = sorted({p for m, p in grouped.keys() if m == model})
        if not pools:
            continue

        seeds_per_pool = [set(grouped[(model, p)].keys()) for p in pools]
        common_seeds = sorted(set.intersection(*seeds_per_pool)) if seeds_per_pool else []
        if len(common_seeds) < 2:
            print(f"\n{model}: fewer than 2 common seeds across all pools, skipping")
            continue

        seed_ranks = {}
        for s in common_seeds:
            gains_s = {p: grouped[(model, p)][s] for p in pools}
            ordered = sorted(gains_s.keys(), key=lambda p: -gains_s[p])
            seed_ranks[s] = {p: rank + 1 for rank, p in enumerate(ordered)}

        print(f"\n{model} (common seeds {common_seeds}):")
        header = f"{'Pool':>5}  " + "".join(f"seed={s:<5}" for s in common_seeds) + " mean_rank"
        print(header)
        print("-" * 78)
        for p in pools:
            ranks = [seed_ranks[s][p] for s in common_seeds]
            mean_rank = statistics.mean(ranks)
            rank_str = "".join(f"{r:<10}" for r in ranks)
            print(f"{p:>5}  {rank_str}{mean_rank:>5.2f}")

        m_n = len(common_seeds)
        k_n = len(pools)
        if k_n >= 2:
            rank_sums = [sum(seed_ranks[s][p] for s in common_seeds) for p in pools]
            mean_rs = m_n * (k_n + 1) / 2
            S = sum((rs - mean_rs) ** 2 for rs in rank_sums)
            W = 12 * S / (m_n ** 2 * (k_n ** 3 - k_n))
            print(f"Kendall's W = {W:.3f}  (0 = no agreement, 1 = perfect agreement)")

def print_local_gain_correlation(rows):
    """Correlation between local-only accuracy and federation gain.

    A strong negative r is consistent with two non-exclusive mechanisms:
    (a) ceiling effect: R6 saturates around the achievable max per model, so
        low local-only leaves more headroom for gain;
    (b) partition-rescue: federation helps most when client 0's local
        partition is weak, sharing information from better-positioned clients.
    Correlation alone cannot distinguish these; reported per model so the
    ceiling story can be sanity-checked against the max R6 observed.
    """
    print("\n" + "=" * 78)
    print("LOCAL-ONLY vs FEDERATION GAIN CORRELATION")
    print("  strong negative r => ceiling effect or partition-rescue mechanism")
    print("=" * 78)

    def pearson_r(pairs):
        clean = [(x, y) for x, y in pairs if x is not None and y is not None]
        n = len(clean)
        if n < 3:
            return None, n
        xs, ys = zip(*clean)
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in clean)
        den_x = sum((x - mx) ** 2 for x in xs)
        den_y = sum((y - my) ** 2 for y in ys)
        if den_x == 0 or den_y == 0:
            return None, n
        return num / (den_x * den_y) ** 0.5, n

    all_pairs = [(r["local_only"], r["gain"]) for r in rows]
    r_all, n_all = pearson_r(all_pairs)
    if r_all is not None:
        print(f"\nAll conditions: r = {r_all:+.3f}  (n = {n_all})")
    else:
        print(f"\nAll conditions: insufficient data (n = {n_all})")

    print()
    print(f"{'Model':<10} {'n':>3}  {'r':>8}   {'max R6':>7}  {'mean R6':>8}")
    print("-" * 78)
    for model in MODELS_ORDER:
        model_pairs = [
            (r["local_only"], r["gain"])
            for r in rows
            if r["model"] == model
        ]
        r_m, n_m = pearson_r(model_pairs)
        r6_vals = [
            r["fed_icl_r6"]
            for r in rows
            if r["model"] == model and r["fed_icl_r6"] is not None
        ]
        r_str = f"{r_m:+8.3f}" if r_m is not None else "     n/a"
        if r6_vals:
            max_r6 = max(r6_vals) * 100
            mean_r6 = statistics.mean(r6_vals) * 100
            r6_str = f"{max_r6:6.1f}%  {mean_r6:7.1f}%"
        else:
            r6_str = "     n/a       n/a"
        print(f"{model:<10} {n_m:>3}  {r_str}   {r6_str}")

def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Analyse Fed-ICL result JSON files.")
    parser.add_argument(
        "--results-dir",
        default=".",
        help="Directory containing result JSON files (default: current directory)",
    )
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"No matching result files found in {args.results_dir}.")

    print(f"Loaded {len(rows)} result files from {args.results_dir}.")

    print_per_condition(rows)
    print_per_model_per_pool(rows)
    print_per_pool_aggregated(rows)
    print_negative_cases(rows)
    print_best_conditions(rows)
    print_verdict_stats(rows)
    print_rank_consistency(rows)
    print_local_gain_correlation(rows)


if __name__ == "__main__":
    main()