"""
compare_variants.py
===================
Head-to-head paired comparison of Fed-ICL vs Fed-ICL-Free.

For every (model, pool, seed) cell present in both result directories,
print the federation gain (R6_gain and HO_gain) under each variant and
the difference (Fed-ICL-Free minus Fed-ICL).
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyse import load_rows, MODELS_ORDER


def main():
    parser = argparse.ArgumentParser(description="Paired Fed-ICL vs Fed-ICL-Free comparison.")
    parser.add_argument("--fed-icl-dir", required=True, help="Directory with Fed-ICL results.")
    parser.add_argument("--fed-icl-free-dir", required=True, help="Directory with Fed-ICL-Free results.")
    args = parser.parse_args()

    rows_a = load_rows(args.fed_icl_dir)
    rows_b = load_rows(args.fed_icl_free_dir)

    map_a = {(r["model"], r["pool"], r["seed"]): r for r in rows_a}
    map_b = {(r["model"], r["pool"], r["seed"]): r for r in rows_b}

    common_keys = sorted(
        set(map_a.keys()) & set(map_b.keys()),
        key=lambda k: (MODELS_ORDER.index(k[0]) if k[0] in MODELS_ORDER else 99, k[1], k[2]),
    )

    print(f"\nFed-ICL:      {len(rows_a)} cells loaded from {args.fed_icl_dir}")
    print(f"Fed-ICL-Free: {len(rows_b)} cells loaded from {args.fed_icl_free_dir}")
    print(f"Common cells: {len(common_keys)}")

    print("\n" + "=" * 92)
    print("PAIRED COMPARISON: Fed-ICL vs Fed-ICL-Free")
    print("  delta_R6 = R6_gain(Fed-ICL-Free) - R6_gain(Fed-ICL)")
    print("  delta_HO = HO_gain(Fed-ICL-Free) - HO_gain(Fed-ICL)")
    print("  positive delta means Fed-ICL-Free improved on the cell")
    print("=" * 92)
    header = (
        f"{'Model':<8} {'Pool':>5} {'Seed':>5}  "
        f"{'R6_FedICL':>10}  {'R6_Free':>9}  {'delta_R6':>9}    "
        f"{'HO_FedICL':>10}  {'HO_Free':>9}  {'delta_HO':>9}"
    )
    print(header)
    print("-" * 92)

    sum_delta_r6 = 0
    sum_delta_ho = 0
    count_r6_better = 0
    count_ho_better = 0

    last_model = None
    for key in common_keys:
        model, pool, seed = key
        a = map_a[key]
        b = map_b[key]

        if model != last_model:
            print()
            last_model = model

        a_r6_gain = a["gain"]
        b_r6_gain = b["gain"]
        a_ho_gain = (a["held_out"] - a["local_only"]) if (a["held_out"] is not None and a["local_only"] is not None) else None
        b_ho_gain = (b["held_out"] - b["local_only"]) if (b["held_out"] is not None and b["local_only"] is not None) else None

        delta_r6 = (b_r6_gain - a_r6_gain) if (a_r6_gain is not None and b_r6_gain is not None) else None
        delta_ho = (b_ho_gain - a_ho_gain) if (a_ho_gain is not None and b_ho_gain is not None) else None

        if delta_r6 is not None:
            sum_delta_r6 += delta_r6
            if delta_r6 > 0:
                count_r6_better += 1
        if delta_ho is not None:
            sum_delta_ho += delta_ho
            if delta_ho > 0:
                count_ho_better += 1

        fmt = lambda v: f"{v*100:+6.1f}pp" if v is not None else "    --"
        print(
            f"{model:<8} {pool:>5} {seed:>5}  "
            f"{fmt(a_r6_gain):>10}  {fmt(b_r6_gain):>9}  {fmt(delta_r6):>9}    "
            f"{fmt(a_ho_gain):>10}  {fmt(b_ho_gain):>9}  {fmt(delta_ho):>9}"
        )

    n = len(common_keys)
    print("\n" + "=" * 92)
    print(f"Summary across {n} paired cells:")
    print(f"  Mean delta_R6: {sum_delta_r6 / n * 100:+.2f}pp   "
          f"Fed-ICL-Free better on R6 in {count_r6_better}/{n} cells")
    print(f"  Mean delta_HO: {sum_delta_ho / n * 100:+.2f}pp   "
          f"Fed-ICL-Free better on HO in {count_ho_better}/{n} cells")


if __name__ == "__main__":
    main()
