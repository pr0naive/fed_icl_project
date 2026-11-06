"""
P1 analysis: arms compared on the same questions within each seed, pooled over seeds.

    python analysis_p2/analyze_p1.py results_p2/p1_alpha1.0_both
"""
import glob, gzip, json, os, sys
from collections import defaultdict
from math import comb

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "results_p2/p1_alpha1.0_both"
ARMS = ["zeroshot", "randlabel", "gt", "fedicl", "fedicl_free", "solo_free"]

R = defaultdict(dict)
for f in glob.glob(os.path.join(D, "*.json")):
    d = json.load(open(f))
    R[d["arm"]][d["config"]["seed"]] = d
seeds = sorted(set.intersection(*[set(R[a]) for a in ARMS]))
print(f"Folder {D}: seeds {seeds}, {len(seeds) * 114} questions per arm\n")

def vec(arm):
    return np.concatenate([np.array(R[arm][s]["per_query_correct"], dtype=bool) for s in seeds])

def mcnemar(a, b):
    A, B = vec(a), vec(b)
    only_a, only_b = int((A & ~B).sum()), int((~A & B).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p

def boot_ci(a, b, reps=10000, rng=np.random.default_rng(0)):
    diff = vec(a).astype(int) - vec(b).astype(int)
    idx = rng.integers(0, len(diff), size=(reps, len(diff)))
    m = diff[idx].mean(1) * 100
    return diff.mean() * 100, np.percentile(m, 2.5), np.percentile(m, 97.5)

print("Accuracy per seed (correct / 114) and pooled")
print(f"{'arm':12s} " + " ".join(f"s{s:<4d}" for s in seeds) + "   pooled")
for a in ARMS:
    c = [R[a][s]["correct"] for s in seeds]
    print(f"{a:12s} " + " ".join(f"{x:<5d}" for x in c) + f"   {sum(c)}/{114*len(seeds)} = {100*sum(c)/(114*len(seeds)):.1f}%")

print("\nPaired comparisons, pooled over all questions (exact McNemar; 95% bootstrap CI of the difference)")
pairs = [("randlabel", "zeroshot"), ("gt", "zeroshot"), ("fedicl", "zeroshot"), ("fedicl_free", "zeroshot"),
         ("solo_free", "zeroshot"), ("gt", "randlabel"), ("fedicl", "gt"), ("fedicl_free", "gt"),
         ("fedicl", "fedicl_free"), ("fedicl_free", "solo_free"), ("fedicl", "solo_free"), ("solo_free", "gt")]
for a, b in pairs:
    oa, ob, p = mcnemar(a, b)
    m, lo, hi = boot_ci(a, b)
    print(f"{a:11s} - {b:11s}: {m:+5.1f} pts [{lo:+5.1f}, {hi:+5.1f}]  only-{a}={oa:3d} only-{b}={ob:3d}  p={p:.3f}")

print("\nIterative arms: accuracy and pseudo-label accuracy by round (mean over seeds)")
for a in ["fedicl", "fedicl_free", "solo_free"]:
    acc = np.array([[r["accuracy"] for r in R[a][s]["rounds"]] for s in seeds]) * 100
    ps = np.array([[np.mean(r["pseudo_label_acc"]) for r in R[a][s]["rounds"][1:]] for s in seeds]) * 100
    print(f"  {a:11s} acc  r0..r6: " + " ".join(f"{x:5.1f}" for x in acc.mean(0)))
    print(f"  {a:11s} pseudo r1..r6:       " + " ".join(f"{x:5.1f}" for x in ps.mean(0)))

print("\nDo iterative arms just reproduce zero-shot answers? (final labels equal to zero-shot answer)")
for a in ["gt", "randlabel", "fedicl", "fedicl_free", "solo_free"]:
    same = [np.mean([x == y for x, y in zip(R[a][s]["final_labels"], R["zeroshot"][s]["final_labels"])]) for s in seeds]
    print(f"  {a:11s}: {100*np.mean(same):.1f}% of answers identical to zero-shot")

print("\nRound 1 vs round 6 server labels (how much the loop changes its own answers)")
for a in ["fedicl", "fedicl_free", "solo_free"]:
    ch = [np.mean([x != y for x, y in zip(R[a][s]["rounds"][1]["labels"], R[a][s]["rounds"][-1]["labels"])]) for s in seeds]
    print(f"  {a:11s}: {100*np.mean(ch):.1f}% of the 114 answers change between round 1 and round 6")

print("\nHealth: unparseable / errors / too_long per arm (summed)")
for a in ARMS:
    b = [R[a][s]["backend"] for s in seeds]
    print(f"  {a:11s}: calls={sum(x['calls'] for x in b)} unparseable={sum(x['unparseable'] for x in b)} "
          f"errors={sum(x['errors'] for x in b)} too_long={sum(x.get('too_long', 0) for x in b)} mode={b[0]['constrain_mode']}")
print("\nData splits:", {s: R['fedicl'][s]['data']['client_sizes'] for s in seeds})
print("Query draws:", {s: R['fedicl'][s]['data'].get('query_seed_used') for s in seeds})
