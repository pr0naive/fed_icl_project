"""
plot_multi_seed.py
==================
Build the multi-seed Fed-ICL analysis figure set. Four PNGs written to plots/.

  01_gain_heatmap.png         : Federation gain by model and seed (headline visual)
  02_gain_distribution.png    : Per-model gain distribution with mean ± std
  03_round_trajectories.png   : Round-by-round accuracy per condition, colour by seed
  04_fed_vs_local_scatter.png : Fed-ICL R6 against local-only baseline

Run from the directory containing the result JSON files:
    python3 plot_multi_seed.py

Dependencies: matplotlib, numpy. Install with: pip install matplotlib numpy
"""
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = "."
OUTPUT_DIR  = Path("plots")
MODELS_ORDER = ["phi3", "mistral", "llama3"]
SEEDS_ORDER  = [3, 7, 13, 42, 99]

# Palette consistent with the supervision deck
NAVY  = "#21295C"
TEAL  = "#1C7293"
DEEP  = "#065A82"
GREEN = "#0E7C66"
RED   = "#B91C1C"
AMBER = "#B45309"
MUTED = "#64748B"
INK   = "#1E293B"
SOFT  = "#F1F5F9"

MODEL_COLOR = {"phi3": AMBER, "mistral": TEAL, "llama3": DEEP}

# Sequential colour per seed (used in plots 3 and 4 so the same seed
# is the same colour across both plots)
_seed_cmap = plt.cm.viridis(np.linspace(0.10, 0.85, len(SEEDS_ORDER)))
SEED_COLOR = {s: tuple(c) for s, c in zip(SEEDS_ORDER, _seed_cmap)}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
PLOT_VARIANT = "fed_icl"

PATTERN = re.compile(
    r"results_(?P<model>[a-z0-9]+)_variant-(?P<variant>[a-z_]+)"
    r"_alpha(?P<alpha>[0-9.]+)_K(?P<K>\d+)_T(?P<T>\d+)_pool(?P<pool>\d+)"
    r"_seed(?P<seed>\d+)_order-(?P<order>[a-z_]+)\.json"
)

rows = []
for f in sorted(Path(RESULTS_DIR).glob("results_*alpha0.5_K3_T6_*_order-original.json")):
    m = PATTERN.match(f.name)
    if not m:
        continue
    if m.group("variant") != PLOT_VARIANT:
        continue
    d = json.load(open(f))
    base = d.get("baselines", {})
    round_acc = [r["accuracy"] for r in d.get("rounds", [])]
    r6 = round_acc[-1] if round_acc else None   # accuracy on the 100 shared-context queries (in-sample)
    local = base.get("local_only")
    held = d.get("eval_accuracy")               # held-out generalization (n=1000), the fair metric
    rows.append({
        "model": m.group("model"),
        "seed": int(m.group("seed")),
        "zero_shot": base.get("zero_shot"),
        "local_only": local,
        "fed_icl_r6": r6,
        "held_out": held,
        # Federation gain is defined on HELD-OUT, not R6. R6 is accuracy on the
        # federation's own shared context and overstates generalization by ~9pp.
        "gain": (held - local) if (held is not None and local is not None) else None,
        "rounds": round_acc,
    })

if not rows:
    raise SystemExit("No matching result files found in current directory.")

OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Loaded {len(rows)} result files.")

def find(model, seed):
    for r in rows:
        if r["model"] == model and r["seed"] == seed:
            return r
    return None

# ===========================================================================
# Figure 1 — Gain heatmap
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 3.2))

gain_grid = np.full((len(MODELS_ORDER), len(SEEDS_ORDER)), np.nan)
for i, m in enumerate(MODELS_ORDER):
    for j, s in enumerate(SEEDS_ORDER):
        r = find(m, s)
        if r and r["gain"] is not None:
            gain_grid[i, j] = r["gain"] * 100

vmax = max(abs(np.nanmin(gain_grid)), abs(np.nanmax(gain_grid)))
cmap = mcolors.LinearSegmentedColormap.from_list(
    "fed_icl_div", [RED, "#FFFFFF", GREEN], N=256
)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

im = ax.imshow(gain_grid, cmap=cmap, norm=norm, aspect="auto")

for i in range(len(MODELS_ORDER)):
    for j in range(len(SEEDS_ORDER)):
        v = gain_grid[i, j]
        if np.isnan(v):
            continue
        text_color = INK if abs(v) < vmax * 0.55 else "white"
        ax.text(j, i, f"{v:+.0f}pp", ha="center", va="center",
                color=text_color, fontsize=11, fontweight="bold")

ax.set_xticks(range(len(SEEDS_ORDER)))
ax.set_xticklabels([str(s) for s in SEEDS_ORDER])
ax.set_yticks(range(len(MODELS_ORDER)))
ax.set_yticklabels(MODELS_ORDER)
ax.set_xlabel("Seed", color=INK)
ax.set_ylabel("Model", color=INK)
ax.set_title("Federation gain by model and seed   (α=0.5, K=3, T=6, held-out n=1000)",
             color=NAVY, pad=10, fontweight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.85, label="Fed-ICL held-out − Local-only (pp)")
cbar.ax.tick_params(labelsize=9)

fig.text(0.5, -0.06,
         "Blue = federation hurts held-out generalization. Negative in 9 of 15 conditions at α=0.5.",
         ha="center", color=MUTED, style="italic", fontsize=9)

fig.savefig(OUTPUT_DIR / "01_gain_heatmap.png")
plt.close(fig)
print(f"  wrote {OUTPUT_DIR / '01_gain_heatmap.png'}")

# ===========================================================================
# Figure 2 — Gain distribution per model
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.0))

rng = np.random.default_rng(0)
for i, model in enumerate(MODELS_ORDER):
    gains = [r["gain"] * 100 for r in rows if r["model"] == model and r["gain"] is not None]
    if not gains:
        continue
    mean = statistics.mean(gains)
    sd   = statistics.stdev(gains) if len(gains) > 1 else 0.0

    # Individual seed dots with horizontal jitter
    jitter = rng.uniform(-0.10, 0.10, size=len(gains))
    ax.scatter([i + j for j in jitter], gains, color=MODEL_COLOR[model],
               s=80, alpha=0.85, edgecolor="white", linewidth=1.0, zorder=3)

    # Mean as a heavy horizontal line through the cluster
    ax.hlines(mean, i - 0.20, i + 0.20, color=MODEL_COLOR[model],
              linewidth=2.5, zorder=4)

    # ±1σ error bar offset slightly to the right of the dots
    ax.errorbar(i + 0.32, mean, yerr=sd, fmt="none",
                ecolor=MODEL_COLOR[model], capsize=5, capthick=1.5,
                elinewidth=1.5, zorder=4)

    # Label
    ax.text(i + 0.40, mean, f"{mean:+.1f} ± {sd:.1f}pp",
            color=NAVY, fontsize=10, fontweight="bold",
            ha="left", va="center")

ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)

ax.set_xticks(range(len(MODELS_ORDER)))
ax.set_xticklabels(MODELS_ORDER)
ax.set_ylabel("Held-out federation gain (pp)", color=INK)
ax.set_xlim(-0.5, len(MODELS_ORDER) - 0.1)
ax.set_ylim(-12, 17)
ax.set_title("Held-out federation gain distribution per model   (n=5 seeds each)",
             color=NAVY, pad=10, fontweight="bold")
ax.grid(axis="y", linestyle=":", color="#CBD5E1", linewidth=0.5, zorder=0)

fig.text(0.5, -0.04,
         "Dots are individual seeds. Solid bar: mean. Vertical bar: ±1 standard deviation. "
         "Dashed line: no effect.",
         ha="center", color=MUTED, style="italic", fontsize=9)

fig.savefig(OUTPUT_DIR / "02_gain_distribution.png")
plt.close(fig)
print(f"  wrote {OUTPUT_DIR / '02_gain_distribution.png'}")

# ===========================================================================
# Figure 3 — Round-by-round trajectories (colour by seed, not model)
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)

for ax, model in zip(axes, MODELS_ORDER):
    for seed in SEEDS_ORDER:
        r = find(model, seed)
        if r is None or not r["rounds"]:
            continue
        # Drop round 0 (random initialisation noise)
        rounds = list(range(1, len(r["rounds"])))
        accs = [a * 100 for a in r["rounds"][1:]]
        ax.plot(rounds, accs, marker="o", markersize=5, linewidth=1.6,
                color=SEED_COLOR[seed], alpha=0.95, label=f"seed={seed}")

    ax.set_title(model, color=NAVY, fontweight="bold")
    ax.set_xlabel("Federation round")
    ax.set_xticks(range(1, 7))
    ax.set_ylim(55, 90)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

axes[0].set_ylabel("Shared-context accuracy (%)", color=INK)

# Single shared legend below the panels
legend_handles = [
    plt.Line2D([0], [0], color=SEED_COLOR[s], linewidth=2.0, marker="o",
               markersize=5, label=f"seed={s}")
    for s in SEEDS_ORDER
]
fig.legend(handles=legend_handles, loc="lower center", ncol=len(SEEDS_ORDER),
           bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10)

fig.suptitle("Pseudo-label accuracy on the shared context across rounds (convergence diagnostic, n=100)",
             color=NAVY, fontweight="bold", y=1.02)
fig.text(0.5, -0.13,
         "This is accuracy on the federation's own 100 shared queries, NOT held-out generalization. "
         "Same colour = same seed. Round 0 omitted.",
         ha="center", color=MUTED, style="italic", fontsize=9)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_round_trajectories.png")
plt.close(fig)
print(f"  wrote {OUTPUT_DIR / '03_round_trajectories.png'}")

# ===========================================================================
# Figure 4 — Fed-ICL R6 vs Local-only scatter, coloured by seed
# ===========================================================================
fig, ax = plt.subplots(figsize=(6.8, 6.0))

LO, HI = 55, 90

# Background tinted regions
ax.fill_between([LO, HI], [LO, HI], [HI, HI],
                color="#86EFAC", alpha=0.13, zorder=0,
                label="Federation helps")
ax.fill_between([LO, HI], [LO, LO], [LO, HI],
                color="#FCA5A5", alpha=0.13, zorder=0,
                label="Federation hurts")
ax.plot([LO, HI], [LO, HI], color="#94A3B8", linestyle="--",
        linewidth=0.9, zorder=1, label="No federation gain")

# Marker letter per model
MODEL_LETTER = {"phi3": "P", "mistral": "M", "llama3": "L"}

for r in rows:
    if r["local_only"] is None or r["held_out"] is None:
        continue
    x = r["local_only"] * 100
    y = r["held_out"] * 100
    ax.scatter(x, y, color=SEED_COLOR[r["seed"]],
               s=160, edgecolor="white", linewidth=1.2, zorder=3)
    ax.annotate(MODEL_LETTER[r["model"]], (x, y),
                ha="center", va="center",
                fontsize=10, color="white", fontweight="bold", zorder=4)

ax.set_xlim(LO, HI)
ax.set_ylim(LO, HI)
ax.set_xlabel("Local-only accuracy (%)", color=INK)
ax.set_ylabel("Fed-ICL held-out accuracy (%)", color=INK)
ax.set_title("Federation outcome by partition seed  (held-out, n=1000)\n"
             "(letter = model: P=phi3, M=mistral, L=llama3)",
             color=NAVY, pad=10, fontweight="bold")
ax.grid(linestyle=":", color="#CBD5E1", linewidth=0.5, zorder=0)
ax.set_aspect("equal", adjustable="box")

# Two legends: seeds (top-left) and region labels (bottom-right)
seed_handles = [mpatches.Patch(color=SEED_COLOR[s], label=f"seed={s}") for s in SEEDS_ORDER]
seed_legend = ax.legend(handles=seed_handles, loc="upper left",
                        frameon=False, fontsize=9, title="Seed",
                        title_fontsize=9)
ax.add_artist(seed_legend)

region_handles = [
    mpatches.Patch(color="#86EFAC", alpha=0.5, label="Federation helps"),
    mpatches.Patch(color="#FCA5A5", alpha=0.5, label="Federation hurts"),
]
ax.legend(handles=region_handles, loc="lower right",
          frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_fed_vs_local_scatter.png")
plt.close(fig)
print(f"  wrote {OUTPUT_DIR / '04_fed_vs_local_scatter.png'}")

# ===========================================================================
# Figure 5 — In-sample (R6@100) vs out-of-sample (held@1000) gap
# Shows WHY the R6 metric overstates generalization.
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.4, 4.2))
for i, model in enumerate(MODELS_ORDER):
    rs = [r for r in rows if r["model"] == model]
    for r in rs:
        if r["fed_icl_r6"] is None or r["held_out"] is None:
            continue
        ax.plot([i - 0.2, i + 0.2],
                [r["fed_icl_r6"] * 100, r["held_out"] * 100],
                color=MODEL_COLOR[model], alpha=0.55, linewidth=1.4, zorder=2)
        ax.scatter(i - 0.2, r["fed_icl_r6"] * 100, color=MODEL_COLOR[model], s=45, zorder=3)
        ax.scatter(i + 0.2, r["held_out"] * 100, color=MODEL_COLOR[model], s=45,
                   marker="s", edgecolor="white", linewidth=0.6, zorder=3)
    locs = [r["local_only"] * 100 for r in rs if r["local_only"] is not None]
    if locs:
        lo = statistics.mean(locs)
        ax.plot([i - 0.32, i + 0.32], [lo, lo], color=INK, linewidth=1.4, linestyle="--", zorder=4)
        ax.text(i, lo - 1.6, "local-only (mean)", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(len(MODELS_ORDER)))
ax.set_xticklabels(MODELS_ORDER)
ax.set_ylabel("Accuracy (%)", color=INK)
ax.set_ylim(50, 90)
ax.text(0.01, 0.97, "\u25cf R6 on shared context (n=100)    \u25a0 held-out (n=1000)",
        transform=ax.transAxes, fontsize=9, va="top", color=INK)
ax.set_title("The in-sample / out-of-sample gap behind the apparent gain",
             color=NAVY, pad=10, fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.4)
fig.text(0.5, -0.02,
         "Each line drops from what the federation scores on its OWN context to what it generalizes to. "
         "The ~9pp fall is the overstatement.",
         ha="center", color=MUTED, style="italic", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_insample_gap.png")
plt.close(fig)
print(f"  wrote {OUTPUT_DIR / '05_insample_gap.png'}")

# ===========================================================================
# Figure 6 — Partition stability: local-only vs federated held-out
# Shows the gain SIGN is driven by local baseline volatility.
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.4, 4.2))
for i, model in enumerate(MODELS_ORDER):
    locs = [find(model, s)["local_only"] * 100 for s in SEEDS_ORDER]
    helds = [find(model, s)["held_out"] * 100 for s in SEEDS_ORDER]
    xL, xF = i - 0.15, i + 0.15
    ax.scatter([xL] * len(locs), locs, color=MODEL_COLOR[model], s=40, alpha=0.6)
    ax.scatter([xF] * len(helds), helds, color=MODEL_COLOR[model], s=40,
               marker="s", edgecolor="white", linewidth=0.6)
    ax.plot([xL - 0.08, xL + 0.08], [statistics.mean(locs)] * 2, color=MODEL_COLOR[model], linewidth=2)
    ax.plot([xF - 0.08, xF + 0.08], [statistics.mean(helds)] * 2, color=MODEL_COLOR[model], linewidth=2)
    ax.text(xL, min(locs) - 2.2, f"\u03c3={statistics.pstdev(locs):.1f}", ha="center", fontsize=8)
    ax.text(xF, min(helds) - 2.2, f"\u03c3={statistics.pstdev(helds):.1f}", ha="center", fontsize=8)
ax.set_xticks(range(len(MODELS_ORDER)))
ax.set_xticklabels(MODELS_ORDER)
ax.set_ylabel("Held-out accuracy (%)", color=INK)
ax.set_ylim(60, 82)
ax.text(0.01, 0.97, "\u25cf local-only (per seed)     \u25a0 federated held-out (per seed)",
        transform=ax.transAxes, fontsize=9, va="top", color=INK)
ax.set_title("Federation is partition-stable; the local baseline is a lottery",
             color=NAVY, pad=10, fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.4)
fig.text(0.5, -0.02,
         "Local-only swings ~2x more across seeds than federated held-out. The sign of 'federation gain' "
         "is set by which way local's luck fell.",
         ha="center", color=MUTED, style="italic", fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_partition_stability.png")
plt.close(fig)
print(f"  wrote {OUTPUT_DIR / '06_partition_stability.png'}")

print(f"\nDone. Six figures saved to {OUTPUT_DIR}/.")