"""
plot_pool_sweep.py
==================
Three figures for the pool-size sweep analysis.

  01_pool_vs_gain.png         : Federation gain vs pool size, one line per model.
  02_pool_seed_heatmap.png    : Pool x seed gain heatmap, three panels (one per model).
  03_local_vs_fed_curves.png  : Local-only and Fed-ICL R6 as paired curves per model.

Conditions with n=1 (no std possible) are excluded from aggregate plots but kept
in the per-condition heatmap.

Run from the directory containing the result JSON files:
    python3 plot_pool_sweep.py
"""
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = "."
OUTPUT_DIR  = Path("plots")
MODELS_ORDER = ["phi3", "mistral", "llama3"]
SEEDS_ORDER  = [3, 7, 13, 42, 99]
POOLS_ORDER  = [30, 60, 120, 250, 500]
DEFAULT_POOL = 250
MIN_SEEDS_FOR_AGGREGATE = 2  # exclude n=1 conditions from aggregate plots

# Palette consistent with multi-seed figures
NAVY  = "#21295C"
TEAL  = "#1C7293"
DEEP  = "#065A82"
GREEN = "#0E7C66"
RED   = "#B91C1C"
AMBER = "#B45309"
MUTED = "#64748B"
INK   = "#1E293B"

MODEL_COLOR = {"phi3": AMBER, "mistral": TEAL, "llama3": DEEP}

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
PATTERN_NEW = re.compile(
    r"results_(?P<model>[a-z0-9]+)_alpha(?P<alpha>[0-9.]+)_K(?P<K>\d+)_T(?P<T>\d+)"
    r"_pool(?P<pool>\d+)_seed(?P<seed>\d+)_order-(?P<order>[a-z_]+)\.json"
)
PATTERN_OLD = re.compile(
    r"results_(?P<model>[a-z0-9]+)_alpha(?P<alpha>[0-9.]+)_K(?P<K>\d+)_T(?P<T>\d+)"
    r"_seed(?P<seed>\d+)_order-(?P<order>[a-z_]+)\.json"
)

rows = []
for f in sorted(Path(RESULTS_DIR).glob("results_*alpha0.5_K3_T6*.json")):
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
    rs = [r["accuracy"] for r in d.get("rounds", [])]
    r6 = rs[-1] if rs else None
    local = base.get("local_only")
    rows.append({
        "model": m.group("model"),
        "seed": int(m.group("seed")),
        "pool": pool,
        "local_only": local,
        "fed_icl_r6": r6,
        "gain": (r6 - local) if (r6 is not None and local is not None) else None,
    })

if not rows:
    raise SystemExit("No matching result files found.")

OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Loaded {len(rows)} result files.")


def aggregate(model, pool):
    """Mean and std of gain, local_only, and r6 for a (model, pool) cell."""
    cell = [r for r in rows if r["model"] == model and r["pool"] == pool]
    if len(cell) < MIN_SEEDS_FOR_AGGREGATE:
        return None
    return {
        "n":        len(cell),
        "gain_m":   statistics.mean(r["gain"] for r in cell) * 100,
        "gain_s":   statistics.stdev(r["gain"] for r in cell) * 100 if len(cell) > 1 else 0.0,
        "local_m":  statistics.mean(r["local_only"] for r in cell) * 100,
        "local_s":  statistics.stdev(r["local_only"] for r in cell) * 100 if len(cell) > 1 else 0.0,
        "r6_m":     statistics.mean(r["fed_icl_r6"] for r in cell) * 100,
        "r6_s":     statistics.stdev(r["fed_icl_r6"] for r in cell) * 100 if len(cell) > 1 else 0.0,
    }

# ===========================================================================
# Figure 1: Pool-vs-gain curves
# ===========================================================================
fig, ax = plt.subplots(figsize=(8.5, 4.5))

for model in MODELS_ORDER:
    xs, means, stds = [], [], []
    for pool in POOLS_ORDER:
        agg = aggregate(model, pool)
        if agg is None:
            continue
        xs.append(pool)
        means.append(agg["gain_m"])
        stds.append(agg["gain_s"])

    if not xs:
        continue
    xs = np.array(xs)
    means = np.array(means)
    stds = np.array(stds)

    # Shaded std band
    ax.fill_between(xs, means - stds, means + stds,
                    color=MODEL_COLOR[model], alpha=0.15, zorder=2)
    # Mean line with markers
    ax.plot(xs, means, marker="o", markersize=7, linewidth=2,
            color=MODEL_COLOR[model], label=model, zorder=3)

ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--", zorder=1)
ax.set_xscale("log")
ax.set_xticks(POOLS_ORDER)
ax.set_xticklabels([str(p) for p in POOLS_ORDER])
ax.minorticks_off()
ax.set_xlabel("Client pool size (log scale)", color=INK)
ax.set_ylabel("Federation gain (pp)", color=INK)
ax.set_title("Federation gain by pool size and model   (α=0.5, K=3, T=6, n=100)",
             color=NAVY, pad=10, fontweight="bold")
ax.grid(axis="y", linestyle=":", color="#CBD5E1", linewidth=0.5, zorder=0)
ax.legend(loc="lower right", frameon=False, title="Model", title_fontsize=10)

fig.text(0.5, -0.04,
         "Solid line: mean gain across seeds. Shaded band: ±1 standard deviation. "
         "Dashed line: no effect.",
         ha="center", color=MUTED, style="italic", fontsize=9)

fig.savefig(OUTPUT_DIR / "01_pool_vs_gain.png")
plt.close(fig)
print(f"  wrote {OUTPUT_DIR / '01_pool_vs_gain.png'}")

# ===========================================================================
# Figure 2: Pool x seed heatmap (3 panels)
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)

# Determine common colour scale across all three panels
all_gains = [r["gain"] * 100 for r in rows if r["gain"] is not None]
vmax = max(abs(min(all_gains)), abs(max(all_gains)))
cmap = mcolors.LinearSegmentedColormap.from_list(
    "fed_icl_div", [RED, "#FFFFFF", GREEN], N=256
)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

for ax, model in zip(axes, MODELS_ORDER):
    grid = np.full((len(SEEDS_ORDER), len(POOLS_ORDER)), np.nan)
    for i, seed in enumerate(SEEDS_ORDER):
        for j, pool in enumerate(POOLS_ORDER):
            match = [r for r in rows if r["model"] == model and r["pool"] == pool and r["seed"] == seed]
            if match and match[0]["gain"] is not None:
                grid[i, j] = match[0]["gain"] * 100

    im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")

    # Annotate each cell
    for i in range(len(SEEDS_ORDER)):
        for j in range(len(POOLS_ORDER)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                        color=MUTED, fontsize=8, style="italic")
                continue
            text_color = INK if abs(v) < vmax * 0.55 else "white"
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(POOLS_ORDER)))
    ax.set_xticklabels([str(p) for p in POOLS_ORDER])
    ax.set_xlabel("Pool size")
    ax.set_title(model, color=NAVY, fontweight="bold")

axes[0].set_yticks(range(len(SEEDS_ORDER)))
axes[0].set_yticklabels([f"seed={s}" for s in SEEDS_ORDER])

cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02,
                    label="Federation gain (pp)")
cbar.ax.tick_params(labelsize=9)

fig.suptitle("Federation gain by seed and pool size, per model",
             color=NAVY, fontweight="bold", y=1.05)
fig.text(0.5, -0.05,
         "Each cell is one (model, pool, seed) experiment. Cells marked n/a were not run at this configuration.",
         ha="center", color=MUTED, style="italic", fontsize=9)

fig.savefig(OUTPUT_DIR / "02_pool_seed_heatmap.png")
plt.close(fig)
print(f"  wrote {OUTPUT_DIR / '02_pool_seed_heatmap.png'}")

# ===========================================================================
# Figure 3: Local-only and Fed-ICL R6 as paired curves
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)

for ax, model in zip(axes, MODELS_ORDER):
    xs, lo_m, lo_s, r6_m, r6_s = [], [], [], [], []
    for pool in POOLS_ORDER:
        agg = aggregate(model, pool)
        if agg is None:
            continue
        xs.append(pool)
        lo_m.append(agg["local_m"])
        lo_s.append(agg["local_s"])
        r6_m.append(agg["r6_m"])
        r6_s.append(agg["r6_s"])

    if not xs:
        continue
    xs = np.array(xs)
    lo_m, lo_s = np.array(lo_m), np.array(lo_s)
    r6_m, r6_s = np.array(r6_m), np.array(r6_s)

    # Local-only as a lighter, dashed line
    ax.fill_between(xs, lo_m - lo_s, lo_m + lo_s,
                    color=MUTED, alpha=0.12, zorder=2)
    ax.plot(xs, lo_m, marker="s", markersize=6, linewidth=1.8,
            linestyle="--", color=MUTED, label="Local-only", zorder=3)

    # Fed-ICL R6 as the model colour, solid
    ax.fill_between(xs, r6_m - r6_s, r6_m + r6_s,
                    color=MODEL_COLOR[model], alpha=0.18, zorder=2)
    ax.plot(xs, r6_m, marker="o", markersize=7, linewidth=2.2,
            color=MODEL_COLOR[model], label="Fed-ICL R6", zorder=3)

    ax.set_xscale("log")
    ax.set_xticks(POOLS_ORDER)
    ax.set_xticklabels([str(p) for p in POOLS_ORDER])
    ax.minorticks_off()
    ax.set_xlabel("Client pool size (log scale)")
    ax.set_title(model, color=NAVY, fontweight="bold")
    ax.grid(axis="y", linestyle=":", color="#CBD5E1", linewidth=0.5, zorder=0)
    ax.set_ylim(50, 90)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

axes[0].set_ylabel("Accuracy (%)", color=INK)

fig.suptitle("Local-only baseline vs Fed-ICL R6, across pool sizes",
             color=NAVY, fontweight="bold", y=1.03)
fig.text(0.5, -0.04,
         "Each panel is one model. Solid line: Fed-ICL after 6 rounds. Dashed line: local-only baseline. "
         "Shaded: ±1 std across seeds.",
         ha="center", color=MUTED, style="italic", fontsize=9)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_local_vs_fed_curves.png")
plt.close(fig)
print(f"  wrote {OUTPUT_DIR / '03_local_vs_fed_curves.png'}")

print(f"\nDone. Three figures saved to {OUTPUT_DIR}/.")