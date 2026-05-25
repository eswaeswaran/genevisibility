#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

# =========================
# CONFIG
# =========================

BASE = Path("/Users/mtongoss/Dropbox/Research/projects/MostStudiedGenes/FilesFromStudents/4string/codeml_outputs_v5/pairwise")

FILES = {
    "lowest500": BASE / "lowest500/dnds.lowest500.tsv",
    "randomBelow10k": BASE / "randomBelow10k/dnds.randomBelow10k.tsv",
    "randomAbove10k": BASE / "randomAbove10k/dnds.randomAbove10k.tsv",
    "top500": BASE / "top500/dnds.top500.tsv",
}

# order in plots / tables (adjust if desired)
GROUP_ORDER = ["lowest500", "randomBelow10k", "randomAbove10k", "top500"]

OUTPUT_DIR = BASE / "publication_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["dN", "dS", "dN_dS"]

# plotting + transform
LOG2_PSEUDOCOUNT = 1e-8   # used for plotting (and optionally stats)
PLOT_POINTS_MAX = 1200    # cap number of jittered points per group to avoid huge files (random subsample)
POINT_ALPHA = 0.25
POINT_SIZE = 7

# winsorize (helps with extreme tails); set to None to disable
# Example: (0.01, 0.99) clips to 1st–99th percentile after transform.
WINSORIZE_QUANTILES = (0.01, 0.99)

# stats
STATS_ON = "raw"          # "raw" or "log2" (log2 uses metric+LOG2_PSEUDOCOUNT)
FDR_Q = 0.10              # q-threshold for calling significance (0.05, 0.1, 0.2 ...)
BOOTSTRAP_N = 4000        # for median/mean CIs; 2000 ok if you want faster
RNG_SEED = 7

# =========================
# HELPERS
# =========================

rng = np.random.default_rng(RNG_SEED)

def read_and_clean_one(group: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.replace("NA", np.nan)
    df = df.dropna(subset=METRICS)
    for m in METRICS:
        df[m] = df[m].astype(float)
    df["group"] = group
    return df

def log2_transform(x: np.ndarray) -> np.ndarray:
    return np.log2(x + LOG2_PSEUDOCOUNT)

def winsorize(x: np.ndarray, q: tuple[float, float] | None) -> np.ndarray:
    if q is None:
        return x
    lo, hi = np.quantile(x, q[0]), np.quantile(x, q[1])
    return np.clip(x, lo, hi)

def bootstrap_ci(x: np.ndarray, func, n: int = BOOTSTRAP_N, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic."""
    x = np.asarray(x)
    if len(x) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = np.array([func(x[i]) for i in idx], dtype=float)
    lo = np.quantile(vals, alpha/2)
    hi = np.quantile(vals, 1 - alpha/2)
    return float(lo), float(hi)

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size in [-1,1]."""
    x = np.asarray(x)
    y = np.asarray(y)
    # vectorized-ish approach using ranks can be tricky with ties; do direct but safe for modest N
    # If your N is huge per group, we can swap to an O(n log n) implementation.
    gt = 0
    lt = 0
    for xv in x:
        gt += np.sum(xv > y)
        lt += np.sum(xv < y)
    return float((gt - lt) / (len(x) * len(y)))

def hodges_lehmann(x: np.ndarray, y: np.ndarray) -> float:
    """Hodges–Lehmann estimator of shift (median of all pairwise differences)."""
    x = np.asarray(x)
    y = np.asarray(y)
    diffs = (x[:, None] - y[None, :]).ravel()
    return float(np.median(diffs))

def format_float(x: float) -> str:
    if np.isnan(x):
        return "NA"
    # compact but readable
    if abs(x) < 1e-3 and x != 0:
        return f"{x:.2e}"
    return f"{x:.4g}"

# =========================
# LOAD DATA
# =========================

dfs = []
for g in GROUP_ORDER:
    df = read_and_clean_one(g, FILES[g])
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# =========================
# STATS TABLES
# =========================

def per_group_descriptives(df: pd.DataFrame, metric: str, scale: str) -> pd.DataFrame:
    rows = []
    for g in GROUP_ORDER:
        vals = df.loc[df["group"] == g, metric].to_numpy()

        if scale == "log2":
            vals_s = log2_transform(vals)
        else:
            vals_s = vals

        vals_s = winsorize(vals_s, WINSORIZE_QUANTILES)

        N = len(vals_s)
        mean = float(np.mean(vals_s))
        median = float(np.median(vals_s))
        sd = float(np.std(vals_s, ddof=1)) if N > 1 else np.nan
        q25 = float(np.quantile(vals_s, 0.25))
        q75 = float(np.quantile(vals_s, 0.75))
        iqr = q75 - q25

        mean_ci = bootstrap_ci(vals_s, np.mean)
        median_ci = bootstrap_ci(vals_s, np.median)

        rows.append({
            "metric": metric,
            "scale": scale,
            "group": g,
            "N": N,
            "mean": mean,
            "mean_ci_low": mean_ci[0],
            "mean_ci_high": mean_ci[1],
            "median": median,
            "median_ci_low": median_ci[0],
            "median_ci_high": median_ci[1],
            "sd": sd,
            "q25": q25,
            "q75": q75,
            "IQR": iqr,
        })
    return pd.DataFrame(rows)

def omnibus_and_pairwise(df: pd.DataFrame, metric: str, scale: str) -> tuple[pd.DataFrame, dict]:
    # arrays per group
    group_vals = []
    for g in GROUP_ORDER:
        v = df.loc[df["group"] == g, metric].to_numpy()
        if scale == "log2":
            v = log2_transform(v)
        v = winsorize(v, WINSORIZE_QUANTILES)
        group_vals.append(v)

    # omnibus tests
    anova_F, anova_p = stats.f_oneway(*group_vals)
    kw_H, kw_p = stats.kruskal(*group_vals)

    # pairwise MWU + FDR + effect sizes
    pairs = []
    raw_ps = []
    for i in range(len(GROUP_ORDER)):
        for j in range(i+1, len(GROUP_ORDER)):
            g1, g2 = GROUP_ORDER[i], GROUP_ORDER[j]
            x, y = group_vals[i], group_vals[j]

            U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            raw_ps.append(p)

            # effect sizes
            hl = hodges_lehmann(x, y)     # shift estimate (x - y)
            cd = cliffs_delta(x, y)

            # bootstrap CI for HL (median shift) via resampling
            # (lighter alternative: CI for median difference; HL is robust and standard)
            def hl_stat(idx_xy):
                # idx_xy is a tuple of indices (not used); we resample directly below
                return 0.0

            # bootstrap HL CI
            # To keep it simple + robust: bootstrap medians separately and subtract.
            # (If you prefer true HL CI, I can swap in a heavier bootstrap of pairwise diffs.)
            med_ci_x = bootstrap_ci(x, np.median)
            med_ci_y = bootstrap_ci(y, np.median)
            med_diff = float(np.median(x) - np.median(y))
            med_diff_ci = (med_ci_x[0] - med_ci_y[1], med_ci_x[1] - med_ci_y[0])

            pairs.append({
                "metric": metric,
                "scale": scale,
                "group1": g1,
                "group2": g2,
                "MWU_U": float(U),
                "p_raw": float(p),
                "HL_shift_est_(g1-g2)": hl,
                "median_diff_(g1-g2)": med_diff,
                "median_diff_ci_low": med_diff_ci[0],
                "median_diff_ci_high": med_diff_ci[1],
                "cliffs_delta_(g1_vs_g2)": cd,
            })

    reject, p_fdr, _, _ = multipletests(raw_ps, method="fdr_bh", alpha=FDR_Q)

    for k, (r, qv) in enumerate(zip(reject, p_fdr)):
        pairs[k]["q_fdr"] = float(qv)
        pairs[k]["significant_at_q"] = bool(r)

    pairwise_df = pd.DataFrame(pairs)

    omnibus = {
        "metric": metric,
        "scale": scale,
        "anova_F": float(anova_F),
        "anova_p": float(anova_p),
        "kruskal_H": float(kw_H),
        "kruskal_p": float(kw_p),
        "FDR_q_threshold": float(FDR_Q),
    }
    return pairwise_df, omnibus

# write stats tables
all_desc = []
all_pairwise = []
omnibus_rows = []

for m in METRICS:
    desc = per_group_descriptives(df_all, m, STATS_ON)
    all_desc.append(desc)

    pw, om = omnibus_and_pairwise(df_all, m, STATS_ON)
    all_pairwise.append(pw)
    omnibus_rows.append(om)

desc_df = pd.concat(all_desc, ignore_index=True)
pairwise_df = pd.concat(all_pairwise, ignore_index=True)
omnibus_df = pd.DataFrame(omnibus_rows)

desc_df.to_csv(OUTPUT_DIR / "descriptive_stats.tsv", sep="\t", index=False)
pairwise_df.to_csv(OUTPUT_DIR / "pairwise_stats.tsv", sep="\t", index=False)
omnibus_df.to_csv(OUTPUT_DIR / "omnibus_tests.tsv", sep="\t", index=False)

# =========================
# PUBLICATION PLOTS (RAIN-CLOUD STYLE)
# =========================

def raincloud_axis(ax, metric: str, plot_scale: str = "log2") -> None:
    # pull data
    vals_by_group = []
    for g in GROUP_ORDER:
        v = df_all.loc[df_all["group"] == g, metric].to_numpy()
        if plot_scale == "log2":
            v = log2_transform(v)
        v = winsorize(v, WINSORIZE_QUANTILES)
        vals_by_group.append(v)

    positions = np.arange(1, len(GROUP_ORDER) + 1)

    # violin (matplotlib draws symmetric; we "fake" half by masking one side visually using x-limits and offset)
    vp = ax.violinplot(vals_by_group, positions=positions, widths=0.85,
                       showmeans=False, showmedians=False, showextrema=False)

    # make violins subtle (no explicit color choices; keep edge)
    for b in vp["bodies"]:
        b.set_alpha(0.25)

    # boxplot overlay
    bp = ax.boxplot(vals_by_group, positions=positions, widths=0.18,
                    showfliers=False, patch_artist=False)

    # jittered points + mean marker
    means = []
    medians = []
    Ns = []
    for i, v in enumerate(vals_by_group):
        Ns.append(len(v))
        means.append(float(np.mean(v)))
        medians.append(float(np.median(v)))

        # subsample for plotting if large
        vv = v
        if len(vv) > PLOT_POINTS_MAX:
            vv = rng.choice(vv, size=PLOT_POINTS_MAX, replace=False)

        # jitter
        xj = rng.normal(loc=positions[i], scale=0.06, size=len(vv))
        ax.scatter(xj, vv, s=POINT_SIZE, alpha=POINT_ALPHA, marker="o", linewidths=0)

    ax.scatter(positions, means, marker="D", s=28, label="Mean")
    # median is already the boxplot line; add small underscore to make it obvious
    ax.scatter(positions, medians, marker="_", s=350, label="Median")

    # x labels with N, mean, median (compact)
    xlabels = []
    for g, n, mu, md in zip(GROUP_ORDER, Ns, means, medians):
        xlabels.append(f"{g}\nN={n}\nmean={format_float(mu)}\nmed={format_float(md)}")
    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, rotation=0)

    ylabel = metric if plot_scale == "raw" else f"log2({metric} + {LOG2_PSEUDOCOUNT:g})"
    ax.set_ylabel(ylabel)

    # add significance stars (based on STATS_ON scale + FDR_q)
    # We'll annotate above the plot using pairwise_df for this metric.
    pw_m = pairwise_df[(pairwise_df["metric"] == metric) & (pairwise_df["scale"] == STATS_ON)].copy()
    # Map group->position
    pos_map = {g: p for g, p in zip(GROUP_ORDER, positions)}

    # Decide y-height for annotations
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    base = y_max + 0.03 * y_span
    step = 0.06 * y_span

    # sort by q-value (most significant first) and only annotate significant
    pw_m = pw_m[pw_m["q_fdr"] <= FDR_Q].sort_values("q_fdr")

    # prevent overlaps: keep a simple "used" dictionary by pair width
    level = 0
    for _, row in pw_m.iterrows():
        g1 = row["group1"]
        g2 = row["group2"]
        x1, x2 = pos_map[g1], pos_map[g2]
        if x1 > x2:
            x1, x2 = x2, x1
        y = base + level * step

        qv = float(row["q_fdr"])
        # star mapping
        if qv <= 0.001:
            stars = "***"
        elif qv <= 0.01:
            stars = "**"
        elif qv <= 0.05:
            stars = "*"
        else:
            stars = "·"  # significant at q=0.1 but not 0.05 etc.

        # bracket
        ax.plot([x1, x1, x2, x2], [y, y + 0.01*y_span, y + 0.01*y_span, y],
                linewidth=1)
        ax.text((x1 + x2) / 2, y + 0.012*y_span, f"{stars} (q={qv:.3g})",
                ha="center", va="bottom", fontsize=9)
        level += 1

    # extend ylim if we added annotations
    if level > 0:
        ax.set_ylim(y_min, base + (level + 1) * step)

def make_figure(plot_scale: str = "log2") -> None:
    # 3-panel publication figure
    fig = plt.figure(figsize=(11, 10))
    axes = [
        fig.add_subplot(3, 1, 1),
        fig.add_subplot(3, 1, 2),
        fig.add_subplot(3, 1, 3),
    ]

    titles = {
        "dN": "Nonsynonymous rate (dN)",
        "dS": "Synonymous rate (dS)",
        "dN_dS": "Omega (dN/dS)",
    }

    for ax, metric in zip(axes, METRICS):
        raincloud_axis(ax, metric, plot_scale=plot_scale)
        ax.set_title(titles.get(metric, metric), fontsize=12)
        ax.grid(False)

    # single legend (mean/median)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)

    fig.suptitle(
        f"Pairwise codeml rates across gene groups (plot={plot_scale}; stats={STATS_ON}; FDR q={FDR_Q})",
        fontsize=13,
        y=0.99
    )

    fig.tight_layout(rect=[0, 0, 0.98, 0.975])

    fig.savefig(OUTPUT_DIR / f"Figure_dnds_groups_{plot_scale}.pdf")
    fig.savefig(OUTPUT_DIR / f"Figure_dnds_groups_{plot_scale}.png", dpi=300)
    plt.close(fig)

def make_per_metric_figs(plot_scale: str = "log2") -> None:
    for metric in METRICS:
        fig = plt.figure(figsize=(9.5, 6.5))
        ax = fig.add_subplot(1, 1, 1)
        raincloud_axis(ax, metric, plot_scale=plot_scale)
        ax.set_title(f"{metric} across groups (plot={plot_scale}; stats={STATS_ON}; FDR q={FDR_Q})", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False)
        fig.tight_layout(rect=[0, 0, 0.98, 0.98])
        fig.savefig(OUTPUT_DIR / f"{metric}_raincloud_{plot_scale}.pdf")
        fig.savefig(OUTPUT_DIR / f"{metric}_raincloud_{plot_scale}.png", dpi=300)
        plt.close(fig)

# generate figures on log2 scale (recommended)
make_figure(plot_scale="log2")
make_per_metric_figs(plot_scale="log2")

# =========================
# WRITE A SHORT HUMAN-READABLE SUMMARY
# =========================

summary_txt = []
summary_txt.append(f"Stats scale: {STATS_ON}\nPlot scale: log2\nFDR q-threshold: {FDR_Q}\n")
summary_txt.append("Omnibus tests:\n")
for _, r in omnibus_df.iterrows():
    summary_txt.append(
        f"- {r['metric']} ({r['scale']}): "
        f"ANOVA p={r['anova_p']:.3g}, Kruskal p={r['kruskal_p']:.3g}\n"
    )
summary_txt.append("\nSignificant pairwise (q <= threshold):\n")
sig = pairwise_df[pairwise_df["q_fdr"] <= FDR_Q].copy()
if len(sig) == 0:
    summary_txt.append("  (none)\n")
else:
    for _, r in sig.sort_values(["metric", "q_fdr"]).iterrows():
        summary_txt.append(
            f"- {r['metric']} [{r['scale']}]: {r['group1']} vs {r['group2']}: "
            f"q={r['q_fdr']:.3g}, HL={r['HL_shift_est_(g1-g2)']:.3g}, "
            f"Δmed={r['median_diff_(g1-g2)']:.3g} "
            f"(CI {r['median_diff_ci_low']:.3g}..{r['median_diff_ci_high']:.3g}), "
            f"Cliff's δ={r['cliffs_delta_(g1_vs_g2)']:.3g}\n"
        )

(OUTPUT_DIR / "summary_readme.txt").write_text("".join(summary_txt), encoding="utf-8")

print("Wrote publication figures + stats to:", OUTPUT_DIR)
print("Files:")
print("  - Figure_dnds_groups_log2.pdf / .png")
print("  - dN_raincloud_log2.pdf / .png (and dS, dN_dS)")
print("  - descriptive_stats.tsv")
print("  - pairwise_stats.tsv")
print("  - omnibus_tests.tsv")
print("  - summary_readme.txt")
