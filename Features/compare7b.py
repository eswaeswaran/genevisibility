#!/usr/bin/env python3
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt


# ----------------- EDIT THESE PATHS -----------------
TOP_TSV        = Path("top500_mane_select_metrics.tsv")
LOWEST_TSV     = Path("lowest500_mane_select_metrics.tsv")
RANDOM_TOP_TSV = Path("random500_mane_select_metrics.tsv")  # random from top 10k
RANDOM_LOW_TSV = Path("random_below10000_500_min10_mane_select_metrics.tsv")  # random below 10k with min 10 pubs
# ----------------------------------------------------

# NEW output folder
OUT_DIR = Path("plots_all4_len_gc_ad_mode")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORDER = ["top", "lowest", "random_top10k", "random_below10k"]

# Only the two stats you want
METRICS = [
    "cds_len_bp_from_protein",  # coding DNA length proxy from protein length
    "mrna_gc_percent",          # GC%
]

LOG_SCALE_METRICS = {"cds_len_bp_from_protein"}  # x log for length plots

USE_FDR = True
ALPHA = 0.05
MIN_N_FOR_TESTS = 10

# KDE / mode params (keep fixed across groups for comparability)
KDE_BW_ADJUST = 0.8
MODE_GRID_N = 2000

# Bootstrap params
BOOT_N = 5000
BOOT_SEED = 7

sns.set_theme(style="whitegrid", context="talk")
DPI = 300


# ---------- IO ----------
def read_tsv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    df = pd.read_csv(p, sep="\t", dtype=str)
    df = df.replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan})
    if "gene" not in df.columns:
        raise ValueError(f"{p} must contain a 'gene' column")

    for c in METRICS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------- multiple testing ----------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = (ranked[i] * n) / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


# ---------- stats: AD + mode bootstrap ----------
def ad_ksamp_pvalue(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Returns (statistic, pvalue_approx).
    SciPy's anderson_ksamp returns an approximate pvalue or a significance_level.
    """
    res = stats.anderson_ksamp([x, y])
    stat = float(res.statistic)

    # SciPy versions differ; handle robustly
    if hasattr(res, "pvalue") and res.pvalue is not None:
        p = float(res.pvalue)
    else:
        # significance_level is usually in percent
        sig = getattr(res, "significance_level", np.nan)
        p = float(sig) / 100.0 if np.isfinite(sig) else np.nan

    return stat, p


def kde_mode(x: np.ndarray, grid: np.ndarray) -> float:
    """
    Mode estimate from Gaussian KDE evaluated on 'grid'.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan

    kde = stats.gaussian_kde(x)
    # Apply bw_adjust by scaling covariance factor:
    # gaussian_kde doesn't have bw_adjust; we can emulate by multiplying factor.
    # factor = kde.factor; new_factor = factor * KDE_BW_ADJUST
    kde.set_bandwidth(bw_method=kde.factor * KDE_BW_ADJUST)

    dens = kde(grid)
    return float(grid[int(np.argmax(dens))])


def bootstrap_mode_diff(x: np.ndarray, y: np.ndarray, grid: np.ndarray, n: int, seed: int) -> Dict[str, float]:
    """
    Bootstrap the difference in modes: mode(x) - mode(y)
    Returns median + 95% CI + two-sided bootstrap p-value.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size < MIN_N_FOR_TESTS or y.size < MIN_N_FOR_TESTS:
        return dict(mode_x=np.nan, mode_y=np.nan, diff=np.nan, ci_low=np.nan, ci_high=np.nan, p_boot=np.nan)

    # point estimate
    mode_x = kde_mode(x, grid)
    mode_y = kde_mode(y, grid)
    diff0 = mode_x - mode_y

    diffs = np.empty(n, dtype=float)
    for i in range(n):
        xb = rng.choice(x, size=x.size, replace=True)
        yb = rng.choice(y, size=y.size, replace=True)
        diffs[i] = kde_mode(xb, grid) - kde_mode(yb, grid)

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value: how often does bootstrap distribution cross 0
    p_boot = 2.0 * min(np.mean(diffs <= 0.0), np.mean(diffs >= 0.0))
    p_boot = float(min(max(p_boot, 0.0), 1.0))

    return dict(mode_x=float(mode_x), mode_y=float(mode_y), diff=float(diff0),
                ci_low=float(ci_low), ci_high=float(ci_high), p_boot=float(p_boot))


def pairwise_tests(long_df: pd.DataFrame, metric: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    For each pair: AD k-sample + bootstrap mode diff.
    FDR correction applied separately for AD p-values and bootstrap p-values (within metric).
    """
    pairs = list(itertools.combinations(ORDER, 2))
    out: Dict[Tuple[str, str], Dict[str, float]] = {}

    # grid for mode estimation
    vals = long_df[metric].dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        grid = np.linspace(0, 1, MODE_GRID_N)
    else:
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        # padding makes KDE mode estimate less boundary-biased
        pad = 0.02 * (vmax - vmin) if vmax > vmin else 1.0
        grid = np.linspace(vmin - pad, vmax + pad, MODE_GRID_N)

    ad_pvals = []
    boot_pvals = []

    for a, b in pairs:
        xa = long_df.loc[long_df["set"] == a, metric].dropna().to_numpy(dtype=float)
        xb = long_df.loc[long_df["set"] == b, metric].dropna().to_numpy(dtype=float)

        xa = xa[np.isfinite(xa)]
        xb = xb[np.isfinite(xb)]

        if xa.size < MIN_N_FOR_TESTS or xb.size < MIN_N_FOR_TESTS:
            ad_stat, ad_p = np.nan, np.nan
            boot = dict(mode_x=np.nan, mode_y=np.nan, diff=np.nan, ci_low=np.nan, ci_high=np.nan, p_boot=np.nan)
        else:
            ad_stat, ad_p = ad_ksamp_pvalue(xa, xb)
            boot = bootstrap_mode_diff(xa, xb, grid=grid, n=BOOT_N, seed=BOOT_SEED)

        out[(a, b)] = {
            "n_A": float(xa.size),
            "n_B": float(xb.size),
            "ad_stat": float(ad_stat),
            "ad_p": float(ad_p),
            "mode_A": float(boot["mode_x"]),
            "mode_B": float(boot["mode_y"]),
            "mode_diff_A_minus_B": float(boot["diff"]),
            "mode_diff_ci_low": float(boot["ci_low"]),
            "mode_diff_ci_high": float(boot["ci_high"]),
            "mode_boot_p": float(boot["p_boot"]),
        }
        ad_pvals.append(ad_p)
        boot_pvals.append(boot["p_boot"])

    # FDR within metric across 6 comparisons (separately per test family)
    ad_pvals = np.array(ad_pvals, dtype=float)
    boot_pvals = np.array(boot_pvals, dtype=float)

    def fdr_map(pv: np.ndarray) -> np.ndarray:
        valid = np.isfinite(pv)
        q = np.full_like(pv, np.nan, dtype=float)
        if valid.sum():
            q[valid] = bh_fdr(pv[valid])
        return q

    ad_q = fdr_map(ad_pvals)
    boot_q = fdr_map(boot_pvals)

    for (a, b), q1, q2 in zip(pairs, ad_q, boot_q):
        out[(a, b)]["ad_q"] = float(q1) if np.isfinite(q1) else np.nan
        out[(a, b)]["mode_boot_q"] = float(q2) if np.isfinite(q2) else np.nan

    return out


# ---------- plotting helpers ----------
def pretty_metric(metric: str) -> str:
    return {
        "cds_len_bp_from_protein": "CDS length (bp)",
        "mrna_gc_percent": "GC content (%)",
    }.get(metric, metric)


def apply_log_x(ax, metric: str, values: np.ndarray):
    if metric not in LOG_SCALE_METRICS:
        return
    values = values[np.isfinite(values)]
    if values.size and np.all(values > 0):
        ax.set_xscale("log")


def add_sig_bars(ax, order: List[str], pair_stats: Dict[Tuple[str, str], Dict[str, float]], key: str):
    """
    Draw sig bars for a chosen q/p key (e.g. 'ad_q' or 'mode_boot_q').
    """
    pos = {name: i for i, name in enumerate(order)}
    y_min, y_max = ax.get_ylim()
    span = max(y_max - y_min, 1e-9)

    y = y_max + 0.06 * span
    h = 0.03 * span
    step = 0.10 * span

    pairs = list(pair_stats.keys())
    pairs.sort(key=lambda ab: abs(pos[ab[1]] - pos[ab[0]]))

    for (a, b) in pairs:
        val = pair_stats[(a, b)].get(key, np.nan)
        label = "n/a" if not np.isfinite(val) else stars(val)

        x1, x2 = pos[a], pos[b]
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.6, c="black")
        ax.text((x1 + x2) / 2, y + h + 0.01 * span, label, ha="center", va="bottom", fontsize=11)
        y += step

    ax.set_ylim(y_min, y + 0.06 * span)


def plot_kde_overlay(df: pd.DataFrame, metric: str):
    """
    Publication-style overlaid KDE (peaks become obvious).
    """
    plt.figure(figsize=(11.2, 6.4))
    ax = plt.gca()

    sns.kdeplot(
        data=df, x=metric, hue="set", hue_order=ORDER,
        common_norm=False, bw_adjust=KDE_BW_ADJUST, linewidth=2.4, ax=ax
    )

    apply_log_x(ax, metric, df[metric].to_numpy(dtype=float))

    title = f"KDE: {pretty_metric(metric)} (shared bw_adjust={KDE_BW_ADJUST})"
    ax.set_title(title, pad=12)
    ax.set_xlabel(pretty_metric(metric))
    ax.set_ylabel("Density")

    # helpful reference for the GC peak you mentioned
    if metric == "mrna_gc_percent":
        ax.axvline(44.0, ls="--", lw=1.6, color="black", alpha=0.55)
        ax.text(44.0, ax.get_ylim()[1]*0.95, "44%", ha="center", va="top", fontsize=11)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"kde_overlay__{metric}.png", dpi=DPI)
    plt.close()


def plot_ridgeline(df: pd.DataFrame, metric: str):
    """
    Ridgeline (joy-style) KDE using FacetGrid.
    """
    # Order for ridgeline: top at top, etc.
    ridge_order = ORDER

    g = sns.FacetGrid(
        df, row="set", row_order=ridge_order,
        height=1.4, aspect=5.0, sharex=True, sharey=False
    )

    # Filled KDE per row
    def _kde(x, color=None, **kwargs):
        sns.kdeplot(
            x=x, bw_adjust=KDE_BW_ADJUST, linewidth=1.8,
            fill=True, alpha=0.75, clip_on=False
        )

    g.map(_kde, metric)

    for ax, label in zip(g.axes.flat, ridge_order):
        ax.set_ylabel("")
        ax.text(0.01, 0.30, label, transform=ax.transAxes, ha="left", va="center", fontsize=12)
        ax.axhline(0, lw=1.0, color="black", alpha=0.35)

    # formatting
    g.fig.subplots_adjust(hspace=-0.55)
    g.set_titles("")
    g.set_xlabels(pretty_metric(metric))
    g.set_ylabels("")
    g.fig.suptitle(f"Ridgeline KDE: {pretty_metric(metric)}", y=1.02)

    # reference line for GC
    if metric == "mrna_gc_percent":
        for ax in g.axes.flat:
            ax.axvline(44.0, ls="--", lw=1.2, color="black", alpha=0.5)

    # x log for length, if appropriate
    if metric in LOG_SCALE_METRICS:
        vals = df[metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size and np.all(vals > 0):
            for ax in g.axes.flat:
                ax.set_xscale("log")

    plt.tight_layout()
    g.savefig(OUT_DIR / f"ridgeline__{metric}.png", dpi=DPI, bbox_inches="tight")
    plt.close(g.fig)


def plot_violin_with_sig(df: pd.DataFrame, metric: str, pair_stats: Dict[Tuple[str, str], Dict[str, float]]):
    """
    Violin + box + points with sig bars.
    We annotate the *mode bootstrap q* by default (peak-oriented),
    and we also write AD q into a caption line.
    """
    d = df[["set", metric]].dropna().copy()
    if d.empty:
        return

    d["set"] = pd.Categorical(d["set"], categories=ORDER, ordered=True)

    n_by_set = d.groupby("set")[metric].count().to_dict()
    tick_labels = [f"{s}\n(N={int(n_by_set.get(s, 0))})" for s in ORDER]

    plt.figure(figsize=(11.2, 6.8))
    ax = plt.gca()

    sns.violinplot(data=d, x="set", y=metric, order=ORDER, inner=None, cut=0, linewidth=1.2, ax=ax)
    sns.boxplot(data=d, x="set", y=metric, order=ORDER, width=0.25, showfliers=False, color="white", linewidth=1.2, ax=ax)
    sns.stripplot(data=d, x="set", y=metric, order=ORDER, jitter=0.22, size=3.6, alpha=0.45, linewidth=0, ax=ax)

    if metric in LOG_SCALE_METRICS:
        vals = d[metric].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size and np.all(vals > 0):
            ax.set_yscale("log")

    ax.set_title(f"{pretty_metric(metric)} across gene sets", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(pretty_metric(metric))
    ax.set_xticklabels(tick_labels)

    # annotate peak-focused significance
    key = "mode_boot_q" if USE_FDR else "mode_boot_p"
    add_sig_bars(ax, ORDER, pair_stats, key=key)

    # caption about what the bars mean + AD
    note1 = "Bars: mode-difference test (bootstrap on KDE mode). Stars show q (BH-FDR) within metric."
    note2 = "AD k-sample (shape) results are saved to the table (pairwise_ad_mode_tests.tsv)."
    ax.text(0.01, -0.18, note1 + "\n" + note2, transform=ax.transAxes, ha="left", va="top", fontsize=10)

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"violin_sig__{metric}.png", dpi=DPI)
    plt.close()


def main():
    top = read_tsv(TOP_TSV).assign(set="top")
    low = read_tsv(LOWEST_TSV).assign(set="lowest")
    rnd = read_tsv(RANDOM_TOP_TSV).assign(set="random_top10k")
    rnd2 = read_tsv(RANDOM_LOW_TSV).assign(set="random_below10k")

    df = pd.concat([top, low, rnd, rnd2], ignore_index=True)
    df = df[df["set"].isin(ORDER)].copy()
    df["set"] = pd.Categorical(df["set"], categories=ORDER, ordered=True)

    # Stats table rows
    rows = []

    for metric in METRICS:
        if metric not in df.columns:
            print(f"Warning: metric '{metric}' not found in input columns.")
            continue

        # Pairwise stats
        ps = pairwise_tests(df, metric)

        # Plots
        plot_kde_overlay(df[["set", metric]].dropna(), metric)
        plot_ridgeline(df[["set", metric]].dropna(), metric)
        plot_violin_with_sig(df, metric, ps)

        # Save stats rows
        for (a, b), dct in ps.items():
            rows.append({
                "metric": metric,
                "metric_pretty": pretty_metric(metric),
                "A": a,
                "B": b,
                "n_A": int(dct["n_A"]),
                "n_B": int(dct["n_B"]),
                "ad_stat": dct["ad_stat"],
                "ad_p": dct["ad_p"],
                "ad_q": dct["ad_q"],
                "mode_A": dct["mode_A"],
                "mode_B": dct["mode_B"],
                "mode_diff_A_minus_B": dct["mode_diff_A_minus_B"],
                "mode_diff_ci_low": dct["mode_diff_ci_low"],
                "mode_diff_ci_high": dct["mode_diff_ci_high"],
                "mode_boot_p": dct["mode_boot_p"],
                "mode_boot_q": dct["mode_boot_q"],
                "ad_stars": ("n/a" if not np.isfinite(dct["ad_q" if USE_FDR else "ad_p"]) else stars(dct["ad_q" if USE_FDR else "ad_p"])),
                "mode_stars": ("n/a" if not np.isfinite(dct["mode_boot_q" if USE_FDR else "mode_boot_p"]) else stars(dct["mode_boot_q" if USE_FDR else "mode_boot_p"])),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "pairwise_ad_mode_tests.tsv", sep="\t", index=False)

    print(f"Wrote plots + tables to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()