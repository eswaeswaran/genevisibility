#!/usr/bin/env python3
from __future__ import annotations

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ============================================================
# EXPRESSION ANALYSIS (pub-ready)
# ============================================================

# ==============================
# CONFIG
# ==============================
EXPR_FILE = "rna_tissue_consensus.tsv"

GROUP_FILES = {
    "top500": "top500.txt",
    "lowest500": "lowest500.txt",
    "randomAbove10k": "randomAbove10k.txt",
    "randomBelow10k": "randomBelow10k.txt",
}

OUTDIR = "expression_pubready_out"

ID_COL = "Gene name"
VAL_COL = "nTPM"

METRICS = ["mean_expr", "max_expr", "tau"]
GROUP_ORDER = list(GROUP_FILES.keys())

# GLOBAL FILTER
MIN_MAX_NTPM = 1.0

LOG_HEADROOM_FACTOR = 100.0
LINEAR_HEADROOM_FACTOR = 1.5

STAR_LEVELS = [(1e-3, "***"), (1e-2, "**"), (5e-2, "*")]
ANNOTATE_Q = 0.05


# ==============================
# HELPERS
# ==============================
def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def read_list(path):
    genes = set()
    with open(path) as f:
        for line in f:
            g = line.strip()
            if g:
                genes.add(g.split()[0])
    return genes

def compute_tau(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    m = np.nanmax(x)
    if not np.isfinite(m) or m <= 0:
        return np.nan
    return float(np.nansum(1 - x/m) / (len(x) - 1))

def stars_from_q(q):
    if not np.isfinite(q):
        return "n.s."
    for thr, sym in STAR_LEVELS:
        if q <= thr:
            return sym
    return "n.s."

def format_q(q):
    if not np.isfinite(q):
        return "NA"
    if q < 1e-3:
        return f"{q:.1e}"
    return f"{q:.3f}"

def metric_title(metric):
    base = {
        "mean_expr": "Mean expression",
        "max_expr": "Max expression",
        "tau": "Tissue specificity (τ)",
    }.get(metric, metric)

    # Add filter only to expression plots
    if metric in ("mean_expr", "max_expr"):
        return f"{base} (max nTPM ≥ {MIN_MAX_NTPM:g})"
    return base

def metric_ylabel(metric):
    return {
        "mean_expr": "Mean nTPM",
        "max_expr": "Max nTPM",
        "tau": "τ (0 = broad, 1 = specific)",
    }.get(metric, metric)

def _bboxes_overlap(a, b, pad=2.0):
    return not (
        (a.x1 + pad) < b.x0 or
        (a.x0 - pad) > b.x1 or
        (a.y1 + pad) < b.y0 or
        (a.y0 - pad) > b.y1
    )


# ==============================
# STATS
# ==============================
def kruskal_omnibus(df, metric):
    arrays = [df[df.group == g][metric].dropna() for g in GROUP_ORDER]
    if any(len(a) == 0 for a in arrays):
        return {"metric": metric, "p": np.nan}
    _, p = kruskal(*arrays)
    return {"metric": metric, "p": p}

def pairwise_mwu(df, metric):
    rows = []
    for a, b in itertools.combinations(GROUP_ORDER, 2):
        x = df[df.group == a][metric].dropna()
        y = df[df.group == b][metric].dropna()
        if len(x) == 0 or len(y) == 0:
            rows.append({"metric": metric, "group1": a, "group2": b, "p": np.nan})
            continue
        _, p = mannwhitneyu(x, y, alternative="two-sided")
        rows.append({"metric": metric, "group1": a, "group2": b, "p": p})

    res = pd.DataFrame(rows)
    mask = np.isfinite(res["p"])
    q = np.full(len(res), np.nan)
    if mask.sum() > 0:
        _, qvals, _, _ = multipletests(res.loc[mask, "p"], method="fdr_bh")
        q[mask] = qvals

    res["q"] = q
    res["sig"] = [stars_from_q(v) for v in res["q"]]
    return res


# ==============================
# PLOTTING
# ==============================
def plot_metric(df, metric, pairwise, outdir):

    arrays = [df[df.group == g][metric].dropna().values for g in GROUP_ORDER]
    allvals = np.concatenate(arrays) if len(arrays) else np.array([np.nan])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.boxplot(arrays, showfliers=False)

    rng = np.random.default_rng(1)
    for i, a in enumerate(arrays, start=1):
        xj = rng.normal(i, 0.06, len(a))
        ax.scatter(xj, a, s=8, alpha=0.3)

    ax.set_xticks(range(1, len(GROUP_ORDER) + 1))
    ax.set_xticklabels(GROUP_ORDER, rotation=25, ha="right")

    if metric in ("mean_expr", "max_expr"):
        ax.set_yscale("log")

    ymin = np.nanmin(allvals)
    ymax = np.nanmax(allvals)

    if ax.get_yscale() == "log":
        bottom = max(ymin * 0.8, 1e-6)
        top = ymax * LOG_HEADROOM_FACTOR
        ax.set_ylim(bottom=bottom, top=top)
        y0 = ymax * 1.3
        bump_mult = 1.18
        line_mult = 1.06
    else:
        yr = ymax - ymin if np.isfinite(ymax - ymin) else 1.0
        bottom = ymin - 0.1 * yr
        top = ymax + LINEAR_HEADROOM_FACTOR * yr
        ax.set_ylim(bottom=bottom, top=top)
        y0 = ymax + 0.2 * yr
        bump_add = 0.12 * yr
        line_add = 0.35 * bump_add

    ax.set_title(metric_title(metric))
    ax.set_ylabel(metric_ylabel(metric))

    sig = pairwise[(pairwise.metric == metric) & (pairwise.q <= ANNOTATE_Q)].copy()
    sig = sig.sort_values("q")

    placed = []
    y = y0

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for _, r in sig.iterrows():
        i1 = GROUP_ORDER.index(r.group1) + 1
        i2 = GROUP_ORDER.index(r.group2) + 1
        label = f"{r.sig} (q={format_q(r.q)})"

        attempts = 0
        while attempts < 200:
            if ax.get_yscale() == "log":
                ytop = y * line_mult
                (line,) = ax.plot([i1, i1, i2, i2], [y, ytop, ytop, y], lw=1)
                ty = ytop * 1.02
            else:
                ytop = y + line_add
                (line,) = ax.plot([i1, i1, i2, i2], [y, ytop, ytop, y], lw=1)
                ty = ytop

            txt = ax.text((i1+i2)/2, ty, label,
                          ha="center", va="bottom", fontsize=9)

            fig.canvas.draw()
            bb = txt.get_window_extent(renderer=renderer)

            if any(_bboxes_overlap(bb, p) for p in placed):
                txt.remove()
                line.remove()
                y = y * bump_mult if ax.get_yscale()=="log" else y + bump_add
                attempts += 1
            else:
                placed.append(bb)
                break

        y = y * 1.10 if ax.get_yscale()=="log" else y + bump_add

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"{metric}.pdf"))
    fig.savefig(os.path.join(outdir, f"{metric}.png"), dpi=300)
    plt.close(fig)


# ==============================
# MAIN
# ==============================
def main():

    ensure_outdir(OUTDIR)
    groups = {k: read_list(v) for k, v in GROUP_FILES.items()}

    df = pd.read_csv(EXPR_FILE, sep="\t")
    df[ID_COL] = df[ID_COL].astype(str)
    df[VAL_COL] = pd.to_numeric(df[VAL_COL], errors="coerce")

    rows = []
    for gene, sub in df.groupby(ID_COL):
        x = sub[VAL_COL].values
        rows.append({
            "HGNC": gene,
            "mean_expr": np.mean(x),
            "max_expr": np.max(x),
            "tau": compute_tau(x)
        })

    gene_stats = pd.DataFrame(rows)

    def assign(g):
        for name, s in groups.items():
            if g in s:
                return name
        return None

    gene_stats["group"] = gene_stats["HGNC"].apply(assign)
    data = gene_stats.dropna(subset=["group"]).copy()

    # APPLY FILTER
    data = data[data.max_expr >= MIN_MAX_NTPM]

    omni = pd.DataFrame([kruskal_omnibus(data, m) for m in METRICS])
    omni.to_csv(os.path.join(OUTDIR, "omnibus.tsv"), sep="\t", index=False)

    pairwise_all = pd.concat([pairwise_mwu(data, m) for m in METRICS])
    pairwise_all.to_csv(os.path.join(OUTDIR, "pairwise.tsv"), sep="\t", index=False)

    for m in METRICS:
        plot_metric(data, m, pairwise_all, OUTDIR)

    print("Finished. Filter displayed in plot titles.")


if __name__ == "__main__":
    main()