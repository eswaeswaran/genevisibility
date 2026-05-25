#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# ==========================
# CONFIG
# ==========================

BASE = Path("/Users/mtongoss/Dropbox/Research/projects/MostStudiedGenes/FilesFromStudents/4string/codeml_outputs_v5/pairwise")

FILES = {
    "lowest500": BASE / "lowest500/dnds.lowest500.tsv",
    "randomAbove10k": BASE / "randomAbove10k/dnds.randomAbove10k.tsv",
    "randomBelow10k": BASE / "randomBelow10k/dnds.randomBelow10k.tsv",
    "top500": BASE / "top500/dnds.top500.tsv",
}

OUTPUT_DIR = BASE / "group_comparisons_log2"
OUTPUT_DIR.mkdir(exist_ok=True)

METRICS = ["dN", "dS", "dN_dS"]

# Small value to avoid log2(0); adjust if you want (e.g. 1e-8)
PSEUDOCOUNT = 1e-8

# ==========================
# LOAD + CLEAN
# ==========================

data = []
for group, file in FILES.items():
    df = pd.read_csv(file, sep="\t")

    df = df.replace("NA", np.nan)
    df = df.dropna(subset=METRICS)

    for m in METRICS:
        df[m] = df[m].astype(float)

    df["group"] = group
    data.append(df)

df_all = pd.concat(data, ignore_index=True)

# ==========================
# PLOT
# ==========================

def log2_transform(series: pd.Series) -> pd.Series:
    return np.log2(series + PSEUDOCOUNT)

def plot_metric(metric: str):
    groups = list(df_all["group"].unique())

    vals_raw = [df_all.loc[df_all["group"] == g, metric].values for g in groups]
    vals_log = [log2_transform(pd.Series(v)).values for v in vals_raw]

    Ns = [len(v) for v in vals_log]
    xlabels = [f"{g}\nN={n}" for g, n in zip(groups, Ns)]

    means = [np.mean(v) for v in vals_log]
    medians = [np.median(v) for v in vals_log]

    plt.figure(figsize=(9, 6))

    # Violin plot
    vparts = plt.violinplot(vals_log, showmeans=False, showmedians=False, showextrema=False)

    # Boxplot overlay (median is shown as a line automatically)
    bp = plt.boxplot(
        vals_log,
        widths=0.18,
        showfliers=False,   # cleaner; change to True if you want outliers
        patch_artist=False
    )

    # Mean markers
    xs = np.arange(1, len(groups) + 1)
    plt.scatter(xs, means, marker="D", s=35, label="Mean")

    # Optional: also mark medians explicitly (boxplot already shows it, but this makes it obvious)
    plt.scatter(xs, medians, marker="_", s=450, label="Median")

    plt.xticks(xs, xlabels, rotation=0)
    plt.ylabel(f"log2({metric} + {PSEUDOCOUNT:g})")
    plt.title(f"{metric} across groups (log2 scale)")

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{metric}_log2_violin_boxplot.png", dpi=300)
    plt.close()

for m in METRICS:
    plot_metric(m)

# ==========================
# STATISTICS (still on raw scale by default)
# If you prefer stats on log2 scale, swap values_raw -> values_log below.
# ==========================

stats_output = []
groups = list(df_all["group"].unique())

for metric in METRICS:
    vals_raw = [df_all.loc[df_all["group"] == g, metric].values for g in groups]

    # ANOVA + Kruskal on RAW values (common choice for interpretability; dN/dS is often non-normal anyway)
    F, p_anova = stats.f_oneway(*vals_raw)
    H, p_kruskal = stats.kruskal(*vals_raw)

    stats_output.append(
        f"\nMetric: {metric}\n"
        f"ANOVA (raw): F={F:.4f}, p={p_anova:.4e}\n"
        f"Kruskal-Wallis (raw): H={H:.4f}, p={p_kruskal:.4e}\n"
    )

    # Pairwise Mann–Whitney on RAW values + FDR
    pairwise_p = []
    pairwise_labels = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            U, p = stats.mannwhitneyu(vals_raw[i], vals_raw[j], alternative="two-sided")
            pairwise_p.append(p)
            pairwise_labels.append(f"{groups[i]} vs {groups[j]}")

    reject, p_adj, _, _ = multipletests(pairwise_p, method="fdr_bh")

    stats_output.append("Pairwise Mann–Whitney (raw; FDR corrected):\n")
    for lbl, p_raw, p_corr, r in zip(pairwise_labels, pairwise_p, p_adj, reject):
        stats_output.append(
            f"{lbl}: raw_p={p_raw:.4e}, FDR_p={p_corr:.4e}, significant={r}\n"
        )

with open(OUTPUT_DIR / "statistical_summary.txt", "w") as f:
    f.writelines(stats_output)

print("Wrote plots + stats to:", OUTPUT_DIR)