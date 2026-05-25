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

OUTPUT_DIR = BASE / "group_comparisons"
OUTPUT_DIR.mkdir(exist_ok=True)

METRICS = ["dN", "dS", "dN_dS"]

# ==========================
# LOAD + CLEAN
# ==========================

data = []

for group, file in FILES.items():
    df = pd.read_csv(file, sep="\t")
    
    # Remove NA rows
    df = df.replace("NA", np.nan)
    df = df.dropna(subset=METRICS)
    
    # Convert to float
    for m in METRICS:
        df[m] = df[m].astype(float)
    
    df["group"] = group
    
    print(f"{group}: N genes after filtering = {len(df)}")
    
    data.append(df)

df_all = pd.concat(data, ignore_index=True)

# ==========================
# PLOTTING FUNCTION
# ==========================

def plot_metric(metric):
    
    groups = df_all["group"].unique()
    values = [df_all[df_all["group"] == g][metric] for g in groups]
    
    plt.figure(figsize=(8,6))
    
    # Violin
    plt.violinplot(values, showmeans=False, showmedians=False)
    
    # Boxplot overlay
    plt.boxplot(values, widths=0.1)
    
    plt.xticks(range(1, len(groups)+1), groups, rotation=45)
    plt.ylabel(metric)
    plt.title(f"{metric} comparison across gene groups")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{metric}_violin_boxplot.png", dpi=300)
    plt.close()

for m in METRICS:
    plot_metric(m)

# ==========================
# STATISTICS
# ==========================

stats_output = []

for metric in METRICS:
    print(f"\n=== {metric} ===")
    
    groups = df_all["group"].unique()
    values = [df_all[df_all["group"] == g][metric] for g in groups]
    
    # ANOVA
    F, p_anova = stats.f_oneway(*values)
    
    # Kruskal
    H, p_kruskal = stats.kruskal(*values)
    
    stats_output.append(
        f"\nMetric: {metric}\n"
        f"ANOVA: F={F:.4f}, p={p_anova:.4e}\n"
        f"Kruskal-Wallis: H={H:.4f}, p={p_kruskal:.4e}\n"
    )
    
    # Pairwise Mann–Whitney
    pairwise_p = []
    pairwise_labels = []
    
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            U, p = stats.mannwhitneyu(values[i], values[j], alternative="two-sided")
            pairwise_p.append(p)
            pairwise_labels.append(f"{groups[i]} vs {groups[j]}")
    
    # FDR correction
    reject, p_adj, _, _ = multipletests(pairwise_p, method="fdr_bh")
    
    stats_output.append("Pairwise Mann–Whitney (FDR corrected):\n")
    for lbl, p_raw, p_corr, r in zip(pairwise_labels, pairwise_p, p_adj, reject):
        stats_output.append(
            f"{lbl}: raw_p={p_raw:.4e}, FDR_p={p_corr:.4e}, significant={r}\n"
        )

# Write stats
with open(OUTPUT_DIR / "statistical_summary.txt", "w") as f:
    f.writelines(stats_output)

print("\nAll results written to:", OUTPUT_DIR)