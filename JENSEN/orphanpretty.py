#!/usr/bin/env python3
"""
Create a publication-ready Seaborn figure comparing orphan vs non-orphan diseases
by median PubMed publication count per disease.

Input:
  DiseaseAttentionResults/disease_attention__knowledge_filtered_with_orphan.csv

Output:
  manuscript/Figures/orphan_vs_nonorphan_median_pubmed.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


# ---------------------- PATHS (EDIT IF NEEDED) ----------------------
INPUT_CSV = Path(
    "DiseaseAttentionResults/disease_attention__knowledge_filtered_with_orphan.csv"
)

MANUSCRIPT_FIG_DIR = Path(
    "/Users/mtongoss/Dropbox/Research/projects/MostStudiedGenes/FilesFromStudents/manuscript/Figures"
)

OUTPUT_FIG = MANUSCRIPT_FIG_DIR / "orphan_vs_nonorphan_median_pubmed.png"
# -------------------------------------------------------------------


def main() -> None:
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing input CSV: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)

    # Stability filter
    df = df[df["n_genes_with_pubmed"] >= 10].copy()

    # Sanity checks
    required = {"is_orphan", "median_pubmed"}
    if not required.issubset(df.columns):
        raise SystemExit(
            f"Input CSV missing required columns. Found: {list(df.columns)}"
        )

    # Label for plotting
    df["DiseaseClass"] = np.where(
        df["is_orphan"], "Orphan diseases", "Non-orphan diseases"
    )

    # Statistics
    orphan_vals = df.loc[df["is_orphan"], "median_pubmed"].to_numpy()
    nonorphan_vals = df.loc[~df["is_orphan"], "median_pubmed"].to_numpy()

    u_stat, p_val = mannwhitneyu(
        orphan_vals, nonorphan_vals, alternative="two-sided"
    )

    med_orphan = float(np.median(orphan_vals))
    med_nonorphan = float(np.median(nonorphan_vals))

    n_orphan = int(df["is_orphan"].sum())
    n_nonorphan = int((~df["is_orphan"]).sum())

    # ---------------------- PLOTTING ----------------------
    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.9,
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
        },
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    order = ["Non-orphan diseases", "Orphan diseases"]

    # Violin: distribution shape
    sns.violinplot(
        data=df,
        x="DiseaseClass",
        y="median_pubmed",
        order=order,
        inner=None,
        cut=0,
        linewidth=0.9,
        ax=ax,
    )

    # Boxplot: median + IQR
    sns.boxplot(
        data=df,
        x="DiseaseClass",
        y="median_pubmed",
        order=order,
        width=0.25,
        showcaps=True,
        showfliers=False,
        whiskerprops={"linewidth": 1},
        boxprops={"alpha": 0.75},
        medianprops={"linewidth": 2},
        ax=ax,
    )

    # Jittered points: data density
    sns.stripplot(
        data=df,
        x="DiseaseClass",
        y="median_pubmed",
        order=order,
        size=3,
        jitter=0.25,
        alpha=0.45,
        linewidth=0,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Median PubMed count per disease")

    # Adjust y-limits to make space for annotations
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.08)

    # Group annotations
    ax.text(
        0,
        ymax * 1.03,
        f"n = {n_nonorphan}\nmedian = {med_nonorphan:.1f}",
        ha="center",
        va="top",
        fontsize=12,
    )

    ax.text(
        1,
        ymax * 1.03,
        f"n = {n_orphan}\nmedian = {med_orphan:.1f}",
        ha="center",
        va="top",
        fontsize=12,
    )

    # p-value bracket
    ax.plot(
        [0, 0, 1, 1],
        [ymax * 0.96, ymax * 0.99, ymax * 0.99, ymax * 0.96],
        lw=1,
        color="black",
    )

    ax.text(
        0.5,
        ymax * 1.005,
        f"Mann–Whitney U p = {p_val:.2e}",
        ha="center",
        va="bottom",
        fontsize=12,
    )

    # Explicit margins (prevents truncation and title overlap)
    plt.subplots_adjust(
        left=0.14,
        right=0.98,
        bottom=0.12,
        top=0.88,
    )

    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG, dpi=300)
    plt.close()

    # Console summary
    print("=== Orphan vs non-orphan disease analysis ===")
    print(f"Non-orphan diseases: n = {n_nonorphan}, median = {med_nonorphan:.1f}")
    print(f"Orphan diseases    : n = {n_orphan}, median = {med_orphan:.1f}")
    print(f"Mann–Whitney U p-value = {p_val:.2e}")
    print(f"Wrote figure: {OUTPUT_FIG.resolve()}")


if __name__ == "__main__":
    main()