#!/usr/bin/env python3
import pandas as pd
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt

# Load annotated table
df = pd.read_csv(
    "DiseaseAttentionResults/disease_attention__knowledge_filtered_with_orphan.csv"
)

# Stability filter
df = df[df["n_genes_with_pubmed"] >= 10].copy()

orph = df[df["is_orphan"]]
nonorph = df[~df["is_orphan"]]

# --- Stats ---
u, p = mannwhitneyu(
    orph["median_pubmed"],
    nonorph["median_pubmed"],
    alternative="two-sided"
)

print("Median PubMed count per disease:")
print(f"  Orphan     : {orph['median_pubmed'].median():.1f}")
print(f"  Non-orphan : {nonorph['median_pubmed'].median():.1f}")
print(f"Mann–Whitney U p-value: {p:.2e}")

# --- Plot ---
plt.figure(figsize=(6, 5))
sns.boxplot(
    data=df,
    x="is_orphan",
    y="median_pubmed",
    width=0.6
)
plt.xticks([0, 1], ["Non-orphan", "Orphan"])
plt.ylabel("Median PubMed count per disease")
plt.xlabel("")
plt.tight_layout()
plt.savefig(
    "Figures/orphan_vs_nonorphan_median_pubmed.png",
    dpi=300
)
plt.close()