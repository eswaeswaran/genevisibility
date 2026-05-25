import pandas as pd

df = pd.read_csv("DiseaseAttentionResults/disease_attention__knowledge_filtered.csv")

# Apply stability filter
df = df[df["n_genes_with_pubmed"] >= 10]

# A) Most understudied by median attention
under_by_median = df.sort_values("median_pubmed", ascending=True).head(20)

# B) Most understudied by lack of top-500 genes
under_by_frac = df.sort_values("frac_in_top500", ascending=True).head(20)

print("Most understudied diseases by median PubMed count:")
print(under_by_median[["Disease", "n_genes_with_pubmed", "median_pubmed", "frac_in_top500"]])

print("\nMost understudied diseases by fraction in top-500:")
print(under_by_frac[["Disease", "n_genes_with_pubmed", "median_pubmed", "frac_in_top500"]])