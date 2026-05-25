#!/usr/bin/env python3
"""
Disease-centric research attention analysis (DISEASES/JensenLab + PubMed counts)

Put this script in the SAME folder as:
  - all_genes.txt
  - human_disease_knowledge_filtered.tsv
  - human_disease_knowledge_full.tsv

Your DISEASES TSVs appear to be HEADERLESS with rows like:
  ENSP00000001146   CYP26B1   DOID:2340   Craniosynostosis   UniProtKB-KW   CURATED   4

This script:
  1) Loads PubMed counts and deduplicates GeneSymbol (keeps MAX PubMed_Count per symbol).
  2) Loads DISEASES disease–gene edges (headerless or headered).
  3) Joins disease genes to PubMed counts and computes disease-level attention metrics.
  4) Writes summary tables and creates a few publication-ready plots.

Outputs:
  ./DiseaseAttentionResults/
    disease_attention__knowledge_filtered.csv
    disease_attention__knowledge_full.csv
    disease_attention__compare_full_vs_filtered.csv
    Figures/
      top_diseases_by_median_pubmed__knowledge_filtered.png
      top_diseases_by_mean_pubmed__knowledge_filtered.png
      top_diseases_by_frac_in_top500__knowledge_filtered.png
      scatter_n_genes_vs_median_pubmed__knowledge_filtered.png
      compare_filtered_vs_full_median_scatter.png

Run:
  python3 analysis.py
Optional:
  python3 analysis.py --score-min 4
  python3 analysis.py --min-genes 10
  python3 analysis.py --top-n 30
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------- Helpers ----------------------------- #

def normalize_symbol(sym: object) -> str:
    """Light HGNC-like normalization: strip, remove internal whitespace, uppercase."""
    if sym is None:
        return ""
    if isinstance(sym, float) and np.isnan(sym):
        return ""
    s = str(sym).strip()
    s = re.sub(r"\s+", "", s)
    return s.upper()

def gini(x: np.ndarray) -> float:
    """Gini coefficient for non-negative array."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.any(x < 0):
        x = x - x.min()
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def looks_like_enspid(s: object) -> bool:
    s = str(s)
    return s.startswith("ENSP") or s.startswith("ENSG") or s.startswith("ENST")

def read_tsv_maybe_header(path: Path) -> pd.DataFrame:
    """
    DISEASES knowledge TSVs often come WITHOUT a header.
    Detect by checking whether the first field looks like an ENSP/ENSG/ENST id.
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n")
    fields = first.split("\t")
    no_header = (len(fields) >= 4) and looks_like_enspid(fields[0])
    if no_header:
        return pd.read_csv(path, sep="\t", header=None, low_memory=False)
    return pd.read_csv(path, sep="\t", header=0, low_memory=False)


# ----------------------------- Loaders ----------------------------- #

def load_pubmed_counts(pubmed_path: Path) -> pd.DataFrame:
    """
    Load all_genes.txt (tab-separated) with columns:
      GeneID, GeneSymbol, PubMed_Count
    Deduplicate GeneSymbol by taking max PubMed_Count (prevents merge errors).
    """
    df = pd.read_csv(pubmed_path, sep="\t", low_memory=False)
    if not {"GeneSymbol", "PubMed_Count"}.issubset(df.columns):
        raise ValueError(
            f"Expected columns GeneSymbol and PubMed_Count in {pubmed_path}. Found: {list(df.columns)}"
        )

    df = df[["GeneSymbol", "PubMed_Count"]].copy()
    df["GeneSymbol"] = df["GeneSymbol"].map(normalize_symbol)

    df["PubMed_Count"] = pd.to_numeric(df["PubMed_Count"], errors="coerce")
    df = df.dropna(subset=["PubMed_Count"])
    df["PubMed_Count"] = df["PubMed_Count"].astype(int)

    # Report + collapse duplicates safely
    dup_mask = df["GeneSymbol"].duplicated(keep=False)
    if dup_mask.any():
        dups = df.loc[dup_mask].sort_values(["GeneSymbol", "PubMed_Count"], ascending=[True, False])
        print(
            f"[INFO] Found {dups['GeneSymbol'].nunique()} duplicated GeneSymbols in PubMed table "
            f"({len(dups)} rows). Keeping max PubMed_Count per symbol."
        )
        print(dups.head(30).to_string(index=False))

    df = df.groupby("GeneSymbol", as_index=False)["PubMed_Count"].max()

    # Rank: 1 = most studied
    df = df.sort_values("PubMed_Count", ascending=False).reset_index(drop=True)
    df["PubMed_Rank"] = np.arange(1, len(df) + 1)

    return df

def load_diseases_edges(path: Path) -> pd.DataFrame:
    """
    Load DISEASES knowledge TSV and return standardized edges:
      GeneSymbol, Disease, Score(optional)

    Headerless common format (7 columns):
      0: Ensembl protein (ENSP...)
      1: Gene symbol
      2: Disease ID (DOID:...)
      3: Disease name
      4: source/vocab (e.g. UniProtKB-KW)
      5: evidence type (e.g. CURATED)
      6: score (integer)
    """
    raw = read_tsv_maybe_header(path)

    # Headerless: integer columns
    if list(raw.columns) == list(range(raw.shape[1])):
        if raw.shape[1] < 4:
            raise ValueError(f"Unexpected DISEASES format in {path}: only {raw.shape[1]} columns.")

        gene_sym_col = 1
        disease_name_col = 3
        score_col = 6 if raw.shape[1] >= 7 else None

        keep_cols = [gene_sym_col, disease_name_col] + ([score_col] if score_col is not None else [])
        edges = raw[keep_cols].copy()
        edges = edges.rename(columns={gene_sym_col: "GeneSymbol", disease_name_col: "Disease"})
        if score_col is not None:
            edges = edges.rename(columns={score_col: "Score"})
    else:
        # Headered variant: try common names
        cols_lower = {c.lower(): c for c in raw.columns}

        def pick(needles):
            for needle in needles:
                for k, orig in cols_lower.items():
                    if k == needle or needle in k:
                        return orig
            return None

        sym_col = pick(["genesymbol", "gene_symbol", "symbol", "hgnc_symbol", "gene"])
        dis_col = pick(["diseasename", "disease_name", "disease"])
        sc_col = pick(["score", "confidence"])

        if sym_col is None or dis_col is None:
            raise ValueError(f"Could not detect gene/disease columns in {path}. Columns: {list(raw.columns)}")

        keep_cols = [sym_col, dis_col] + ([sc_col] if sc_col is not None else [])
        edges = raw[keep_cols].copy()
        edges = edges.rename(columns={sym_col: "GeneSymbol", dis_col: "Disease"})
        if sc_col is not None:
            edges = edges.rename(columns={sc_col: "Score"})

    # Normalize
    edges["GeneSymbol"] = edges["GeneSymbol"].map(normalize_symbol)
    edges["Disease"] = edges["Disease"].astype(str).str.strip()
    edges = edges[(edges["GeneSymbol"] != "") & (edges["Disease"] != "")].copy()

    if "Score" in edges.columns:
        edges["Score"] = pd.to_numeric(edges["Score"], errors="coerce")

    return edges


# ----------------------------- Analysis ----------------------------- #

def plot_over_vs_understudied(
    summary: pd.DataFrame,
    outpath: Path,
    metric: str = "median_pubmed",
    top_n: int = 15,
    min_genes: int = 10,
):
    """
    Create a two-panel figure:
      Left: most overstudied diseases
      Right: most understudied diseases
    """
    df = summary.copy()
    df = df[df["n_genes_with_pubmed"] >= min_genes].dropna(subset=[metric])

    over = df.sort_values(metric, ascending=False).head(top_n)
    under = df.sort_values(metric, ascending=True).head(top_n)

    fig, axes = plt.subplots(
        ncols=2, figsize=(14, max(6, 0.4 * top_n)), sharex=False
    )

    # Overstudied
    sns.barplot(
        data=over.sort_values(metric),
        x=metric, y="Disease",
        ax=axes[0], color="#d95f02"
    )
    axes[0].set_title("Overstudied diseases\n(high median PubMed count)")
    axes[0].set_xlabel("Median PubMed count")
    axes[0].set_ylabel("")

    # Understudied
    sns.barplot(
        data=under.sort_values(metric, ascending=False),
        x=metric, y="Disease",
        ax=axes[1], color="#1b9e77"
    )
    axes[1].set_title("Understudied diseases\n(low median PubMed count)")
    axes[1].set_xlabel("Median PubMed count")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def disease_summary(
    edges: pd.DataFrame,
    pubmed: pd.DataFrame,
    *,
    score_min: Optional[float] = None,
    label: str = "knowledge_filtered",
    top_rank_cutoffs: Tuple[int, ...] = (500, 2000, 10000),
) -> pd.DataFrame:
    """
    Build disease-level summary metrics from disease-gene edges joined to PubMed counts.
    """
    df = edges.copy()

    if score_min is not None and "Score" in df.columns:
        df = df[df["Score"] >= score_min].copy()

    joined = df.merge(pubmed, on="GeneSymbol", how="left", validate="m:1")
    joined["HasPubMed"] = joined["PubMed_Count"].notna()

    def _agg(group: pd.DataFrame) -> pd.Series:
        genes_total = group["GeneSymbol"].nunique()
        sub = group.dropna(subset=["PubMed_Count"]).copy()
        genes_with_pub = sub["GeneSymbol"].nunique()
        coverage = genes_with_pub / genes_total if genes_total else np.nan

        counts = sub["PubMed_Count"].to_numpy(dtype=float)
        ranks = sub["PubMed_Rank"].to_numpy(dtype=float)

        out = {
            "dataset": label,
            "n_genes_total": genes_total,
            "n_genes_with_pubmed": genes_with_pub,
            "coverage_pubmed": coverage,
            "median_pubmed": np.nanmedian(counts) if counts.size else np.nan,
            "mean_pubmed": np.nanmean(counts) if counts.size else np.nan,
            "p90_pubmed": np.nanpercentile(counts, 90) if counts.size else np.nan,
            "gini_pubmed": gini(counts) if counts.size else np.nan,
            "median_rank": np.nanmedian(ranks) if ranks.size else np.nan,
        }

        for k in top_rank_cutoffs:
            out[f"frac_in_top{k}"] = float(np.mean(sub["PubMed_Rank"] <= k)) if genes_with_pub else np.nan

        if genes_with_pub:
            top = sub.sort_values("PubMed_Count", ascending=False).head(5)
            out["top_genes"] = "; ".join([f"{r.GeneSymbol}({int(r.PubMed_Count)})" for r in top.itertuples()])
        else:
            out["top_genes"] = ""

        return pd.Series(out)

    res = joined.groupby("Disease", sort=False).apply(_agg).reset_index()
    res = res.sort_values(["median_pubmed", "n_genes_with_pubmed"], ascending=[False, False]).reset_index(drop=True)
    return res

def plot_top_diseases_bar(
    summary: pd.DataFrame,
    outpath: Path,
    metric: str,
    title: str,
    top_n: int,
    min_genes: int,
) -> None:
    d = summary.dropna(subset=[metric]).copy()
    d = d[d["n_genes_with_pubmed"] >= min_genes].copy()
    d = d.sort_values(metric, ascending=False).head(top_n).iloc[::-1]

    plt.figure(figsize=(11, max(6, 0.35 * len(d))))
    ax = sns.barplot(data=d, x=metric, y="Disease", orient="h")
    ax.set_title(title)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_ylabel("")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

def plot_scatter(
    summary: pd.DataFrame,
    outpath: Path,
    x: str,
    y: str,
    title: str,
    min_genes: int,
) -> None:
    d = summary.dropna(subset=[x, y]).copy()
    d = d[d["n_genes_with_pubmed"] >= min_genes].copy()

    plt.figure(figsize=(9.5, 7))
    ax = sns.scatterplot(
        data=d, x=x, y=y, size="n_genes_with_pubmed",
        sizes=(25, 250), alpha=0.75, legend=False
    )
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

def compare_filtered_full(filtered: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    a = filtered[["Disease", "median_pubmed", "mean_pubmed", "n_genes_with_pubmed", "coverage_pubmed"]].copy()
    b = full[["Disease", "median_pubmed", "mean_pubmed", "n_genes_with_pubmed", "coverage_pubmed"]].copy()

    a = a.rename(columns={
        "median_pubmed": "median_pubmed_filtered",
        "mean_pubmed": "mean_pubmed_filtered",
        "n_genes_with_pubmed": "n_genes_with_pubmed_filtered",
        "coverage_pubmed": "coverage_filtered",
    })
    b = b.rename(columns={
        "median_pubmed": "median_pubmed_full",
        "mean_pubmed": "mean_pubmed_full",
        "n_genes_with_pubmed": "n_genes_with_pubmed_full",
        "coverage_pubmed": "coverage_full",
    })

    m = a.merge(b, on="Disease", how="outer")
    m["delta_median_full_minus_filtered"] = m["median_pubmed_full"] - m["median_pubmed_filtered"]
    m["delta_mean_full_minus_filtered"] = m["mean_pubmed_full"] - m["mean_pubmed_filtered"]
    return m.sort_values("delta_median_full_minus_filtered", ascending=False).reset_index(drop=True)

def plot_compare_median(filtered: pd.DataFrame, full: pd.DataFrame, outpath: Path, min_genes: int) -> None:
    a = filtered[["Disease", "median_pubmed", "n_genes_with_pubmed"]].copy()
    b = full[["Disease", "median_pubmed", "n_genes_with_pubmed"]].copy()
    a = a.rename(columns={"median_pubmed": "median_filtered", "n_genes_with_pubmed": "n_filtered"})
    b = b.rename(columns={"median_pubmed": "median_full", "n_genes_with_pubmed": "n_full"})
    m = a.merge(b, on="Disease", how="inner")
    m = m[(m["n_filtered"] >= min_genes) & (m["n_full"] >= min_genes)].copy()
    if m.empty:
        return

    plt.figure(figsize=(8.5, 8.0))
    ax = sns.scatterplot(data=m, x="median_filtered", y="median_full", alpha=0.7)
    lim = np.nanmax([m["median_filtered"].max(), m["median_full"].max()])
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    ax.set_xlabel("Median PubMed count (filtered)")
    ax.set_ylabel("Median PubMed count (full)")
    ax.set_title("Disease-level attention: filtered vs full (median PubMed count)")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------- Main ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Disease-centric research attention analysis (DISEASES + PubMed).")
    parser.add_argument("--score-min", type=float, default=None,
                        help="Optional minimum DISEASES association score (knowledge files often use integers).")
    parser.add_argument("--min-genes", type=int, default=10,
                        help="Minimum genes with PubMed counts required for a disease to be plotted (default 10).")
    parser.add_argument("--top-n", type=int, default=25,
                        help="How many diseases to show in bar plots (default 25).")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    pubmed_path = here / "all_genes.txt"
    filt_path = here / "human_disease_knowledge_filtered.tsv"
    full_path = here / "human_disease_knowledge_full.tsv"

    outdir = here / "DiseaseAttentionResults"
    figdir = outdir / "Figures"
    ensure_dir(figdir)

    sns.set_theme(
        style="whitegrid",
        context="talk",
        font_scale=0.85,
        rc={"axes.spines.top": False, "axes.spines.right": False, "grid.linestyle": "--", "grid.alpha": 0.35},
    )

    pubmed = load_pubmed_counts(pubmed_path)
    edges_filt = load_diseases_edges(filt_path)
    edges_full = load_diseases_edges(full_path)

    summ_filt = disease_summary(edges_filt, pubmed, score_min=args.score_min, label="knowledge_filtered")
    summ_full = disease_summary(edges_full, pubmed, score_min=args.score_min, label="knowledge_full")

    ensure_dir(outdir)
    summ_filt.to_csv(outdir / "disease_attention__knowledge_filtered.csv", index=False)
    summ_full.to_csv(outdir / "disease_attention__knowledge_full.csv", index=False)

    comp = compare_filtered_full(summ_filt, summ_full)
    comp.to_csv(outdir / "disease_attention__compare_full_vs_filtered.csv", index=False)

    # Plots (filtered dataset)
    plot_top_diseases_bar(
        summ_filt,
        figdir / "top_diseases_by_median_pubmed__knowledge_filtered.png",
        metric="median_pubmed",
        title="Top diseases by median PubMed count (DISEASES Knowledge, filtered)",
        top_n=args.top_n,
        min_genes=args.min_genes,
    )
    plot_top_diseases_bar(
        summ_filt,
        figdir / "top_diseases_by_mean_pubmed__knowledge_filtered.png",
        metric="mean_pubmed",
        title="Top diseases by mean PubMed count (DISEASES Knowledge, filtered)",
        top_n=args.top_n,
        min_genes=args.min_genes,
    )
    plot_top_diseases_bar(
        summ_filt,
        figdir / "top_diseases_by_frac_in_top500__knowledge_filtered.png",
        metric="frac_in_top500",
        title="Top diseases by fraction of genes in top-500 (DISEASES Knowledge, filtered)",
        top_n=args.top_n,
        min_genes=args.min_genes,
    )
    plot_scatter(
        summ_filt,
        figdir / "scatter_n_genes_vs_median_pubmed__knowledge_filtered.png",
        x="n_genes_with_pubmed",
        y="median_pubmed",
        title="Disease gene-set size vs median PubMed count (DISEASES Knowledge, filtered)",
        min_genes=args.min_genes,
    )

    # Compare filtered vs full medians
    plot_compare_median(
        summ_filt,
        summ_full,
        figdir / "compare_filtered_vs_full_median_scatter.png",
        min_genes=args.min_genes,
    )

    # Console summary
    print("=== Loaded ===")
    print(f"PubMed genes (unique symbols): {len(pubmed):,}")
    print(f"Edges (knowledge filtered):   {len(edges_filt):,}")
    print(f"Edges (knowledge full):       {len(edges_full):,}")
    print("")
    print("=== Output ===")
    print(outdir)
    print("")
    print("Top 10 diseases by median PubMed count (filtered):")
    cols = ["Disease", "n_genes_with_pubmed", "median_pubmed", "mean_pubmed", "coverage_pubmed", "top_genes"]
    print(summ_filt[cols].head(10).to_string(index=False))

    if args.score_min is not None and "Score" not in edges_filt.columns:
        print("\n[WARN] --score-min provided but no Score column detected; ignored.")
    plot_over_vs_understudied(
    summ_filt,
    figdir / "overstudied_vs_understudied_diseases.png",
    metric="median_pubmed",
    top_n=15,
    min_genes=args.min_genes,
)

if __name__ == "__main__":
    main()