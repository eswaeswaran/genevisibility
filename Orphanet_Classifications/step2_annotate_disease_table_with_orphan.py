#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def norm(s: str) -> str:
    """Same normalization as Step 1."""
    s = str(s).strip().lower()
    s = s.replace("&amp;", "and")
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_orphan_names(path: Path) -> set[str]:
    names = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                names.add(t)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orphan-names", type=Path, default=Path("orphanet_disease_names.txt"))
    ap.add_argument(
        "--disease-csv",
        type=Path,
        default=Path("DiseaseAttentionResults/disease_attention__knowledge_filtered.csv"),
        help="Disease-level summary CSV from your DISEASES analysis",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("DiseaseAttentionResults/disease_attention__knowledge_filtered_with_orphan.csv"),
    )
    ap.add_argument(
        "--min-genes",
        type=int,
        default=10,
        help="Only for reporting summary; output file keeps all rows",
    )
    args = ap.parse_args()

    orphan_names = load_orphan_names(args.orphan_names)
    df = pd.read_csv(args.disease_csv)

    if "Disease" not in df.columns:
        raise SystemExit(f"Expected a 'Disease' column in {args.disease_csv}. Found: {list(df.columns)}")

    df["Disease_norm"] = df["Disease"].map(norm)
    df["is_orphan"] = df["Disease_norm"].isin(orphan_names)

    # Write full annotated table
    out = df.drop(columns=["Disease_norm"])
    out.to_csv(args.out, index=False)

    # Report: overall match rates and after stability filter
    print("=== Orphan annotation summary ===")
    print(f"Input diseases: {len(df):,}")
    print("is_orphan counts:")
    print(df["is_orphan"].value_counts(dropna=False).to_string())

    if "n_genes_with_pubmed" in df.columns:
        d2 = df[df["n_genes_with_pubmed"] >= args.min_genes].copy()
        print(f"\nAfter filter n_genes_with_pubmed >= {args.min_genes}: {len(d2):,} diseases")
        print(d2["is_orphan"].value_counts(dropna=False).to_string())
    else:
        print("\n[WARN] Column 'n_genes_with_pubmed' not found; cannot report filtered counts.")

    # Show a few matched examples
    matched = df[df["is_orphan"]].head(15)[["Disease"]]
    print("\nExample matched orphan diseases (first 15):")
    print(matched.to_string(index=False))

    print(f"\nWrote: {args.out.resolve()}")


if __name__ == "__main__":
    main()