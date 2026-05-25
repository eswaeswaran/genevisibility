#!/usr/bin/env python3
"""
Step 1: Parse Orphanet "Classifications of rare diseases" XML files (en_product3_*.xml)
and build a reference set of orphan disease names (+ optional synonyms) for later matching.

Usage:
  1) Put this script in the SAME folder as the downloaded XMLs, or point --xml-dir to it
  2) Run:
       python3 step1_build_orphanet_name_set.py --xml-dir Orphanet_Classifications
  3) Outputs:
       orphanet_disease_names.txt
       orphanet_disease_names.tsv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import xml.etree.ElementTree as ET


def norm(s: str) -> str:
    """Conservative normalization for name matching."""
    s = s.strip().lower()
    s = s.replace("&amp;", "and")
    s = re.sub(r"[’'`]", "", s)                 # remove apostrophes
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)        # keep letters, digits, spaces, hyphen
    s = re.sub(r"\s+", " ", s).strip()
    return s


def local(tag: str) -> str:
    """Strip XML namespace if present."""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def extract_names_from_tree(root: ET.Element, include_synonyms: bool) -> set[str]:
    """
    Robustly extract <Name> and optional synonyms from Orphanet XML.
    We do not rely on a single path; we scan for elements named 'Disorder'.
    """
    names: set[str] = set()

    for elem in root.iter():
        if local(elem.tag).lower() != "disorder":
            continue

        # Primary disease name
        name_el = None
        for child in elem.iter():
            if local(child.tag).lower() == "name":
                name_el = child
                break
        if name_el is not None and name_el.text:
            n = norm(name_el.text)
            if n:
                names.add(n)

        if not include_synonyms:
            continue

        # Synonyms can appear under various tag structures; we pick any element whose local tag is 'Synonym'
        for syn in elem.iter():
            if local(syn.tag).lower() == "synonym" and syn.text:
                sn = norm(syn.text)
                if sn:
                    names.add(sn)

    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xml-dir",
        type=Path,
        default=Path("Orphanet_Classifications"),
        help="Directory containing en_product3_*.xml files",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="en_product3_*.xml",
        help="Glob pattern for XML files",
    )
    ap.add_argument(
        "--include-synonyms",
        action="store_true",
        help="Also include Orphanet synonyms in the name set (recommended)",
    )
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="orphanet_disease_names",
        help="Output prefix (writes .txt and .tsv)",
    )
    args = ap.parse_args()

    xml_dir: Path = args.xml_dir
    files = sorted(xml_dir.glob(args.pattern))

    if not files:
        raise SystemExit(f"No files matched {args.pattern} in {xml_dir.resolve()}")

    all_names: set[str] = set()
    per_file_counts = []

    for fp in files:
        try:
            tree = ET.parse(fp)
        except ET.ParseError as e:
            print(f"[WARN] Could not parse {fp.name}: {e}")
            continue

        root = tree.getroot()
        names = extract_names_from_tree(root, include_synonyms=args.include_synonyms)
        all_names |= names
        per_file_counts.append((fp.name, len(names)))

    # Write outputs
    out_txt = Path(f"{args.out_prefix}.txt")
    out_tsv = Path(f"{args.out_prefix}.tsv")

    with out_txt.open("w", encoding="utf-8") as f:
        for n in sorted(all_names):
            f.write(n + "\n")

    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("name_norm\n")
        for n in sorted(all_names):
            f.write(n + "\n")

    # Report
    print("=== Orphanet XML parsing summary ===")
    print(f"XML dir: {xml_dir.resolve()}")
    print(f"Files matched: {len(files)}")
    print(f"Include synonyms: {bool(args.include_synonyms)}")
    print(f"Unique normalized names collected: {len(all_names):,}")
    print(f"Wrote: {out_txt.resolve()}")
    print(f"Wrote: {out_tsv.resolve()}")
    print("\nPer-file extracted name counts (first 10):")
    for fn, c in per_file_counts[:10]:
        print(f"  {fn}: {c:,}")
    if len(per_file_counts) > 10:
        print("  ...")


if __name__ == "__main__":
    main()