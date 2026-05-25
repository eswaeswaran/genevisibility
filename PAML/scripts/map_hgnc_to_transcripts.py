#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

GENE_DEFS = "gene-definitions.txt"

INPUT_FILES = [
    "lowest500.txt",
    "randomAbove10k.txt",
    "randomBelow10k.txt",
    "top500.txt",
]

# Output filenames will be "<input_basename>.transcript_ids.txt"
OUT_SUFFIX = ".transcript_ids.txt"


def load_symbol_to_transcripts(gene_defs_path: Path) -> dict[str, list[str]]:
    """
    Parse gene-definitions.txt (tab-separated).
    Uses:
      - transcriptid column (header: transcriptid)
      - symbol column (header: prot)  <-- this appears to contain HGNC symbols in your example
    Returns mapping: symbol -> list of transcript IDs (in file order).
    """
    if not gene_defs_path.exists():
        raise FileNotFoundError(f"Missing {gene_defs_path.resolve()}")

    with gene_defs_path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n")
        if not header:
            raise ValueError("gene-definitions.txt is empty or missing a header line")

        cols = header.split("\t")
        try:
            tx_idx = cols.index("transcriptid")
            sym_idx = cols.index("prot")
        except ValueError as e:
            raise ValueError(
                f"Expected header columns 'transcriptid' and 'prot'. Found: {cols}"
            ) from e

        m: dict[str, list[str]] = defaultdict(list)

        for line_no, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) <= max(tx_idx, sym_idx):
                # malformed row; skip
                continue

            tx = parts[tx_idx].strip()
            sym = parts[sym_idx].strip()

            if not tx or not sym:
                continue

            m[sym].append(tx)

    return dict(m)


def read_symbols(path: Path) -> list[str]:
    """
    Reads an input list: one HGNC symbol per line.
    Blank lines are ignored. Surrounding whitespace stripped.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.resolve()}")

    syms: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            syms.append(s)
    return syms


def main() -> int:
    cwd = Path(".").resolve()
    gene_defs_path = cwd / GENE_DEFS

    symbol_to_txs = load_symbol_to_transcripts(gene_defs_path)

    total_in = 0
    total_mapped = 0

    print(f"Loaded {len(symbol_to_txs):,} unique symbols from {GENE_DEFS}")

    for in_name in INPUT_FILES:
        in_path = cwd / in_name
        syms = read_symbols(in_path)

        out_path = cwd / (in_path.stem + OUT_SUFFIX)

        mapped = 0
        written_lines = 0

        with out_path.open("w", encoding="utf-8") as out:
            for sym in syms:
                total_in += 1
                txs = symbol_to_txs.get(sym)
                if not txs:
                    continue

                # If a symbol has multiple transcript IDs, write just the first one.
                # If you'd rather write ALL transcript IDs, see note below.
                out.write(txs[0] + "\n")
                mapped += 1
                written_lines += 1

        total_mapped += mapped
        print(
            f"{in_name}: {mapped:,}/{len(syms):,} mapped "
            f"({(mapped/len(syms)*100 if syms else 0):.1f}%). "
            f"Wrote {written_lines:,} lines to {out_path.name}"
        )

    print(f"\nTOTAL: {total_mapped:,}/{total_in:,} HGNC symbols mapped across all lists.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
