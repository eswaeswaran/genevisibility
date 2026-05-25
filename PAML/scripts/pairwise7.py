#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass

import numpy as np

# =========================
# CONFIG
# =========================
PHYLIP_DIR = Path("/Users/mtongoss/Dropbox/Research/projects/MostStudiedGenes/FilesFromStudents/4string/geneset-masked-nodups-v4/orthologs-geneset-masked-nodups-v4")
PH_SUFFIX = ".ph"

LISTS = {
    "lowest500": Path("lowest500.transcript_ids.txt"),
    "randomAbove10k": Path("randomAbove10k.transcript_ids.txt"),
    "randomBelow10k": Path("randomBelow10k.transcript_ids.txt"),
    "top500": Path("top500.transcript_ids.txt"),
}

OUTPUT_ROOT = Path("codeml_outputs_v5")
CODEML_CANDIDATES = [Path("./codeml"), None]  # None => PATH

RUN_PAIRWISE = True
RUN_M0 = False  # unused

# Strict PHYLIP taxon-name field width
PHYLIP_NAME_WIDTH = 10

# Optional HGNC mapping: transcript_id <tab> HGNC_symbol
HGNC_MAP_TSV = Path("transcript_to_hgnc.tsv")

# Pair to extract
SUMMARY_TAXON_A = "hg18"
SUMMARY_TAXON_B = "rheMac2"

# =========================
# CODEML CTL
# =========================
CODEML_CTL_PAIRWISE = """\
      seqfile = {seqfile}
     treefile = /dev/null
      outfile = {outfile}

        noisy = 3
      verbose = 1
      runmode = -2

      seqtype = 1
    CodonFreq = 2
        clock = 0
        model = 0
      NSsites = 0

        icode = 0
    fix_kappa = 0
        kappa = 2
    fix_omega = 0
        omega = 1

     cleandata = 0
        ndata = 1
"""

# =========================
# STATS
# =========================
@dataclass
class Stats:
    total_ph: int = 0
    matched: int = 0
    ran: int = 0
    ran_ok: int = 0
    ran_no_sites: int = 0
    skipped_not_in_list: int = 0
    skipped_bad_header: int = 0
    skipped_bad_taxa: int = 0
    failed_other: int = 0

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")

def read_id_set(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing list file: {path.resolve()}")
    out: set[str] = set()
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if s:
            out.add(s)
    return out

def find_codeml() -> str:
    for cand in CODEML_CANDIDATES:
        if cand is None:
            p = shutil.which("codeml")
            if p:
                return p
        else:
            if cand.exists() and os.access(cand, os.X_OK):
                return str(cand.resolve())
    raise FileNotFoundError("Could not find codeml. Put it in ./codeml or add it to PATH.")

# =========================
# HGNC mapping
# =========================
def load_hgnc_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not txt:
        return {}

    header = txt[0].split("\t")
    header_l = [h.strip().lower() for h in header]
    tid_idx = None
    sym_idx = None

    if len(header) >= 2 and any(x in header_l for x in ("transcript", "transcript_id", "transcriptid", "enst")):
        for i, h in enumerate(header_l):
            if h in ("transcript", "transcript_id", "transcriptid", "enst"):
                tid_idx = i
            if h in ("hgnc", "hgnc_symbol", "symbol", "gene", "gene_symbol"):
                sym_idx = i
        start = 1
    else:
        start = 0

    mp: dict[str, str] = {}
    for ln in txt[start:]:
        if not ln.strip():
            continue
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        if tid_idx is None or sym_idx is None:
            tid = parts[0].strip()
            sym = parts[1].strip()
        else:
            if tid_idx >= len(parts) or sym_idx >= len(parts):
                continue
            tid = parts[tid_idx].strip()
            sym = parts[sym_idx].strip()
        if tid and sym:
            mp[tid] = sym
    return mp

# =========================
# PHYLIP parsing + FIXED gap removal
# =========================

_SEQ_CHARS = set("ACGTN-")
def _is_seq_token(tok: str) -> bool:
    t = tok.strip().upper()
    if not t:
        return False
    # accept codon tokens like ATG or --- or longer chunks
    return set(t) <= _SEQ_CHARS

def parse_phylip_any(path: Path) -> tuple[list[str], dict[str, str]]:
    """
    Parse PHYLIP-ish file that may include:
      - standard: name + seq tokens on same line (tokens can be codon-spaced)
      - weird: name alone on a line, sequence tokens on following lines
    Returns (taxa_in_order, seqs) where seqs are uppercase strings with only ACGTN-.
    """
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # drop empty lines
    lines = [ln.rstrip() for ln in raw_lines if ln.strip()]

    if not lines:
        raise ValueError("empty file")

    # header must have ntax nchar
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError("missing 'ntax nchar' in header")
    ntax = int(header[0])
    # nchar is not trusted (because tokens/spaces); we compute from sequences
    _ = int(header[1])

    taxa: list[str] = []
    seqs: dict[str, list[str]] = {}

    i = 1
    current_name: str | None = None

    def flush_current():
        nonlocal current_name
        current_name = None

    while i < len(lines) and len(taxa) < ntax:
        parts = lines[i].split()
        if not parts:
            i += 1
            continue

        # If line begins with a name and has sequence tokens on same line:
        # heuristic: first token is name AND (either there are seq tokens OR next lines are seq tokens).
        name = parts[0]
        rest = parts[1:]

        if rest and all(_is_seq_token(t) for t in rest):
            # name + sequence tokens in this line
            taxa.append(name)
            seqs[name] = rest.copy()
            flush_current()
            i += 1
            continue

        # If line is name-only (or name + non-seq junk), treat as a taxon header
        # and consume subsequent lines that are purely sequence tokens.
        taxa.append(name)
        seqs[name] = []
        current_name = name
        i += 1

        while i < len(lines):
            nxt = lines[i].split()
            if not nxt:
                i += 1
                continue
            # If the entire next line looks like sequence tokens, append
            if all(_is_seq_token(t) for t in nxt):
                seqs[current_name].extend(nxt)
                i += 1
                continue
            # Otherwise assume it's the next taxon header
            break

        flush_current()

    if len(taxa) < 2:
        raise ValueError(f"parsed ntax={len(taxa)} (<2)")

    # join tokens, normalize to ACGTN-
    seq_strs: dict[str, str] = {}
    lengths = set()
    for t in taxa:
        s = "".join(seqs[t]).upper().replace(".", "N")
        s = re.sub(r"[^ACGTN\-]", "", s)
        seq_strs[t] = s
        lengths.add(len(s))

    if len(lengths) != 1:
        raise ValueError(f"unequal sequence lengths: {sorted(lengths)}")

    L = next(iter(lengths))
    if L % 3 != 0:
        raise ValueError(f"sequence length {L} not divisible by 3")

    return taxa, seq_strs

def ungap_keep_N(taxa: list[str], seqs: dict[str, str]) -> tuple[int, dict[str, str]]:
    """
    Remove codon columns where ANY taxon has a gap ('-' anywhere in the codon).
    Keep N/NNN (ambiguous) codons.
    Returns (new_nchar, new_seqs).
    """
    L = len(next(iter(seqs.values())))
    codon_count = L // 3

    keep_codons: list[int] = []
    for c in range(codon_count):
        col = [seqs[t][3*c:3*c+3] for t in taxa]
        if any("-" in cod for cod in col):
            continue
        keep_codons.append(c)

    new_seqs: dict[str, str] = {}
    for t in taxa:
        new_seqs[t] = "".join(seqs[t][3*c:3*c+3] for c in keep_codons)

    new_nchar = len(next(iter(new_seqs.values())))
    return new_nchar, new_seqs

def write_phylip_sequential(path: Path, taxa: list[str], seqs: dict[str, str], nchar: int) -> None:
    """
    Write standard PHYLIP sequential: header + one line per taxon (name padded to 10).
    """
    out = [f"{len(taxa)} {nchar}"]
    for t in taxa:
        out.append(f"{t:<{PHYLIP_NAME_WIDTH}}{seqs[t]}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")

def validate_phylip_for_codeml(path: Path) -> tuple[bool, str]:
    """
    Strong validation: we must be able to parse, sequences equal length, length multiple of 3.
    """
    try:
        taxa, seqs = parse_phylip_any(path)
        if len(taxa) < 2:
            return False, "ntax < 2"
        L = len(next(iter(seqs.values())))
        if L <= 0:
            return False, "nchar <= 0"
        if L % 3 != 0:
            return False, f"nchar={L} not divisible by 3"
        return True, "OK"
    except Exception as e:
        return False, f"bad format: {e}"

# =========================
# Parse pairwise output for hg18 vs rheMac2 line
# =========================
_TLINE_RE = re.compile(
    r"^\s*t=\s*([0-9.eE+-]+)\s+S=\s*([0-9.eE+-]+)\s+N=\s*([0-9.eE+-]+)\s+dN/dS=\s*([0-9.eE+-]+)\s+dN\s*=\s*([0-9.eE+-]+)\s+dS\s*=\s*([0-9.eE+-]+)\s*\.?\s*$"
)

def extract_pairwise_tline(codeml_out_text: str, a: str, b: str) -> tuple[bool, str, dict[str, str]]:
    lines = codeml_out_text.splitlines()
    a_l = a.lower()
    b_l = b.lower()

    want_next_t = False
    for ln in lines:
        lnl = ln.lower()

        if (a_l in lnl and b_l in lnl) and ("vs" in lnl or ".." in lnl or "-" in lnl or "(" in lnl):
            want_next_t = True
            continue

        if want_next_t:
            m = _TLINE_RE.match(ln)
            if m:
                d = {
                    "t": m.group(1),
                    "S": m.group(2),
                    "N": m.group(3),
                    "omega": m.group(4),
                    "dN": m.group(5),
                    "dS": m.group(6),
                }
                return True, ln.strip(), d

            if (a_l in lnl and b_l in lnl) is False and (" vs" in lnl or "vs." in lnl or ".." in lnl):
                want_next_t = False

    return False, "", {}

# =========================
# CODEML runner (pairwise only)
# =========================
def run_codeml_pairwise(codeml_bin: str, ph_file: Path, out_dir: Path) -> tuple[str, Path | None, Path]:
    """
    Preprocess alignment:
      - parse PHYLIP-ish
      - delete codons with gaps ONLY (keep N)
      - write cleaned PHYLIP into workdir
    Then run codeml (runmode=-2).
    Returns (status, outfile_path_or_None, work_dir)
      status in {"OK","NO_SITES","CODEML_ERROR","OUTFILE_MISSING","ZERO_SITES"}
    """
    gene_id = ph_file.stem
    work_dir = out_dir / "_work" / gene_id
    safe_mkdir(work_dir)

    # Parse + remove gap codons (keep N)
    try:
        taxa, seqs = parse_phylip_any(ph_file)
        new_nchar, new_seqs = ungap_keep_N(taxa, seqs)
    except Exception as e:
        return f"CODEML_ERROR: preprocess failed: {e}", None, work_dir

    if new_nchar == 0:
        # This is the exact case you showed (but with the fix, it should only happen if EVERYTHING is gapped)
        cleaned = work_dir / f"{gene_id}.cleaned.ph"
        write_phylip_sequential(cleaned, taxa, {t: "" for t in taxa}, 0)
        return "ZERO_SITES", None, work_dir

    # Write cleaned seqfile
    cleaned = work_dir / f"{gene_id}.cleaned.ph"
    write_phylip_sequential(cleaned, taxa, new_seqs, new_nchar)

    local_outfile = work_dir / f"{gene_id}.pairwise.codeml.out"
    ctl_text = CODEML_CTL_PAIRWISE.format(seqfile=cleaned.name, outfile=local_outfile.name)
    ctl_path = work_dir / "codeml.ctl"
    ctl_path.write_text(ctl_text, encoding="utf-8")

    try:
        proc = subprocess.run(
            [codeml_bin],
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as e:
        return f"CODEML_ERROR: exception launching codeml: {e}", None, work_dir

    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    (work_dir / "codeml.stdout.txt").write_text(stdout_text, encoding="utf-8")
    (work_dir / "codeml.stderr.txt").write_text(stderr_text, encoding="utf-8")

    no_sites = ("no sites. Got nothing to do" in stdout_text) or ("do not have any resolved nucleotides" in stdout_text)
    if proc.returncode != 0 and not no_sites:
        return f"CODEML_ERROR: returncode {proc.returncode}", None, work_dir

    # Copy artifacts out
    shutil.copy2(ctl_path, out_dir / f"{gene_id}.pairwise.ctl")
    shutil.copy2(cleaned, out_dir / f"{gene_id}.pairwise.cleaned.ph")
    shutil.copy2(work_dir / "codeml.stdout.txt", out_dir / f"{gene_id}.pairwise.stdout.txt")
    shutil.copy2(work_dir / "codeml.stderr.txt", out_dir / f"{gene_id}.pairwise.stderr.txt")

    if no_sites:
        if local_outfile.exists():
            shutil.copy2(local_outfile, out_dir / local_outfile.name)
        return "NO_SITES", (out_dir / local_outfile.name) if (out_dir / local_outfile.name).exists() else None, work_dir

    if not local_outfile.exists():
        return "OUTFILE_MISSING", None, work_dir

    shutil.copy2(local_outfile, out_dir / local_outfile.name)
    return "OK", out_dir / local_outfile.name, work_dir

# =========================
# Pipeline
# =========================
def run_pairwise_for_lists(codeml_bin: str) -> None:
    print(f"\n==============================")
    print(f"RUNNING MODEL: pairwise (ONLY)")
    print(f"==============================")

    ph_files = sorted(PHYLIP_DIR.glob(f"*{PH_SUFFIX}"))
    if not ph_files:
        raise FileNotFoundError(f"No {PH_SUFFIX} files found in {PHYLIP_DIR}")

    list_sets = {name: read_id_set(path) for name, path in LISTS.items()}

    model_root = OUTPUT_ROOT / "pairwise"
    safe_mkdir(model_root)
    out_dirs = {name: model_root / name for name in LISTS.keys()}
    for d in out_dirs.values():
        safe_mkdir(d)

    hgnc_map = load_hgnc_map(HGNC_MAP_TSV)
    stats = {name: Stats(total_ph=len(ph_files)) for name in LISTS.keys()}

    for list_name, idset in list_sets.items():
        out_dir = out_dirs[list_name]
        print(f"\n--- pairwise | list: {list_name} ---")

        log_bad_header = out_dir / "skipped_bad_header.txt"
        log_failed = out_dir / "failed_other.txt"
        for lp in [log_bad_header, log_failed]:
            if lp.exists():
                lp.unlink()

        summary_path = out_dir / f"dnds.{list_name}.tsv"
        if summary_path.exists():
            summary_path.unlink()
        append_log(
            summary_path,
            "\t".join([
                "transcript_id",
                "hgnc",
                "taxon_a",
                "taxon_b",
                "t",
                "S",
                "N",
                "dN_dS",
                "dN",
                "dS",
                "status",
                "raw_line",
            ])
        )

        for ph in ph_files:
            gene_id = ph.stem

            if gene_id not in idset:
                stats[list_name].skipped_not_in_list += 1
                continue

            stats[list_name].matched += 1

            ok_hdr, why = validate_phylip_for_codeml(ph)
            if not ok_hdr:
                stats[list_name].skipped_bad_header += 1
                append_log(log_bad_header, f"{gene_id}\t{why}")
                hgnc = hgnc_map.get(gene_id, gene_id)
                append_log(
                    summary_path,
                    "\t".join([gene_id, hgnc, SUMMARY_TAXON_A, SUMMARY_TAXON_B,
                              "NA","NA","NA","NA","NA","NA","BAD_FORMAT",""])
                )
                continue

            stats[list_name].ran += 1
            status, outfile_path, _work_dir = run_codeml_pairwise(codeml_bin, ph, out_dir)

            if status == "OK":
                stats[list_name].ran_ok += 1
            elif status in ("NO_SITES", "ZERO_SITES"):
                stats[list_name].ran_no_sites += 1
            else:
                stats[list_name].failed_other += 1
                append_log(log_failed, f"{gene_id}\t{status}")

            hgnc = hgnc_map.get(gene_id, gene_id)
            if outfile_path and outfile_path.exists():
                out_text = outfile_path.read_text(encoding="utf-8", errors="replace")
                found, raw_line, d = extract_pairwise_tline(out_text, SUMMARY_TAXON_A, SUMMARY_TAXON_B)
                if found:
                    append_log(
                        summary_path,
                        "\t".join([
                            gene_id,
                            hgnc,
                            SUMMARY_TAXON_A,
                            SUMMARY_TAXON_B,
                            d["t"],
                            d["S"],
                            d["N"],
                            d["omega"],
                            d["dN"],
                            d["dS"],
                            status,
                            raw_line,
                        ])
                    )
                else:
                    append_log(
                        summary_path,
                        "\t".join([
                            gene_id, hgnc, SUMMARY_TAXON_A, SUMMARY_TAXON_B,
                            "NA","NA","NA","NA","NA","NA",
                            f"{status};PAIR_NOT_FOUND",
                            "",
                        ])
                    )
            else:
                append_log(
                    summary_path,
                    "\t".join([
                        gene_id, hgnc, SUMMARY_TAXON_A, SUMMARY_TAXON_B,
                        "NA","NA","NA","NA","NA","NA",
                        status,
                        "",
                    ])
                )

        st = stats[list_name]
        print(
            f"{list_name}: matched={st.matched:,} ran={st.ran:,} "
            f"ran_ok={st.ran_ok:,} ran_no_sites={st.ran_no_sites:,} "
            f"skipped_bad_header={st.skipped_bad_header:,} failed_other={st.failed_other:,}"
        )
        print(f"Summary written: {summary_path.resolve()}")

    print(f"\nPairwise outputs written under: {model_root.resolve()}")

def main() -> int:
    if not PHYLIP_DIR.exists():
        raise FileNotFoundError(f"PHYLIP_DIR not found: {PHYLIP_DIR.resolve()}")

    codeml_bin = find_codeml()
    print(f"Using codeml: {codeml_bin}")

    for name, p in LISTS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing list file for {name}: {p.resolve()}")

    safe_mkdir(OUTPUT_ROOT)

    if RUN_PAIRWISE:
        run_pairwise_for_lists(codeml_bin)

    print(f"\nDONE. All outputs under: {OUTPUT_ROOT.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
