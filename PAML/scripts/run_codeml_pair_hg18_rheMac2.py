#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass

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

HGNC_MAP_TSV = Path("transcript_to_hgnc.tsv")

# The pair you want
SUMMARY_TAXON_A = "hg18"
SUMMARY_TAXON_B = "rheMac2"

PHYLIP_NAME_WIDTH = 10

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
    skipped_bad_format: int = 0
    skipped_missing_pair: int = 0
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
# PHYLIP-ish parser (supports codon-spaced)
# =========================
_SEQ_CHARS = set("ACGTN-")
def _is_seq_token(tok: str) -> bool:
    t = tok.strip().upper()
    return bool(t) and set(t) <= _SEQ_CHARS

def parse_phylip_any(path: Path) -> tuple[list[str], dict[str, str]]:
    lines = [ln.rstrip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty file")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError("missing header ntax nchar")
    ntax = int(header[0])

    taxa: list[str] = []
    seq_tokens: dict[str, list[str]] = {}

    i = 1
    while i < len(lines) and len(taxa) < ntax:
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        name = parts[0]
        rest = parts[1:]

        taxa.append(name)
        seq_tokens[name] = []

        if rest and all(_is_seq_token(t) for t in rest):
            seq_tokens[name].extend(rest)
            i += 1
            continue

        # name-only line: consume following pure-seq lines
        i += 1
        while i < len(lines):
            nxt = lines[i].split()
            if not nxt:
                i += 1
                continue
            if all(_is_seq_token(t) for t in nxt):
                seq_tokens[name].extend(nxt)
                i += 1
            else:
                break

    if len(taxa) < 2:
        raise ValueError("parsed <2 taxa")

    seqs: dict[str, str] = {}
    lengths = set()
    for t in taxa:
        s = "".join(seq_tokens[t]).upper().replace(".", "N")
        s = re.sub(r"[^ACGTN\-]", "", s)
        seqs[t] = s
        lengths.add(len(s))

    if len(lengths) != 1:
        raise ValueError(f"unequal sequence lengths: {sorted(lengths)}")

    L = next(iter(lengths))
    if L % 3 != 0:
        raise ValueError(f"length {L} not multiple of 3")

    return taxa, seqs

def subset_pair_and_drop_gap_codons(seqs: dict[str, str], a: str, b: str) -> tuple[int, dict[str, str]]:
    """
    Make 2-taxon alignment with taxa a and b only.
    Remove codon columns where either has '-' (gap). Keep N codons.
    """
    if a not in seqs or b not in seqs:
        raise KeyError("missing one or both taxa")

    sa = seqs[a]
    sb = seqs[b]
    L = len(sa)
    codon_count = L // 3

    keep = []
    for c in range(codon_count):
        ca = sa[3*c:3*c+3]
        cb = sb[3*c:3*c+3]
        if "-" in ca or "-" in cb:
            continue
        keep.append(c)

    new_a = "".join(sa[3*c:3*c+3] for c in keep)
    new_b = "".join(sb[3*c:3*c+3] for c in keep)
    nchar = len(new_a)
    return nchar, {a: new_a, b: new_b}

def write_phylip_sequential(path: Path, taxa: list[str], seqs: dict[str, str], nchar: int) -> None:
    out = [f"{len(taxa)} {nchar}"]
    for t in taxa:
        out.append(f"{t:<{PHYLIP_NAME_WIDTH}}{seqs[t]}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")

# =========================
# Parse codeml output line
# =========================
_TLINE_RE = re.compile(
    r"^\s*t=\s*([0-9.eE+-]+)\s+S=\s*([0-9.eE+-]+)\s+N=\s*([0-9.eE+-]+)\s+dN/dS=\s*([0-9.eE+-]+)\s+dN\s*=\s*([0-9.eE+-]+)\s+dS\s*=\s*([0-9.eE+-]+)\s*\.?\s*$"
)

def extract_first_tline(codeml_out_text: str) -> tuple[bool, str, dict[str, str]]:
    for ln in codeml_out_text.splitlines():
        m = _TLINE_RE.match(ln)
        if m:
            d = {"t": m.group(1), "S": m.group(2), "N": m.group(3),
                 "omega": m.group(4), "dN": m.group(5), "dS": m.group(6)}
            return True, ln.strip(), d
    return False, "", {}

# =========================
# codeml run (PAIR ONLY)
# =========================
def run_codeml_for_target_pair(codeml_bin: str, ph_file: Path, out_dir: Path) -> tuple[str, Path | None]:
    gene_id = ph_file.stem
    work_dir = out_dir / "_work" / gene_id
    safe_mkdir(work_dir)

    try:
        taxa, seqs = parse_phylip_any(ph_file)
        nchar, pair_seqs = subset_pair_and_drop_gap_codons(seqs, SUMMARY_TAXON_A, SUMMARY_TAXON_B)
    except KeyError:
        return "MISSING_PAIR", None
    except Exception as e:
        return f"BAD_FORMAT: {e}", None

    if nchar == 0:
        cleaned = work_dir / f"{gene_id}.pair.cleaned.ph"
        write_phylip_sequential(cleaned, [SUMMARY_TAXON_A, SUMMARY_TAXON_B], {SUMMARY_TAXON_A:"", SUMMARY_TAXON_B:""}, 0)
        return "NO_SITES", None

    cleaned = work_dir / f"{gene_id}.pair.cleaned.ph"
    write_phylip_sequential(cleaned, [SUMMARY_TAXON_A, SUMMARY_TAXON_B], pair_seqs, nchar)

    local_outfile = work_dir / f"{gene_id}.pairwise.codeml.out"
    ctl_path = work_dir / "codeml.ctl"
    ctl_path.write_text(
        CODEML_CTL_PAIRWISE.format(seqfile=cleaned.name, outfile=local_outfile.name),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [codeml_bin],
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    (work_dir / "codeml.stdout.txt").write_text(stdout_text, encoding="utf-8")
    (work_dir / "codeml.stderr.txt").write_text(stderr_text, encoding="utf-8")

    no_sites = ("no sites. Got nothing to do" in stdout_text) or ("do not have any resolved nucleotides" in stdout_text)

    # Copy artifacts for debugging
    shutil.copy2(cleaned, out_dir / f"{gene_id}.pair.cleaned.ph")
    shutil.copy2(ctl_path, out_dir / f"{gene_id}.pairwise.ctl")
    shutil.copy2(work_dir / "codeml.stdout.txt", out_dir / f"{gene_id}.pairwise.stdout.txt")
    shutil.copy2(work_dir / "codeml.stderr.txt", out_dir / f"{gene_id}.pairwise.stderr.txt")

    if no_sites:
        if local_outfile.exists():
            shutil.copy2(local_outfile, out_dir / local_outfile.name)
        return "NO_SITES", (out_dir / local_outfile.name) if (out_dir / local_outfile.name).exists() else None

    if proc.returncode != 0:
        return f"CODEML_ERROR: returncode {proc.returncode}", None

    if not local_outfile.exists():
        return "OUTFILE_MISSING", None

    shutil.copy2(local_outfile, out_dir / local_outfile.name)
    return "OK", out_dir / local_outfile.name

# =========================
# Pipeline
# =========================
def run_for_lists(codeml_bin: str) -> None:
    print("\n==============================")
    print(f"RUNNING TARGET PAIR ONLY: {SUMMARY_TAXON_A} vs {SUMMARY_TAXON_B}")
    print("==============================")

    ph_files = sorted(PHYLIP_DIR.glob(f"*{PH_SUFFIX}"))
    if not ph_files:
        raise FileNotFoundError(f"No {PH_SUFFIX} files found in {PHYLIP_DIR}")
    list_sets = {name: read_id_set(path) for name, path in LISTS.items()}

    model_root = OUTPUT_ROOT / "pairwise"  # keep same structure
    safe_mkdir(model_root)
    out_dirs = {name: model_root / name for name in LISTS.keys()}
    for d in out_dirs.values():
        safe_mkdir(d)

    hgnc_map = load_hgnc_map(HGNC_MAP_TSV)
    stats = {name: Stats(total_ph=len(ph_files)) for name in LISTS.keys()}

    for list_name, idset in list_sets.items():
        out_dir = out_dirs[list_name]
        print(f"\n--- list: {list_name} ---")

        summary_path = out_dir / f"dnds.{list_name}.tsv"
        if summary_path.exists():
            summary_path.unlink()
        append_log(summary_path, "\t".join([
            "transcript_id","hgnc","taxon_a","taxon_b","t","S","N","dN_dS","dN","dS","status","raw_line"
        ]))

        for ph in ph_files:
            gene_id = ph.stem
            if gene_id not in idset:
                stats[list_name].skipped_not_in_list += 1
                continue

            stats[list_name].matched += 1
            stats[list_name].ran += 1

            status, outfile_path = run_codeml_for_target_pair(codeml_bin, ph, out_dir)

            if status == "OK":
                stats[list_name].ran_ok += 1
            elif status in ("NO_SITES",):
                stats[list_name].ran_no_sites += 1
            elif status == "MISSING_PAIR":
                stats[list_name].skipped_missing_pair += 1
            elif status.startswith("BAD_FORMAT"):
                stats[list_name].skipped_bad_format += 1
            else:
                stats[list_name].failed_other += 1

            hgnc = hgnc_map.get(gene_id, gene_id)

            if outfile_path and outfile_path.exists():
                out_text = outfile_path.read_text(encoding="utf-8", errors="replace")
                found, raw_line, d = extract_first_tline(out_text)
                if found:
                    append_log(summary_path, "\t".join([
                        gene_id, hgnc, SUMMARY_TAXON_A, SUMMARY_TAXON_B,
                        d["t"], d["S"], d["N"], d["omega"], d["dN"], d["dS"], status, raw_line
                    ]))
                else:
                    append_log(summary_path, "\t".join([
                        gene_id, hgnc, SUMMARY_TAXON_A, SUMMARY_TAXON_B,
                        "NA","NA","NA","NA","NA","NA", f"{status};TLINE_NOT_FOUND",""
                    ]))
            else:
                append_log(summary_path, "\t".join([
                    gene_id, hgnc, SUMMARY_TAXON_A, SUMMARY_TAXON_B,
                    "NA","NA","NA","NA","NA","NA", status, ""
                ]))

        st = stats[list_name]
        print(
            f"{list_name}: matched={st.matched:,} ran={st.ran:,} ran_ok={st.ran_ok:,} "
            f"ran_no_sites={st.ran_no_sites:,} missing_pair={st.skipped_missing_pair:,} "
            f"bad_format={st.skipped_bad_format:,} failed_other={st.failed_other:,}"
        )
        print(f"Summary written: {summary_path.resolve()}")

def main() -> int:
    if not PHYLIP_DIR.exists():
        raise FileNotFoundError(f"PHYLIP_DIR not found: {PHYLIP_DIR.resolve()}")

    for _, p in LISTS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing list file: {p.resolve()}")

    codeml_bin = find_codeml()
    print(f"Using codeml: {codeml_bin}")
    safe_mkdir(OUTPUT_ROOT)

    run_for_lists(codeml_bin)

    print(f"\nDONE. All outputs under: {OUTPUT_ROOT.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
