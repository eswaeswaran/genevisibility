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

# ---- IMPORTANT CHANGES REQUESTED ----
# 1) Only pairwise runs are executed (no one-ratio / m0)
RUN_PAIRWISE = True
RUN_M0 = False  # force OFF

# 2) Always keep ambiguous data
# (codeml still may report "no sites" when a sequence is all N; we will *still record the run* and emit NA summary)
# -----------------------------------

# Canonical 6-taxon topology (kept here but unused since RUN_M0=False)
CANONICAL_TREE_NEWICK = "(((hg18,panTro2),rheMac2),((mm8,rn4),canFam2));"

# Strict PHYLIP taxon-name field width
PHYLIP_NAME_WIDTH = 10

# 3) HGNC mapping (optional). Expected TSV with at least:
#    transcript_id <tab> HGNC_symbol
# If missing, we fall back to transcript_id as "gene name".
HGNC_MAP_TSV = Path("transcript_to_hgnc.tsv")

# 4) The specific pair you want summarized
SUMMARY_TAXON_A = "hg18"
SUMMARY_TAXON_B = "rheMac2"

# =========================
# CODEML CTL TEMPLATES
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
    ran: int = 0               # count of genes we attempted to run codeml for (including NO_SITES)
    ran_ok: int = 0            # codeml produced outfile
    ran_no_sites: int = 0      # codeml ran but reported no sites / unresolved nucleotides
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
    """
    Optional TSV mapping transcript_id -> HGNC_symbol.
    Accepts either:
      transcript<TAB>HGNC
    or a headered TSV containing columns like transcript_id / hgnc / symbol.
    """
    if not path.exists():
        return {}
    txt = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not txt:
        return {}

    # detect header
    header = txt[0].split("\t")
    header_l = [h.strip().lower() for h in header]
    tid_idx = None
    sym_idx = None
    if len(header) >= 2 and any(x in header_l for x in ("transcript", "transcript_id", "transcriptid", "enst")):
        # headered
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
# PHYLIP header validation
# =========================
def read_phylip_header(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError("missing 'ntax nchar' in header")
            return int(parts[0]), int(parts[1])
    raise ValueError("empty file")

def validate_phylip_for_codeml(path: Path) -> tuple[bool, str]:
    try:
        ntax, nchar = read_phylip_header(path)
    except Exception as e:
        return False, f"bad header: {e}"
    if ntax < 2:
        return False, f"ntax={ntax} (<2)"
    if nchar <= 0:
        return False, f"nchar={nchar} (<=0)"
    if nchar % 3 != 0:
        return False, f"nchar={nchar} not divisible by 3"
    return True, "OK"

# =========================
# Parse pairwise output for hg18 vs rheMac2 line
# =========================
_TLINE_RE = re.compile(
    r"^\s*t=\s*([0-9.eE+-]+)\s+S=\s*([0-9.eE+-]+)\s+N=\s*([0-9.eE+-]+)\s+dN/dS=\s*([0-9.eE+-]+)\s+dN\s*=\s*([0-9.eE+-]+)\s+dS\s*=\s*([0-9.eE+-]+)\s*\.?\s*$"
)

def extract_pairwise_tline(codeml_out_text: str, a: str, b: str) -> tuple[bool, str, dict[str, str]]:
    """
    Attempts to find the specific pairwise block for taxa (a,b) and extract the 't= ... dN/dS= ...' line.
    Returns (found, raw_line, parsed_fields)
    parsed_fields keys: t,S,N,omega,dN,dS
    """
    lines = codeml_out_text.splitlines()
    a_l = a.lower()
    b_l = b.lower()

    want_next_t = False
    for ln in lines:
        lnl = ln.lower()

        # Heuristic: codeml usually prints something like "hg18 .. rheMac2" or "2 (hg18) vs. 3 (rheMac2)"
        # We mark that the next matching t-line belongs to that pair.
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

            # If we hit another comparison header without seeing t=, stop waiting
            if (a_l in lnl and b_l in lnl) is False and (" vs" in lnl or "vs." in lnl or ".." in lnl):
                want_next_t = False

    # Fallback: if codeml output only contains one t-line and it is the pair of interest (rare), try direct scan
    for ln in lines:
        m = _TLINE_RE.match(ln)
        if m:
            # not safe to assume which pair; keep as not-found to avoid wrong extraction
            break

    return False, "", {}

# =========================
# CODEML runner (pairwise only)
# =========================
def run_codeml_pairwise(codeml_bin: str, ph_file: Path, out_dir: Path) -> tuple[str, Path | None, Path]:
    """
    Returns (status, outfile_path_or_None, work_dir)
      status in {"OK","NO_SITES","CODEML_ERROR","OUTFILE_MISSING"}
    """
    gene_id = ph_file.stem
    work_dir = out_dir / "_work" / gene_id
    safe_mkdir(work_dir)

    local_seq = work_dir / ph_file.name
    shutil.copy2(ph_file, local_seq)

    local_outfile = work_dir / f"{gene_id}.pairwise.codeml.out"
    ctl_text = CODEML_CTL_PAIRWISE.format(seqfile=local_seq.name, outfile=local_outfile.name)

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

    # Copy standard artifacts out *even if NO_SITES* so you can inspect later
    shutil.copy2(ctl_path, out_dir / f"{gene_id}.pairwise.ctl")
    shutil.copy2(work_dir / "codeml.stdout.txt", out_dir / f"{gene_id}.pairwise.stdout.txt")
    shutil.copy2(work_dir / "codeml.stderr.txt", out_dir / f"{gene_id}.pairwise.stderr.txt")

    if no_sites:
        # codeml typically won't create outfile; treat as a recorded run with NA summary
        if local_outfile.exists():
            shutil.copy2(local_outfile, out_dir / local_outfile.name)
        return "NO_SITES", (out_dir / local_outfile.name) if (out_dir / local_outfile.name).exists() else None, work_dir

    if not local_outfile.exists():
        return "OUTFILE_MISSING", None, work_dir

    # success: copy outfile
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

    # load optional HGNC mapping once
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

        # Summary file requested:
        # "extract the line ... for hg18 vs rheMac2 along with HGNC gene name in an extra file
        #  (named after the group and starting with dnds.)"
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
                # still emit summary row (NA) so downstream has a full table if desired
                hgnc = hgnc_map.get(gene_id, gene_id)
                append_log(
                    summary_path,
                    "\t".join([gene_id, hgnc, SUMMARY_TAXON_A, SUMMARY_TAXON_B,
                              "NA","NA","NA","NA","NA","NA","BAD_HEADER",""])
                )
                continue

            # run codeml pairwise (ALWAYS counted as a run attempt for matched genes)
            stats[list_name].ran += 1
            status, outfile_path, _work_dir = run_codeml_pairwise(codeml_bin, ph, out_dir)

            if status == "OK":
                stats[list_name].ran_ok += 1
            elif status == "NO_SITES":
                stats[list_name].ran_no_sites += 1
            else:
                stats[list_name].failed_other += 1
                append_log(log_failed, f"{gene_id}\t{status}")

            # Extract hg18 vs rheMac2 t-line (or NA)
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
                # No outfile (common for NO_SITES). Still emit a row.
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

    # only pairwise
    if RUN_PAIRWISE:
        run_pairwise_for_lists(codeml_bin)

    print(f"\nDONE. All outputs under: {OUTPUT_ROOT.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())