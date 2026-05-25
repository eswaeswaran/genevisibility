#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass

# ---------- CONFIG ----------
PHYLIP_DIR = Path("/Users/mtongoss/Dropbox/Research/projects/MostStudiedGenes/FilesFromStudents/4string/geneset-masked-nodups-v4/orthologs-geneset-masked-nodups-v4")

LISTS = {
    "lowest500": Path("lowest500.transcript_ids.txt"),
    "randomAbove10k": Path("randomAbove10k.transcript_ids.txt"),
    "randomBelow10k": Path("randomBelow10k.transcript_ids.txt"),
    "top500": Path("top500.transcript_ids.txt"),
}

OUTPUT_ROOT = Path("codeml_pairwise_outputs")
CODEML_CANDIDATES = [Path("./codeml"), None]  # None => search PATH

PH_SUFFIX = ".ph"

# cleandata=0 helps masked alignments; codeml still may say NO_SITES for all-ambiguous cases
CODEML_CTL_TEMPLATE = """\
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
# ---------------------------


@dataclass
class Stats:
    total_ph: int = 0
    matched: int = 0
    ran: int = 0
    skipped_not_in_list: int = 0
    skipped_bad_header: int = 0
    skipped_no_sites: int = 0
    failed_other: int = 0


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def read_id_set(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing list file: {path.resolve()}")
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s:
            ids.add(s)
    return ids


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


def read_phylip_header(path: Path) -> tuple[int, int]:
    """
    Reads the first non-empty line and interprets it as: ntax nchar
    Works for both sequential and interleaved PHYLIP.
    """
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
    """
    Minimal validation that won't mistakenly reject interleaved PHYLIP.
    """
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


def run_one(codeml_bin: str, ph_file: Path, out_dir: Path) -> tuple[bool, str]:
    """
    Run codeml in per-gene working directory inside out_dir/_work/<geneid>.
    Copy outputs to out_dir as <geneid>.*.
    """
    gene_id = ph_file.stem
    work_dir = out_dir / "_work" / gene_id
    safe_mkdir(work_dir)

    local_seq = work_dir / ph_file.name
    shutil.copy2(ph_file, local_seq)

    local_outfile = work_dir / f"{gene_id}.codeml.out"
    ctl_path = work_dir / "codeml.ctl"
    ctl_path.write_text(
        CODEML_CTL_TEMPLATE.format(seqfile=local_seq.name, outfile=local_outfile.name),
        encoding="utf-8",
    )

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
        return False, f"CODEML_ERROR: exception launching codeml: {e}"

    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    (work_dir / "codeml.stdout.txt").write_text(stdout_text, encoding="utf-8")
    (work_dir / "codeml.stderr.txt").write_text(stderr_text, encoding="utf-8")

    # Your observed failure mode
    if ("no sites. Got nothing to do" in stdout_text) or ("do not have any resolved nucleotides" in stdout_text):
        return False, "NO_SITES"

    if proc.returncode != 0:
        return False, f"CODEML_ERROR: returncode {proc.returncode}"

    if not local_outfile.exists():
        return False, "OUTFILE_MISSING"

    # Copy key outputs
    shutil.copy2(local_outfile, out_dir / local_outfile.name)
    shutil.copy2(ctl_path, out_dir / f"{gene_id}.ctl")
    shutil.copy2(work_dir / "codeml.stdout.txt", out_dir / f"{gene_id}.stdout.txt")
    shutil.copy2(work_dir / "codeml.stderr.txt", out_dir / f"{gene_id}.stderr.txt")

    return True, "OK"


def main() -> int:
    if not PHYLIP_DIR.exists():
        raise FileNotFoundError(f"PHYLIP_DIR not found: {PHYLIP_DIR.resolve()}")

    codeml_bin = find_codeml()
    print(f"Using codeml: {codeml_bin}")

    list_sets = {}
    for name, path in LISTS.items():
        list_sets[name] = read_id_set(path)
        print(f"Loaded {len(list_sets[name]):,} transcript IDs from {path}")

    safe_mkdir(OUTPUT_ROOT)
    out_dirs = {name: OUTPUT_ROOT / name for name in LISTS.keys()}
    for d in out_dirs.values():
        safe_mkdir(d)

    ph_files = sorted(PHYLIP_DIR.glob(f"*{PH_SUFFIX}"))
    if not ph_files:
        raise FileNotFoundError(f"No {PH_SUFFIX} files found in {PHYLIP_DIR}")
    stats = {name: Stats(total_ph=len(ph_files)) for name in LISTS.keys()}

    for list_name, idset in list_sets.items():
        out_dir = out_dirs[list_name]
        print(f"\n=== Running list: {list_name} ===")

        log_bad = out_dir / "skipped_bad_header.txt"
        log_nosites = out_dir / "skipped_no_sites.txt"
        log_failed = out_dir / "failed_other.txt"

        # reset logs each run
        for lp in [log_bad, log_nosites, log_failed]:
            if lp.exists():
                lp.unlink()

        for ph in ph_files:
            gene_id = ph.stem

            if gene_id not in idset:
                stats[list_name].skipped_not_in_list += 1
                continue

            stats[list_name].matched += 1

            ok_hdr, why = validate_phylip_for_codeml(ph)
            if not ok_hdr:
                stats[list_name].skipped_bad_header += 1
                append_log(log_bad, f"{gene_id}\t{why}")
                continue

            ok, msg = run_one(codeml_bin, ph, out_dir)
            if ok:
                stats[list_name].ran += 1
            else:
                if msg == "NO_SITES":
                    stats[list_name].skipped_no_sites += 1
                    append_log(log_nosites, gene_id)
                else:
                    stats[list_name].failed_other += 1
                    append_log(log_failed, f"{gene_id}\t{msg}")
                    print(f"[FAIL] {gene_id}: {msg}")

        st = stats[list_name]
        print(
            f"{list_name}: matched={st.matched:,} ran={st.ran:,} "
            f"skipped_bad_header={st.skipped_bad_header:,} skipped_no_sites={st.skipped_no_sites:,} "
            f"failed_other={st.failed_other:,}"
        )

    print("\n=== SUMMARY ===")
    for list_name, st in stats.items():
        print(
            f"{list_name}: total_ph={st.total_ph:,} matched={st.matched:,} ran={st.ran:,} "
            f"skipped_not_in_list={st.skipped_not_in_list:,} "
            f"skipped_bad_header={st.skipped_bad_header:,} skipped_no_sites={st.skipped_no_sites:,} "
            f"failed_other={st.failed_other:,}"
        )

    print(f"\nOutputs written under: {OUTPUT_ROOT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
