#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass

# ---------- CONFIG ----------
PHYLIP_DIR = Path("/Users/mtongoss/Dropbox/Research/projects/MostStudiedGenes/FilesFromStudents/4string/geneset-masked-nodups-v4/orthologs-geneset-masked-nodups-v4")

# transcript-id list files (one transcript ID per line)
LISTS = {
    "lowest500": Path("lowest500.transcript_ids.txt"),
    "randomAbove10k": Path("randomAbove10k.transcript_ids.txt"),
    "randomBelow10k": Path("randomBelow10k.transcript_ids.txt"),
    "top500": Path("top500.transcript_ids.txt"),
}

# where to store outputs (created under current folder)
OUTPUT_ROOT = Path("codeml_pairwise_outputs")

# codeml binary: prefer ./codeml else PATH
CODEML_CANDIDATES = [Path("./codeml"), None]  # None => "codeml" from PATH

# codeml options for pairwise dN/dS (runmode = -2)
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

     cleandata = 1
        ndata = 1
"""

# transcript id is the filename without trailing ".ph"
PH_SUFFIX = ".ph"
# ---------------------------


@dataclass
class Stats:
    total_ph: int = 0
    matched: int = 0
    ran: int = 0
    skipped_not_in_list: int = 0
    skipped_missing_file: int = 0
    failed: int = 0


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
            # PATH
            p = shutil.which("codeml")
            if p:
                return p
        else:
            if cand.exists() and os.access(cand, os.X_OK):
                return str(cand.resolve())
    raise FileNotFoundError("Could not find codeml. Put it in ./codeml or add it to PATH.")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_one(codeml_bin: str, ph_file: Path, out_dir: Path) -> tuple[bool, str]:
    """
    Run codeml in a per-gene working directory inside out_dir/_work/<geneid>.
    Copy final outputs to out_dir/<geneid>.*.
    """
    gene_id = ph_file.stem  # removes .ph
    work_dir = out_dir / "_work" / gene_id
    safe_mkdir(work_dir)

    # Copy the seqfile into the work dir (codeml likes local paths)
    local_seq = work_dir / ph_file.name
    shutil.copy2(ph_file, local_seq)

    # Prepare outputs
    local_outfile = work_dir / f"{gene_id}.codeml.out"
    ctl_path = work_dir / "codeml.ctl"
    ctl_text = CODEML_CTL_TEMPLATE.format(seqfile=local_seq.name, outfile=local_outfile.name)
    ctl_path.write_text(ctl_text, encoding="utf-8")

    # Run codeml
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
        return False, f"Exception launching codeml: {e}"

    # Save stdout/stderr logs
    (work_dir / "codeml.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (work_dir / "codeml.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        return False, f"codeml returned code {proc.returncode}. See {work_dir}/codeml.stderr.txt"

    if not local_outfile.exists():
        return False, f"codeml finished but outfile missing: {local_outfile}"

    # Copy key outputs to the main out_dir (flat, easy to browse)
    safe_mkdir(out_dir)
    shutil.copy2(local_outfile, out_dir / local_outfile.name)
    shutil.copy2(work_dir / "codeml.stdout.txt", out_dir / f"{gene_id}.stdout.txt")
    shutil.copy2(work_dir / "codeml.stderr.txt", out_dir / f"{gene_id}.stderr.txt")
    shutil.copy2(ctl_path, out_dir / f"{gene_id}.ctl")

    return True, "OK"


def main() -> int:
    if not PHYLIP_DIR.exists():
        raise FileNotFoundError(f"PHYLIP_DIR not found: {PHYLIP_DIR.resolve()}")

    codeml_bin = find_codeml()
    print(f"Using codeml: {codeml_bin}")

    # Load transcript-id sets
    list_sets: dict[str, set[str]] = {}
    for name, path in LISTS.items():
        list_sets[name] = read_id_set(path)
        print(f"Loaded {len(list_sets[name]):,} transcript IDs from {path}")

    # Create output folders
    safe_mkdir(OUTPUT_ROOT)
    out_dirs = {name: OUTPUT_ROOT / name for name in LISTS.keys()}
    for d in out_dirs.values():
        safe_mkdir(d)

    # Collect .ph files
    ph_files = sorted(PHYLIP_DIR.glob(f"*{PH_SUFFIX}"))
    stats = {name: Stats() for name in LISTS.keys()}
    for name in stats:
        stats[name].total_ph = len(ph_files)

    # Run per list (keeps outputs separated cleanly)
    for list_name, idset in list_sets.items():
        out_dir = out_dirs[list_name]
        print(f"\n=== Running list: {list_name} ===")
        ran_here = 0
        matched_here = 0
        failed_here = 0

        for ph in ph_files:
            gene_id = ph.stem
            if gene_id not in idset:
                stats[list_name].skipped_not_in_list += 1
                continue

            matched_here += 1
            stats[list_name].matched += 1

            ok, msg = run_one(codeml_bin, ph, out_dir)
            if ok:
                ran_here += 1
                stats[list_name].ran += 1
            else:
                failed_here += 1
                stats[list_name].failed += 1
                print(f"[FAIL] {gene_id}: {msg}")

        print(
            f"List {list_name}: matched {matched_here:,}, ran {ran_here:,}, failed {failed_here:,}, "
            f"skipped(not in list) {stats[list_name].skipped_not_in_list:,}"
        )

    print("\n=== SUMMARY ===")
    for list_name, st in stats.items():
        print(
            f"{list_name}: total_ph={st.total_ph:,} matched={st.matched:,} ran={st.ran:,} "
            f"failed={st.failed:,} skipped_not_in_list={st.skipped_not_in_list:,}"
        )

    print(f"\nOutputs written under: {OUTPUT_ROOT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())