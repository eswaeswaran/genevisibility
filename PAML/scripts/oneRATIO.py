#!/usr/bin/env python3
from __future__ import annotations

import os
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

RUN_PAIRWISE = True
RUN_M0 = True

# Canonical 6-taxon topology (edit if you prefer a different rooting/topology)
# This matches your taxa names from codeml output.
CANONICAL_TREE_NEWICK = "(((hg18,panTro2),rheMac2),((mm8,rn4),canFam2));"

# Strict PHYLIP taxon-name field width
PHYLIP_NAME_WIDTH = 10

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

CODEML_CTL_M0 = """\
      seqfile = {seqfile}
     treefile = {treefile}
      outfile = {outfile}

        noisy = 3
      verbose = 1
      runmode = 0

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
    skipped_not_in_list: int = 0
    skipped_bad_header: int = 0
    skipped_bad_taxa: int = 0
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
# PHYLIP header + taxa extraction (robust)
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

def extract_taxa_strict_phylip(path: Path) -> list[str]:
    """
    Reads ntax from header, then takes the FIRST ntax non-empty lines and extracts
    the taxon name as the first PHYLIP_NAME_WIDTH characters (strict PHYLIP),
    falling back to split() for short lines.
    Works for sequential and interleaved PHYLIP as long as the first block has names.
    """
    ntax, _ = read_phylip_header(path)
    taxa: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        # skip to first non-empty header line
        for line in f:
            if line.strip():
                break
        # now capture next ntax non-empty lines
        for line in f:
            if len(taxa) >= ntax:
                break
            if not line.strip():
                continue
            if len(line) >= PHYLIP_NAME_WIDTH:
                name = line[:PHYLIP_NAME_WIDTH].strip()
                if not name:
                    # fallback
                    parts = line.strip().split()
                    name = parts[0] if parts else ""
            else:
                parts = line.strip().split()
                name = parts[0] if parts else ""
            if name:
                taxa.append(name)

    # unique-preserve-order
    uniq = list(dict.fromkeys(taxa))
    return uniq

# =========================
# NEWICK prune: canonical -> induced subtree
# =========================
class Node:
    __slots__ = ("name", "children")
    def __init__(self, name: str | None = None, children: list["Node"] | None = None):
        self.name = name
        self.children = children or []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

def parse_newick(s: str) -> Node:
    s = s.strip()
    if not s.endswith(";"):
        raise ValueError("Newick must end with ';'")
    s = s[:-1]
    i = 0
    stack: list[Node] = []
    cur = Node()

    def read_name() -> str:
        nonlocal i
        start = i
        while i < len(s) and s[i] not in ",()":
            if s[i] == ":":
                i += 1
                while i < len(s) and s[i] not in ",()":
                    i += 1
                break
            i += 1
        return s[start:i].strip()

    while i < len(s):
        ch = s[i]
        if ch == "(":
            stack.append(cur)
            new = Node()
            cur.children.append(new)
            cur = new
            i += 1
        elif ch == ",":
            parent = stack[-1]
            new = Node()
            parent.children.append(new)
            cur = new
            i += 1
        elif ch == ")":
            i += 1
            nm = read_name()
            if nm:
                cur.name = nm
            cur = stack.pop()
        else:
            nm = read_name()
            if nm:
                cur.name = nm
    return cur

def prune_tree(root: Node, keep: set[str]) -> Node | None:
    if root.is_leaf():
        return root if (root.name in keep) else None
    new_children: list[Node] = []
    for c in root.children:
        pc = prune_tree(c, keep)
        if pc is not None:
            new_children.append(pc)
    root.children = new_children
    if len(root.children) == 0:
        return None
    if len(root.children) == 1:
        return root.children[0]  # suppress degree-2
    return root

def to_newick(root: Node) -> str:
    if root.is_leaf():
        if not root.name:
            raise ValueError("Leaf without name")
        return root.name
    return "(" + ",".join(to_newick(c) for c in root.children) + ")" + (root.name or "")

def make_induced_tree(taxa: list[str]) -> tuple[bool, str, str]:
    keep = set(taxa)
    base = parse_newick(CANONICAL_TREE_NEWICK)
    pruned = prune_tree(base, keep)
    if pruned is None:
        return False, "pruning removed all leaves", ""

    # count leaves
    leaves: list[str] = []
    def collect(n: Node):
        if n.is_leaf():
            leaves.append(n.name or "")
        else:
            for cc in n.children:
                collect(cc)
    collect(pruned)
    leaves = [x for x in leaves if x]
    if len(set(leaves)) < 2:
        return False, f"tree after pruning has <2 leaves: {leaves}", ""

    return True, "OK", to_newick(pruned) + ";"

# =========================
# CODEML runner
# =========================
def run_codeml(codeml_bin: str, model_name: str, ph_file: Path, out_dir: Path) -> tuple[bool, str]:
    gene_id = ph_file.stem
    work_dir = out_dir / "_work" / gene_id
    safe_mkdir(work_dir)

    local_seq = work_dir / ph_file.name
    shutil.copy2(ph_file, local_seq)

    local_outfile = work_dir / f"{gene_id}.{model_name}.codeml.out"

    if model_name == "pairwise":
        ctl_text = CODEML_CTL_PAIRWISE.format(seqfile=local_seq.name, outfile=local_outfile.name)

    elif model_name == "m0":
        taxa = extract_taxa_strict_phylip(ph_file)
        if len(set(taxa)) < 2:
            return False, f"BAD_TAXA: extracted <2 taxa from phylip first block: {taxa}"

        ok_tree, why, newick = make_induced_tree(taxa)
        if not ok_tree:
            return False, f"BAD_TAXA: {why}"

        tree_path = work_dir / "species.tree"
        tree_path.write_text(newick + "\n", encoding="utf-8")
        ctl_text = CODEML_CTL_M0.format(seqfile=local_seq.name, treefile=tree_path.name, outfile=local_outfile.name)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

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
        return False, f"CODEML_ERROR: exception launching codeml: {e}"

    stdout_text = proc.stdout or ""
    stderr_text = proc.stderr or ""
    (work_dir / "codeml.stdout.txt").write_text(stdout_text, encoding="utf-8")
    (work_dir / "codeml.stderr.txt").write_text(stderr_text, encoding="utf-8")

    if ("no sites. Got nothing to do" in stdout_text) or ("do not have any resolved nucleotides" in stdout_text):
        return False, "NO_SITES"

    if proc.returncode != 0:
        return False, f"CODEML_ERROR: returncode {proc.returncode}"

    if not local_outfile.exists():
        return False, "OUTFILE_MISSING"

    # copy outputs (flat)
    shutil.copy2(local_outfile, out_dir / local_outfile.name)
    shutil.copy2(ctl_path, out_dir / f"{gene_id}.{model_name}.ctl")
    shutil.copy2(work_dir / "codeml.stdout.txt", out_dir / f"{gene_id}.{model_name}.stdout.txt")
    shutil.copy2(work_dir / "codeml.stderr.txt", out_dir / f"{gene_id}.{model_name}.stderr.txt")
    if model_name == "m0":
        shutil.copy2(work_dir / "species.tree", out_dir / f"{gene_id}.species.tree")

    return True, "OK"

# =========================
# Pipeline
# =========================
def run_model_for_lists(codeml_bin: str, model_name: str) -> None:
    print(f"\n==============================")
    print(f"RUNNING MODEL: {model_name}")
    print(f"==============================")

    ph_files = sorted(PHYLIP_DIR.glob(f"*{PH_SUFFIX}"))
    if not ph_files:
        raise FileNotFoundError(f"No {PH_SUFFIX} files found in {PHYLIP_DIR}")

    list_sets = {name: read_id_set(path) for name, path in LISTS.items()}

    model_root = OUTPUT_ROOT / model_name
    safe_mkdir(model_root)
    out_dirs = {name: model_root / name for name in LISTS.keys()}
    for d in out_dirs.values():
        safe_mkdir(d)

    stats = {name: Stats(total_ph=len(ph_files)) for name in LISTS.keys()}

    for list_name, idset in list_sets.items():
        out_dir = out_dirs[list_name]
        print(f"\n--- {model_name} | list: {list_name} ---")

        log_bad_header = out_dir / "skipped_bad_header.txt"
        log_bad_taxa = out_dir / "skipped_bad_taxa.txt"
        log_nosites = out_dir / "skipped_no_sites.txt"
        log_failed = out_dir / "failed_other.txt"

        for lp in [log_bad_header, log_bad_taxa, log_nosites, log_failed]:
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
                append_log(log_bad_header, f"{gene_id}\t{why}")
                continue

            ok, msg = run_codeml(codeml_bin, model_name, ph, out_dir)
            if ok:
                stats[list_name].ran += 1
            else:
                if msg == "NO_SITES":
                    stats[list_name].skipped_no_sites += 1
                    append_log(log_nosites, gene_id)
                elif msg.startswith("BAD_TAXA:"):
                    stats[list_name].skipped_bad_taxa += 1
                    append_log(log_bad_taxa, f"{gene_id}\t{msg}")
                else:
                    stats[list_name].failed_other += 1
                    append_log(log_failed, f"{gene_id}\t{msg}")

        st = stats[list_name]
        print(
            f"{list_name}: matched={st.matched:,} ran={st.ran:,} "
            f"skipped_bad_header={st.skipped_bad_header:,} skipped_bad_taxa={st.skipped_bad_taxa:,} "
            f"skipped_no_sites={st.skipped_no_sites:,} failed_other={st.failed_other:,}"
        )

    print(f"\nModel outputs written under: {model_root.resolve()}")

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
        run_model_for_lists(codeml_bin, "pairwise")
    if RUN_M0:
        run_model_for_lists(codeml_bin, "m0")

    print(f"\nDONE. All outputs under: {OUTPUT_ROOT.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
