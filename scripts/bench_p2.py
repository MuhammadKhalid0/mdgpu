#!/usr/bin/env python3
"""Part-2 benchmark: compare three force-evaluation modes.

For each lattice size N, runs the simulator three times:

  baseline   : naive O(N^2), no PBC, no cutoff       (Part 1 reference)
  cutoff     : naive O(N^2) but skips pairs > rcut   (PBC + cutoff)
  cells      : cell-linked-list,  PBC + cutoff       (Part 2 acceleration)

Output: CSV with one row per N, three timing columns.
"""
import os
import re
import subprocess
import sys

ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN   = os.path.join(ROOT, "md")
INP   = os.path.join(ROOT, "inputs")
STEPS = 200
RCUT  = 2.5   # in sigma units
SIGMA = 1.0

# Reasonable size sweep for the three modes. We stop "baseline" earlier
# than "cells" because O(N^2) becomes very slow.
SIZES = [4**3, 6**3, 8**3, 10**3, 12**3, 14**3, 16**3,
         20**3, 24**3, 28**3, 32**3, 40**3]   # 64 .. 64 000
STEPS = 200
RCUT  = 2.5   # in sigma units
SIGMA = 1.0

MS_RE = re.compile(r"Mean ms / step\s*:\s*([\d.eE+-]+)")
TS_RE = re.compile(r"Total sim time\s*:\s*([\d.eE+-]+)")


def run(args):
    out = subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)
    m = MS_RE.search(out)
    t = TS_RE.search(out)
    if not (m and t):
        sys.stderr.write("Parse failure for: " + " ".join(args) + "\n")
        sys.stderr.write(out + "\n")
        return None, None
    return float(m.group(1)), float(t.group(1))


def box_for(n_cubed_filename):
    boxfile = os.path.join(INP, n_cubed_filename + ".box")
    if not os.path.exists(boxfile):
        return None
    with open(boxfile) as f:
        return float(f.readline().strip())


print("N,baseline_ms,cutoff_ms,cells_ms")
for n in SIZES:
    fname = f"bench_{n:06d}.in"
    fpath = os.path.join(INP, fname)
    if not os.path.exists(fpath):
        sys.stderr.write(f"# missing {fpath}\n")
        continue
    L = box_for(fname)
    if L is None:
        sys.stderr.write(f"# missing box file for {fname}\n")
        continue

    # baseline (naive O(N^2), no PBC, no cutoff)
    ms, _ = run([BIN, "-i", fpath, "-n", str(STEPS),
                 "-d", "0", "-t", "0.001"])
    baseline_ms = f"{ms:.4f}" if ms else ""

    # cutoff (naive but with PBC + cutoff)
    ms, _ = run([BIN, "-i", fpath, "-n", str(STEPS),
                 "-d", "0", "-t", "0.001",
                 "-L", f"{L}", "-c", str(RCUT * SIGMA),
                 "-A", "naive"])
    cutoff_ms = f"{ms:.4f}" if ms else ""

    # cells (cell-linked list). Pad box to at least 3.5*rcut so the
    # algorithm constraint L >= 3*rcut is always satisfied even for
    # the smallest lattices (extra empty space around the lattice has
    # no effect on physics or timing — just gives the cell grid room).
    L_cells = max(L, 3.5 * RCUT * SIGMA)
    ms, _ = run([BIN, "-i", fpath, "-n", str(STEPS),
                 "-d", "0", "-t", "0.001",
                 "-L", f"{L_cells}", "-c", str(RCUT * SIGMA),
                 "-A", "cells"])
    cells_ms = f"{ms:.4f}" if ms else ""

    print(f"{n},{baseline_ms},{cutoff_ms},{cells_ms}", flush=True)
