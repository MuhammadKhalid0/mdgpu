#!/usr/bin/env python3
"""Run the simulator over benchmark inputs and print a CSV scaling table."""
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN  = os.path.join(ROOT, "md")

SIZES = [4**3, 6**3, 8**3, 10**3, 12**3, 14**3, 16**3,
         20**3, 24**3, 28**3, 32**3, 40**3]
STEPS = 200

print("N,ms_per_step,total_s")
for n in SIZES:
    fn = os.path.join(ROOT, "inputs", f"bench_{n:06d}.in")
    if not os.path.exists(fn):
        print(f"# missing {fn}", file=sys.stderr); continue
    out = subprocess.check_output(
        [BIN, "-i", fn, "-n", str(STEPS), "-d", "0", "-t", "0.001"],
        text=True)
    m = re.search(r"Mean ms / step\s*:\s*([\d.eE+-]+)", out)
    t = re.search(r"Total sim time\s*:\s*([\d.eE+-]+)", out)
    if m and t:
        print(f"{n},{float(m.group(1)):.4f},{float(t.group(1)):.4f}",
              flush=True)
    else:
        print(f"# parse failure for N={n}", file=sys.stderr)
        print(out, file=sys.stderr)
