#!/usr/bin/env python3
"""Verification case for the DEM solver.

Reads the VTK sequence of the single-particle drop and computes the total
kinetic energy of the system at each dumped time step. A particle falling
under gravity and bouncing on the floor should lose kinetic energy at every
bounce because of the spring-dashpot damping -> energy decays in steps.

Usage:  python3 scripts/dem_energy.py [out/dem_drop] [dt] [dump_every]
Writes: dem_energy.csv  (columns: step, time, kinetic_energy)
"""
import os
import sys
import glob
import numpy as np

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "out/dem_drop"
DT     = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0008
DUMP   = int(sys.argv[3])   if len(sys.argv) > 3 else 20


def read_vtk(path):
    """Return (masses, velocities) arrays from one legacy-ASCII VTK file."""
    with open(path) as f:
        lines = f.read().split("\n")
    n = 0
    i = 0
    while i < len(lines):
        if lines[i].startswith("POINTS"):
            n = int(lines[i].split()[1])
            break
        i += 1
    masses = None
    vels = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("SCALARS m"):
            i += 2  # skip LOOKUP_TABLE line
            masses = np.array([float(lines[i + k]) for k in range(n)])
            i += n
        elif ln.startswith("VECTORS v"):
            i += 1
            vels = np.array([[float(x) for x in lines[i + k].split()]
                             for k in range(n)])
            i += n
        else:
            i += 1
    return masses, vels


files = sorted(glob.glob(os.path.join(OUTDIR, "*.vtk")))
if not files:
    sys.exit(f"No VTK files found in {OUTDIR}")

rows = []
for fn in files:
    base = os.path.basename(fn)
    step = int("".join(ch for ch in base if ch.isdigit()))
    m, v = read_vtk(fn)
    ke = 0.5 * np.sum(m * np.sum(v * v, axis=1))
    rows.append((step, step * DT, ke))

with open("dem_energy.csv", "w") as f:
    f.write("step,time,kinetic_energy\n")
    for step, t, ke in rows:
        f.write(f"{step},{t:.6f},{ke:.8f}\n")

print("wrote dem_energy.csv")
print("step      time        KE")
for step, t, ke in rows[::max(1, len(rows)//20)]:
    print(f"{step:6d}  {t:9.4f}  {ke:12.6f}")
