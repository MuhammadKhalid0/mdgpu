#!/usr/bin/env python3
"""Generate input files for the md-gpu simulator.

Format produced (matches the C++ loader):
    line 1:   N
    lines:    m  x y z   vx vy vz
"""
import os
import math
import numpy as np

INP = os.path.join(os.path.dirname(__file__), "..", "inputs")
os.makedirs(INP, exist_ok=True)

SIGMA = 1.0
R_MIN = 2 ** (1.0 / 6.0) * SIGMA  # Lennard-Jones equilibrium distance ~ 1.1225


def write(name, particles):
    """particles: list of (m, x, y, z, vx, vy, vz)."""
    path = os.path.join(INP, name)
    with open(path, "w") as f:
        f.write(f"{len(particles)}\n")
        for p in particles:
            f.write("{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(*p))
    print(f"wrote {path}  ({len(particles)} particles)")


def write_dem(name, particles, box_L):
    """DEM particles: list of (m, r, x, y, z, vx, vy, vz).
    Also writes a side-car .box file with the wall-box edge length."""
    path = os.path.join(INP, name)
    with open(path, "w") as f:
        f.write(f"{len(particles)}\n")
        for p in particles:
            f.write(("{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} "
                     "{:.8f} {:.8f} {:.8f}\n").format(*p))
    with open(path + ".box", "w") as f:
        f.write(f"{box_L:.8f}\n")
    print(f"wrote {path}  ({len(particles)} particles, box={box_L})")


# ---------------------------------------------------------------------------
# 2-particle sanity tests
# ---------------------------------------------------------------------------
# stable distance: r = 2^(1/6) sigma -> force = 0
write("two_stable.in", [
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, R_MIN, 0.0, 0.0, 0.0, 0.0, 0.0),
])

# attraction: r > r_min  -> particles pulled together
write("two_attract.in", [
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 1.6 * SIGMA, 0.0, 0.0, 0.0, 0.0, 0.0),
])

# repulsion: r < r_min  -> particles pushed apart
write("two_repel.in", [
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 0.95 * SIGMA, 0.0, 0.0, 0.0, 0.0, 0.0),
])


# ---------------------------------------------------------------------------
# Scenario 1 : collision of two lattice blocks (different masses)
# ---------------------------------------------------------------------------
def make_block(nx, ny, nz, origin, vel, mass, spacing=R_MIN):
    p = []
    ox, oy, oz = origin
    vx, vy, vz = vel
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p.append((mass,
                          ox + i * spacing,
                          oy + j * spacing,
                          oz + k * spacing,
                          vx, vy, vz))
    return p

block_a = make_block(6, 6, 6, origin=(-10.0, 0.0, 0.0),
                     vel=(2.0, 0.0, 0.0), mass=1.0)
block_b = make_block(6, 6, 6, origin=( 4.0, 0.0, 0.0),
                     vel=(-2.0, 0.0, 0.0), mass=3.0)
# center b so blocks face each other
shift_y = -(6 - 1) * R_MIN / 2.0
shift_z = -(6 - 1) * R_MIN / 2.0
def shift(parts, dy, dz):
    return [(m, x, y+dy, z+dz, vx, vy, vz) for (m,x,y,z,vx,vy,vz) in parts]
write("collision.in", shift(block_a + block_b, shift_y, shift_z))


# ---------------------------------------------------------------------------
# Scenario 2 : perturbed FCC-like lattice (relaxation / phonons)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
nx = ny = nz = 8
lattice = make_block(nx, ny, nz, origin=(0,0,0), vel=(0,0,0), mass=1.0)
perturbed = []
for (m,x,y,z,vx,vy,vz) in lattice:
    dx, dy, dz = rng.normal(0, 0.05*R_MIN, 3)
    vx, vy, vz = rng.normal(0, 0.3, 3)
    perturbed.append((m, x+dx, y+dy, z+dz, vx, vy, vz))
write("lattice.in", perturbed)


# ---------------------------------------------------------------------------
# Scenario 3 : gas cloud collapsing toward a heavy center
# ---------------------------------------------------------------------------
TARGET_N = 500
MIN_DIST = 0.9 * SIGMA   # minimum inter-particle distance to avoid LJ explosion
accepted = []
attempts = 0
while len(accepted) < TARGET_N and attempts < 200000:
    attempts += 1
    pos = rng.normal(0, 6.0, 3)
    # skip particles too close to origin (heavy core region)
    if np.linalg.norm(pos) < 2.0:
        continue
    # reject if too close to any already-accepted particle
    if accepted:
        diffs = np.array(accepted) - pos
        dists = np.linalg.norm(diffs, axis=1)
        if dists.min() < MIN_DIST:
            continue
    accepted.append(pos)
positions = np.array(accepted)
N = len(positions)
# velocities pointing inward, small tangential noise
radial = -positions / np.linalg.norm(positions, axis=1, keepdims=True)
velocities = 0.5 * radial + rng.normal(0, 0.2, positions.shape)
parts = [(1.0, *positions[i].tolist(), *velocities[i].tolist())
         for i in range(N)]
# one heavy core
parts.append((50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
write("cloud.in", parts)


# ---------------------------------------------------------------------------
# Scaling benchmark inputs (cubic lattices of varying size)
# ---------------------------------------------------------------------------
for n in (4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40):
    parts = make_block(n, n, n, origin=(0,0,0), vel=(0,0,0), mass=1.0)
    # tiny thermal noise
    parts = [(m, x, y, z,
              *rng.normal(0, 0.1, 3).tolist())
             for (m,x,y,z,_,__,___) in parts]
    # Recenter the lattice around the origin so it fits in a box
    # [-L/2, L/2) of edge length L = n * R_MIN.
    box_L = n * R_MIN
    half  = 0.5 * (n - 1) * R_MIN
    parts = [(m, x - half, y - half, z - half, vx, vy, vz)
             for (m, x, y, z, vx, vy, vz) in parts]
    name = f"bench_{n**3:06d}.in"
    write(name, parts)
    # Side-car file with the natural box size for PBC benchmarks.
    with open(os.path.join(INP, name + ".box"), "w") as f:
        f.write(f"{box_L:.8f}\n")


# ===========================================================================
# Exercise 3 : Discrete Element Method (DEM) granular scenarios
# Format per particle: m  r  x y z  vx vy vz
# Box has fixed walls at +/- box_L/2. Gravity is applied at run time (-z).
# ===========================================================================

def no_overlap_sample(n_target, radius, x_range, y_range, z_range,
                      min_gap=1.05, max_attempts=400000):
    """Rejection-sample n_target sphere centres so no two spheres of the
    given radius overlap (centres at least 2*radius*min_gap apart)."""
    pts = []
    min_d = 2.0 * radius * min_gap
    attempts = 0
    while len(pts) < n_target and attempts < max_attempts:
        attempts += 1
        p = np.array([rng.uniform(*x_range),
                      rng.uniform(*y_range),
                      rng.uniform(*z_range)])
        if pts:
            d = np.linalg.norm(np.array(pts) - p, axis=1)
            if d.min() < min_d:
                continue
        pts.append(p)
    return np.array(pts)


# --- Scenario A: single particle drop (verification case) ------------------
# One sphere released from a height; it falls under gravity, hits the floor
# and bounces, losing kinetic energy at each bounce (spring-dashpot damping).
BOX_DROP = 24.0
drop = [(1.0, 1.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0)]  # m r x y z vx vy vz
write_dem("dem_drop.in", drop, BOX_DROP)


# --- Scenario B: granular pile (particles rain down and settle) ------------
# ~600 spheres dropped from the upper region of the box; they collide,
# dissipate energy and settle into a heap on the floor.
BOX_PILE = 30.0
R_PILE   = 0.6
pile_pts = no_overlap_sample(
    600, R_PILE,
    x_range=(-8, 8), y_range=(-8, 8), z_range=(-2, 13),
    min_gap=1.1)
pile = [(1.0, R_PILE, x, y, z,
         *(rng.normal(0, 0.2, 3) + np.array([0, 0, -1.0])).tolist())
        for (x, y, z) in pile_pts]
write_dem("dem_pile.in", pile, BOX_PILE)


# --- Scenario C: column collapse (granular "dam break") --------------------
# A tall packed column of spheres in the centre collapses sideways under
# gravity and spreads across the floor.
BOX_COL = 40.0
R_COL   = 0.7
col_pts = no_overlap_sample(
    900, R_COL,
    x_range=(-4, 4), y_range=(-4, 4), z_range=(-19, 5),
    min_gap=1.05)
col = [(1.0, R_COL, x, y, z, 0.0, 0.0, 0.0) for (x, y, z) in col_pts]
write_dem("dem_collapse.in", col, BOX_COL)


# --- Scenario D: two-size mixture rain -------------------------------------
# Large and small spheres rain into the box: size segregation ("Brazil-nut")
# style demonstration.
BOX_MIX = 30.0
mix_pts_big   = no_overlap_sample(150, 1.0,
                                  (-9, 9), (-9, 9), (0, 12), min_gap=1.15)
mix = [(2.0, 1.0, x, y, z, 0.0, 0.0, -1.0) for (x, y, z) in mix_pts_big]
mix_pts_small = no_overlap_sample(450, 0.5,
                                  (-9, 9), (-9, 9), (0, 12), min_gap=1.15)
mix += [(0.5, 0.5, x, y, z, 0.0, 0.0, -1.0) for (x, y, z) in mix_pts_small]
write_dem("dem_mix.in", mix, BOX_MIX)

