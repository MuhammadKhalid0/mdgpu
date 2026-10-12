# md-gpu — GPU-accelerated Molecular Dynamics and Granular (DEM) Simulation

A single-file CUDA particle simulator (`src/md.cu`, double precision) developed
for the HESP course and run on the NHR@FAU **TinyGPU** cluster. It supports:

| Part | Physics / feature | Flags |
|------|-------------------|-------|
| 1 | Lennard-Jones (LJ) particles, Velocity-Verlet, open domain, optional cutoff, O(N²) shared-memory tiled kernel | default |
| 2 | Periodic boundary conditions (minimum image) and a GPU **cell-linked-list** neighbour search | `-L`, `-c`, `-A cells` |
| 3 | **Granular / DEM**: spring-dashpot contacts between spheres of different radii, gravity, fixed box walls | `-P dem -k -y -g` |

Output is written as legacy ASCII VTK files, viewable directly in ParaView.

## Repository layout

```
md-gpu/
├── Makefile                  # build + test/scenario/benchmark targets
├── src/md.cu                 # the whole simulator (kernels + host code)
├── scripts/
│   ├── generate_inputs.py    # sanity, scenario, benchmark and DEM inputs
│   ├── bench.py              # Part 1 scaling benchmark
│   ├── bench_p2.py           # Part 2: baseline vs cutoff vs cell list
│   ├── dem_energy.py         # DEM verification: kinetic energy of the drop test
│   └── run_*.sh              # SLURM job scripts (pipeline, bench, dem, cloud)
├── inputs/                   # generated particle files (*.in, *.in.box)
├── bench_results.csv         # Part 1 timings
├── bench_p2_results.csv      # Part 2 timings
├── dem_energy.csv            # DEM drop-test kinetic energy vs time
└── md-*.out                  # SLURM logs of the runs that produced the results
```

## Build

```bash
module load cuda/12.6.2     # on TinyGPU
make                        # builds ./md as a fat binary (sm_70/75/80/86)
```

For a single architecture: `make GENCODE="-gencode arch=compute_86,code=sm_86"`.
`make debug` builds an unoptimised `-g -G` binary.

## Usage

```
./md -i INPUT [options]
  -i  PATH     input particle file (required)
  -o  DIR      output directory (default: out)
  -p  PREFIX   output file prefix (default: particles)
  -n  STEPS    number of time steps (default: 1000)
  -d  EVERY    dump every N steps, 0 = no output (default: 100)
  -t  DT       time step (default: 0.001)
  -s  SIGMA    LJ sigma (default: 1.0)
  -e  EPS      LJ epsilon (default: 1.0)
  -c  RCUT     force cutoff, 0 = none (default: 0)
  -L  LBOX     cubic box edge length; enables PBC (LJ) or walls (DEM)
  -A  ALGO     naive | cells   (cells needs -L > 0, and -c > 0 for LJ)
  -b  BLOCK    CUDA block size (default: 128)
  -P  PHYS     lj | dem (default: lj)
  -k  KSPRING  DEM contact stiffness (default: 1000)
  -y  GAMMA    DEM contact damping (default: 5)
  -g  GRAV     gravity magnitude, applied as (0,0,-g) (default: 0)
```

Examples:

```bash
make inputs                                   # generate all input files

# Part 1: two blocks colliding, open domain
./md -i inputs/collision.in -n 4000 -d 50 -t 0.002 -o out/collision -p p

# Part 2: periodic box with cell lists
./md -i inputs/bench_008000.in -L <box> -c 2.5 -A cells -n 200 -d 0

# Part 3: granular pile in a walled box
./md -i inputs/dem_pile.in -P dem -A cells -L 30 -g 9.81 -k 2000 -y 12 \
     -n 12000 -d 100 -t 0.0008 -o out/dem_pile -p p
```

`make tests`, `make scenarios`, `make dem`, `make bench` and `make bench_p2`
run the complete sets. For the cluster, submit the scripts in `scripts/` with
`sbatch` (e.g. `sbatch scripts/run_slurm.sh`).

## Input format

```
N
m  x y z  vx vy vz          # LJ:  one particle per line
m  r  x y z  vx vy vz       # DEM: radius added after the mass
```

Initial accelerations are computed before the first step. The `*.in.box`
companion files hold the box size belonging to the benchmark/DEM inputs.

## Output format

Legacy ASCII VTK (`UNSTRUCTURED_GRID`: points, mass `m`, velocity `v`), one
file per dump: `<outdir>/<prefix>_NNNNNN.vtk`. In ParaView open the sequence,
apply *Glyph* (Sphere) scaled by `m`, and press play.

## Method

**Lennard-Jones force**

    U(r) = 4ε[(σ/r)^12 − (σ/r)^6]

**Velocity-Verlet** (three kernels per step: position update → force
evaluation → velocity update):

    x(t+Δt) = x + v·Δt + ½·a·Δt²
    v(t+Δt) = v + ½·(a(t) + a(t+Δt))·Δt

**Force algorithms**

* `naive` – all-pairs with shared-memory tiling, O(N²).
* `cells` – particles are binned into a uniform grid with cell edge ≥ cutoff;
  the list is built on the GPU with a lock-free `atomicExch` prepend, and each
  thread visits only its 27 neighbouring cells (PBC wrap). Cost scales ~O(N).

**DEM contacts** – overlapping spheres feel a normal spring (`-k`) plus a
dashpot (`-y`) acting on the relative normal velocity, gravity along −z and
reflecting walls at the box faces. For `cells` the search range is derived
from the largest particle radius.

## Scenarios

| Input | Description |
|-------|-------------|
| `two_stable / two_attract / two_repel` | 2-particle LJ sanity tests (r = 2^(1/6)σ, 1.6σ, 0.95σ) |
| `collision` | two blocks (mass 1 vs 3) colliding along x |
| `lattice` | perturbed LJ lattice at equilibrium spacing |
| `cloud` | Gaussian gas cloud collapsing onto a heavy attractor |
| `dem_drop` | single sphere dropped on the floor and bouncing |
| `dem_pile`, `dem_collapse`, `dem_mix` | granular pile, column collapse, mixed-size grains |

## Verification

Results were verified afterwards with a simulator/visualiser (inspecting the
VTK output of each scenario):

* **LJ sanity tests** behave as expected: the particle pair at the equilibrium
  distance stays put, the 1.6σ pair attracts, the 0.95σ pair repels.
* **DEM single-sphere drop** — `scripts/dem_energy.py` computes the kinetic
  energy from the VTK sequence (`dem_energy.csv`). The energy grows while the
  sphere falls and drops at each bounce because of the dashpot, i.e. the
  contact model dissipates energy as intended.
* **Cell lists** reproduce the cutoff-based all-pairs behaviour while being
  much faster (below).

## Performance

Milliseconds per step (`bench_p2_results.csv`):

| N | baseline (all pairs) | with cutoff | cell list |
|------:|------:|------:|------:|
| 1 000 | 1.15 | 0.96 | 0.58 |
| 4 096 | 4.70 | 3.20 | 0.50 |
| 13 824 | 31.3 | 19.8 | 1.08 |
| 32 768 | 147.9 | 92.5 | 1.61 |
| 64 000 | 577.2 | 357.4 | 3.63 |

The all-pairs kernel grows quadratically in N, while the cell-list version
grows roughly linearly, giving about a 160× speed-up at N = 64 000. Part 1
timings (`bench_results.csv`) show the same O(N²) behaviour.

## Cleaning up

```bash
make clean       # remove binaries
make distclean   # also remove generated inputs and ./out
```
