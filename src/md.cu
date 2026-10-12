// md.cu - GPU-accelerated molecular dynamics with Lennard-Jones potential
// and Velocity-Verlet integration.
//
// Part 1 features:
//   * O(N^2) all-pairs LJ force kernel with shared-memory tiling
//   * Velocity-Verlet integrator
//   * Optional cut-off radius (-c)
//
// Part 2 acceleration features:
//   * Periodic boundary conditions in a cubic box (-L)
//     - Particles wrapped to [-L/2, L/2) every step
//     - Minimum-image convention for pair distances
//   * Cell-linked list ("binning") acceleration (-A cells)
//     - Cells of edge length >= rcut form a uniform 3D grid
//     - GPU build via atomicExch (lock-free linked-list prepend)
//     - Each thread visits its 27 neighbour cells (with PBC wrap)
//
// Build: see ../Makefile
//
// Input file format (plain text):
//   line 1:  N
//   lines 2..N+1:  m  x y z  vx vy vz
//
// Output: VTK legacy ASCII files (one per dump).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <iomanip>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                              \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Apply minimum-image convention in 1D: returns dx shifted into [-L/2, L/2).
__device__ __forceinline__ double mic1(double dx, double L, double invL) {
    // round to nearest integer multiple of L and subtract
    return dx - L * rint(dx * invL);
}

// Wrap positions back into the periodic box [-L/2, L/2).
// Only called when PBC is active (Lbox > 0).
__global__ void wrap_pos_kernel(double4* pos_m, int N, double L, double invL)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double4 p = pos_m[i];
    p.x -= L * floor(p.x * invL + 0.5);
    p.y -= L * floor(p.y * invL + 0.5);
    p.z -= L * floor(p.z * invL + 0.5);
    pos_m[i] = p;
}

// Compute Lennard-Jones acceleration for each particle (naive O(N^2)).
// LJ potential: U(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]
// Force on i from j (vector form):
//   F_ij = 48 eps / r^2 * [ (sigma/r)^12 - 0.5 (sigma/r)^6 ] * (r_i - r_j)
// Acceleration a_i = (1/m_i) * sum_j F_ij
//
// Uses shared-memory tiling (O(N^2/p) per thread).
// If Lbox > 0, applies minimum-image convention (PBC).
__global__ void lj_accel_kernel(const double4* __restrict__ pos_m, // (x,y,z,m)
                                double3*       __restrict__ acc,
                                int            N,
                                double         sigma,
                                double         eps,
                                double         rcut2,
                                double         Lbox,
                                double         invL)
{
    extern __shared__ double4 tile[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double4 pi = (i < N) ? pos_m[i] : make_double4(0.0, 0.0, 0.0, 1.0);

    double ax = 0.0, ay = 0.0, az = 0.0;

    const double sigma2 = sigma * sigma;
    const double eps48  = 48.0 * eps;
    const bool   pbc    = (Lbox > 0.0);

    int numTiles = (N + blockDim.x - 1) / blockDim.x;
    for (int t = 0; t < numTiles; ++t) {
        int j = t * blockDim.x + threadIdx.x;
        tile[threadIdx.x] = (j < N) ? pos_m[j]
                                    : make_double4(0.0, 0.0, 0.0, 0.0);
        __syncthreads();

        int jmax = min(blockDim.x, N - t * blockDim.x);
        #pragma unroll 8
        for (int k = 0; k < jmax; ++k) {
            int jg = t * blockDim.x + k;
            if (jg == i) continue;
            double4 pj = tile[k];
            double dx = pi.x - pj.x;
            double dy = pi.y - pj.y;
            double dz = pi.z - pj.z;
            if (pbc) {
                dx = mic1(dx, Lbox, invL);
                dy = mic1(dy, Lbox, invL);
                dz = mic1(dz, Lbox, invL);
            }
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < 1.0e-12) continue;
            if (rcut2 > 0.0 && r2 > rcut2) continue;
            double inv2  = sigma2 / r2;
            double inv6  = inv2 * inv2 * inv2;
            double inv12 = inv6 * inv6;
            // magnitude of F/r  =  48 eps / r^2 * (inv12 - 0.5 inv6)
            double fOverR = eps48 * (inv12 - 0.5 * inv6) / r2;
            ax += fOverR * dx;
            ay += fOverR * dy;
            az += fOverR * dz;
        }
        __syncthreads();
    }

    if (i < N) {
        double invm = 1.0 / pi.w;
        acc[i] = make_double3(ax * invm, ay * invm, az * invm);
    }
}

// ---------------------------------------------------------------------------
// Cell-linked-list kernels (Part 2 acceleration)
// ---------------------------------------------------------------------------

// Reset all cell heads to -1 (empty list sentinel).
__global__ void bin_reset_kernel(int* cellHead, int nCells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;
    cellHead[i] = -1;
}

// Build linked lists: each particle prepends itself to its cell's list.
// Lock-free using atomicExch on the head pointer.
__global__ void bin_build_kernel(const double4* __restrict__ pos_m,
                                 int* cellHead, int* nextOf,
                                 int N, int nx, double Lbox, double invCell)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double4 p = pos_m[i];
    // Map position in [-L/2, L/2) to cell index in [0, nx).
    int cx = (int)floor((p.x + 0.5 * Lbox) * invCell);
    int cy = (int)floor((p.y + 0.5 * Lbox) * invCell);
    int cz = (int)floor((p.z + 0.5 * Lbox) * invCell);
    if (cx < 0) cx = 0; else if (cx >= nx) cx = nx - 1;
    if (cy < 0) cy = 0; else if (cy >= nx) cy = nx - 1;
    if (cz < 0) cz = 0; else if (cz >= nx) cz = nx - 1;
    int c = cx + nx * (cy + nx * cz);
    // Atomic linked-list prepend: nextOf[i] := old head; head := i
    int old = atomicExch(&cellHead[c], i);
    nextOf[i] = old;
}

// LJ force computation using the cell-linked list.
// Each thread = one particle i. Visits the 27 neighbour cells (with PBC
// wrap-around) and walks each cell's linked list.
__global__ void lj_accel_cells_kernel(const double4* __restrict__ pos_m,
                                      double3*       __restrict__ acc,
                                      const int* __restrict__ cellHead,
                                      const int* __restrict__ nextOf,
                                      int N, int nx,
                                      double sigma, double eps, double rcut2,
                                      double Lbox, double invL, double invCell)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double4 pi = pos_m[i];

    // Cell index of particle i.
    int cx = (int)floor((pi.x + 0.5 * Lbox) * invCell);
    int cy = (int)floor((pi.y + 0.5 * Lbox) * invCell);
    int cz = (int)floor((pi.z + 0.5 * Lbox) * invCell);
    if (cx < 0) cx = 0; else if (cx >= nx) cx = nx - 1;
    if (cy < 0) cy = 0; else if (cy >= nx) cy = nx - 1;
    if (cz < 0) cz = 0; else if (cz >= nx) cz = nx - 1;

    const double sigma2 = sigma * sigma;
    const double eps48  = 48.0 * eps;

    double ax = 0.0, ay = 0.0, az = 0.0;

    #pragma unroll
    for (int dz = -1; dz <= 1; ++dz) {
        int nz = cz + dz; if (nz < 0) nz += nx; else if (nz >= nx) nz -= nx;
        #pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy; if (ny < 0) ny += nx; else if (ny >= nx) ny -= nx;
            #pragma unroll
            for (int dxi = -1; dxi <= 1; ++dxi) {
                int nxi = cx + dxi; if (nxi < 0) nxi += nx;
                else if (nxi >= nx) nxi -= nx;
                int c = nxi + nx * (ny + nx * nz);
                int j = cellHead[c];
                while (j >= 0) {
                    if (j != i) {
                        double4 pj = pos_m[j];
                        double dx = mic1(pi.x - pj.x, Lbox, invL);
                        double dy = mic1(pi.y - pj.y, Lbox, invL);
                        double dz2 = mic1(pi.z - pj.z, Lbox, invL);
                        double r2 = dx*dx + dy*dy + dz2*dz2;
                        if (r2 >= 1.0e-12 &&
                            (rcut2 <= 0.0 || r2 <= rcut2)) {
                            double inv2  = sigma2 / r2;
                            double inv6  = inv2 * inv2 * inv2;
                            double inv12 = inv6 * inv6;
                            double fOverR = eps48 * (inv12 - 0.5 * inv6) / r2;
                            ax += fOverR * dx;
                            ay += fOverR * dy;
                            az += fOverR * dz2;
                        }
                    }
                    j = nextOf[j];
                }
            }
        }
    }

    double invm = 1.0 / pi.w;
    acc[i] = make_double3(ax * invm, ay * invm, az * invm);
}

// ---------------------------------------------------------------------------
// Discrete Element Method (DEM) kernels  -- Exercise 3
// ---------------------------------------------------------------------------
//
// Granular spheres interacting via the linear spring-dashpot contact model.
// Two spheres i,j (radii Ri,Rj) interact ONLY while overlapping, i.e. when
//   d = |x_i - x_j| < Ri + Rj.
// Overlap depth:  delta = (Ri + Rj) - d  > 0
// Unit normal  :  n = (x_i - x_j) / d          (points from j to i)
// Normal force on i:
//   F_n = ( k * delta  -  gamma * (v_rel . n) ) * n
//        \____spring___/   \______dashpot______/
// The dashpot term removes energy on approach -> dissipative (granular).
//
// Body force: gravity  a += (0, 0, -g).
// Boundaries: fixed walls of a box [-half, half]^3. Each wall acts like a
// spring-dashpot contact against an infinitely heavy plane.

// Apply the six box walls to a particle. Accumulates force (not accel).
__device__ __forceinline__ void apply_walls(double3 p, double3 v, double R,
                                            double k, double gamma, double half,
                                            double& fx, double& fy, double& fz)
{
    // For each axis: lower wall at -half (normal +), upper wall at +half (normal -).
    // x
    double pen = R - (p.x + half);          // lower x
    if (pen > 0.0) fx += k * pen - gamma * v.x;
    pen = R - (half - p.x);                 // upper x
    if (pen > 0.0) fx += -(k * pen) - gamma * v.x;
    // y
    pen = R - (p.y + half);
    if (pen > 0.0) fy += k * pen - gamma * v.y;
    pen = R - (half - p.y);
    if (pen > 0.0) fy += -(k * pen) - gamma * v.y;
    // z
    pen = R - (p.z + half);
    if (pen > 0.0) fz += k * pen - gamma * v.z;
    pen = R - (half - p.z);
    if (pen > 0.0) fz += -(k * pen) - gamma * v.z;
}

// Naive O(N^2) DEM force kernel (spring-dashpot + gravity + walls).
__global__ void dem_accel_kernel(const double4* __restrict__ pos_m,
                                 const double3* __restrict__ vel,
                                 const double*  __restrict__ rad,
                                 double3*       __restrict__ acc,
                                 int N, double k, double gamma,
                                 double gz, double half, int useWalls)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double4 pi = pos_m[i];
    double3 vi = vel[i];
    double  Ri = rad[i];

    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        double4 pj = pos_m[j];
        double dx = pi.x - pj.x;
        double dy = pi.y - pj.y;
        double dz = pi.z - pj.z;
        double d2 = dx*dx + dy*dy + dz*dz;
        double sumR = Ri + rad[j];
        if (d2 >= sumR * sumR || d2 < 1.0e-18) continue;
        double d  = sqrt(d2);
        double inv = 1.0 / d;
        double nx = dx * inv, ny = dy * inv, nz = dz * inv;
        double delta = sumR - d;
        double3 vj = vel[j];
        double vrx = vi.x - vj.x, vry = vi.y - vj.y, vrz = vi.z - vj.z;
        double vn = vrx*nx + vry*ny + vrz*nz;
        double fmag = k * delta - gamma * vn;
        fx += fmag * nx;
        fy += fmag * ny;
        fz += fmag * nz;
    }

    if (useWalls) {
        double3 pv = make_double3(pi.x, pi.y, pi.z);
        apply_walls(pv, vi, Ri, k, gamma, half, fx, fy, fz);
    }

    double invm = 1.0 / pi.w;
    acc[i] = make_double3(fx * invm, fy * invm, fz * invm + gz);
}

// Cell-list DEM force kernel (non-periodic, clamped neighbour cells).
__global__ void dem_accel_cells_kernel(const double4* __restrict__ pos_m,
                                       const double3* __restrict__ vel,
                                       const double*  __restrict__ rad,
                                       double3*       __restrict__ acc,
                                       const int* __restrict__ cellHead,
                                       const int* __restrict__ nextOf,
                                       int N, int nx, double k, double gamma,
                                       double gz, double half, double Lbox,
                                       double invCell)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double4 pi = pos_m[i];
    double3 vi = vel[i];
    double  Ri = rad[i];

    int cx = (int)floor((pi.x + 0.5 * Lbox) * invCell);
    int cy = (int)floor((pi.y + 0.5 * Lbox) * invCell);
    int cz = (int)floor((pi.z + 0.5 * Lbox) * invCell);
    if (cx < 0) cx = 0; else if (cx >= nx) cx = nx - 1;
    if (cy < 0) cy = 0; else if (cy >= nx) cy = nx - 1;
    if (cz < 0) cz = 0; else if (cz >= nx) cz = nx - 1;

    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int dz = -1; dz <= 1; ++dz) {
        int nz = cz + dz; if (nz < 0 || nz >= nx) continue;   // clamp (no wrap)
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = cy + dy; if (ny < 0 || ny >= nx) continue;
            for (int dxi = -1; dxi <= 1; ++dxi) {
                int nxi = cx + dxi; if (nxi < 0 || nxi >= nx) continue;
                int c = nxi + nx * (ny + nx * nz);
                int j = cellHead[c];
                while (j >= 0) {
                    if (j != i) {
                        double4 pj = pos_m[j];
                        double dx = pi.x - pj.x;
                        double dy2 = pi.y - pj.y;
                        double dz2 = pi.z - pj.z;
                        double d2 = dx*dx + dy2*dy2 + dz2*dz2;
                        double sumR = Ri + rad[j];
                        if (d2 < sumR * sumR && d2 >= 1.0e-18) {
                            double d = sqrt(d2);
                            double inv = 1.0 / d;
                            double nxn = dx*inv, nyn = dy2*inv, nzn = dz2*inv;
                            double delta = sumR - d;
                            double3 vj = vel[j];
                            double vrx = vi.x - vj.x;
                            double vry = vi.y - vj.y;
                            double vrz = vi.z - vj.z;
                            double vn = vrx*nxn + vry*nyn + vrz*nzn;
                            double fmag = k * delta - gamma * vn;
                            fx += fmag * nxn;
                            fy += fmag * nyn;
                            fz += fmag * nzn;
                        }
                    }
                    j = nextOf[j];
                }
            }
        }
    }

    double3 pv = make_double3(pi.x, pi.y, pi.z);
    apply_walls(pv, vi, Ri, k, gamma, half, fx, fy, fz);

    double invm = 1.0 / pi.w;
    acc[i] = make_double3(fx * invm, fy * invm, fz * invm + gz);
}

// Velocity-Verlet: position half + position update.
// x(t+dt) = x(t) + v(t) dt + 0.5 a(t) dt^2
// v_half  = v(t) + 0.5 a(t) dt
__global__ void verlet_pos_kernel(double4* pos_m,
                                  double3* vel,
                                  const double3* acc,
                                  int N, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double4 p = pos_m[i];
    double3 v = vel[i];
    double3 a = acc[i];
    p.x += v.x * dt + 0.5 * a.x * dt * dt;
    p.y += v.y * dt + 0.5 * a.y * dt * dt;
    p.z += v.z * dt + 0.5 * a.z * dt * dt;
    v.x += 0.5 * a.x * dt;
    v.y += 0.5 * a.y * dt;
    v.z += 0.5 * a.z * dt;
    pos_m[i] = p;
    vel[i]   = v;
}

// Velocity-Verlet: velocity full step.
// v(t+dt) = v_half + 0.5 a(t+dt) dt
__global__ void verlet_vel_kernel(double3* vel,
                                  const double3* acc_new,
                                  int N, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double3 v = vel[i];
    double3 a = acc_new[i];
    v.x += 0.5 * a.x * dt;
    v.y += 0.5 * a.y * dt;
    v.z += 0.5 * a.z * dt;
    vel[i] = v;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

struct Config {
    std::string input  = "";
    std::string outDir = "out";
    std::string outPrefix = "particles";
    int    nSteps   = 1000;
    int    dumpEvery = 100;
    double dt       = 0.001;
    double sigma    = 1.0;
    double eps      = 1.0;
    double rcut     = 0.0; // 0 => no cutoff
    int    block    = 128;
    double Lbox     = 0.0; // 0 => open domain (no PBC)
    std::string algo = "naive"; // naive | cells
    // --- Exercise 3 (DEM) ---
    std::string phys = "lj";    // lj | dem
    double kspring  = 1000.0;   // contact spring stiffness
    double gamma    = 5.0;      // contact damping coefficient
    double grav     = 0.0;      // gravity magnitude (applied as -z)
};

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s -i input.in [options]\n"
        "  -i  PATH       input particle file (required)\n"
        "  -o  DIR        output directory (default: out)\n"
        "  -p  PREFIX     output file prefix (default: particles)\n"
        "  -n  STEPS      number of time steps (default: 1000)\n"
        "  -d  EVERY      dump every N steps (default: 100; 0 = no dumps)\n"
        "  -t  DT         time step (default: 0.001)\n"
        "  -s  SIGMA      Lennard-Jones sigma (default: 1.0)\n"
        "  -e  EPS        Lennard-Jones epsilon (default: 1.0)\n"
        "  -c  RCUT       force cutoff radius (default: 0 = none)\n"
        "  -L  LBOX       cubic box edge length, enables PBC\n"
        "                 (default: 0 = open / infinite domain)\n"
        "  -A  ALGO       force algorithm: naive | cells (default: naive)\n"
        "                 (cells requires -L > 0 and -c > 0)\n"
        "  -b  BLOCK      CUDA block size (default: 128)\n"
        "  --- Exercise 3 (granular / DEM) ---\n"
        "  -P  PHYS       physics: lj | dem (default: lj)\n"
        "                 dem = spring-dashpot granular spheres\n"
        "  -k  KSPRING    contact spring stiffness (dem, default: 1000)\n"
        "  -y  GAMMA      contact damping coefficient (dem, default: 5)\n"
        "  -g  GRAV       gravity magnitude, applied as (0,0,-g) (default: 0)\n"
        "                 (with -P dem, -L LBOX defines fixed box walls)\n"
        "  -h             this help\n",
        prog);
}

static bool parseArgs(int argc, char** argv, Config& c) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* opt) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", opt);
                std::exit(1);
            }
            return std::string(argv[++i]);
        };
        if      (a == "-i") c.input = need("-i");
        else if (a == "-o") c.outDir = need("-o");
        else if (a == "-p") c.outPrefix = need("-p");
        else if (a == "-n") c.nSteps = std::stoi(need("-n"));
        else if (a == "-d") c.dumpEvery = std::stoi(need("-d"));
        else if (a == "-t") c.dt = std::stod(need("-t"));
        else if (a == "-s") c.sigma = std::stod(need("-s"));
        else if (a == "-e") c.eps = std::stod(need("-e"));
        else if (a == "-c") c.rcut = std::stod(need("-c"));
        else if (a == "-L") c.Lbox = std::stod(need("-L"));
        else if (a == "-A") c.algo = need("-A");
        else if (a == "-P") c.phys = need("-P");
        else if (a == "-k") c.kspring = std::stod(need("-k"));
        else if (a == "-y") c.gamma = std::stod(need("-y"));
        else if (a == "-g") c.grav = std::stod(need("-g"));
        else if (a == "-b") c.block = std::stoi(need("-b"));
        else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
        else { fprintf(stderr, "Unknown arg: %s\n", a.c_str()); return false; }
    }
    if (c.input.empty()) { usage(argv[0]); return false; }
    if (c.algo != "naive" && c.algo != "cells") {
        fprintf(stderr, "Unknown algorithm '%s' (use naive|cells)\n",
                c.algo.c_str());
        return false;
    }
    if (c.phys != "lj" && c.phys != "dem") {
        fprintf(stderr, "Unknown physics '%s' (use lj|dem)\n", c.phys.c_str());
        return false;
    }
    if (c.algo == "cells") {
        // LJ cells need an explicit cutoff; DEM derives its range from the
        // particle radii (2*Rmax), so only the box is required there.
        if (c.Lbox <= 0.0) {
            fprintf(stderr, "Algorithm 'cells' requires -L > 0.\n");
            return false;
        }
        if (c.phys == "lj" && c.rcut <= 0.0) {
            fprintf(stderr, "Algorithm 'cells' with lj requires -c > 0.\n");
            return false;
        }
    }
    return true;
}

// Load particle file.
//   line1 = N
//   LJ  format (hasRadius == false):  m   x y z  vx vy vz
//   DEM format (hasRadius == true) :  m r x y z  vx vy vz
static int loadInput(const std::string& path,
                     std::vector<double4>& pos_m,
                     std::vector<double3>& vel,
                     std::vector<double>&  rad,
                     bool hasRadius)
{
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); std::exit(1); }
    int N;
    f >> N;
    if (!f || N <= 0) {
        fprintf(stderr, "Bad input: expected positive N on first line\n");
        std::exit(1);
    }
    pos_m.resize(N);
    vel.resize(N);
    rad.resize(N);
    for (int i = 0; i < N; ++i) {
        double m, r = 0.5, x, y, z, vx, vy, vz;
        if (hasRadius) f >> m >> r >> x >> y >> z >> vx >> vy >> vz;
        else           f >> m >>      x >> y >> z >> vx >> vy >> vz;
        if (!f) {
            fprintf(stderr, "Bad input at particle %d\n", i);
            std::exit(1);
        }
        pos_m[i] = make_double4(x, y, z, m);
        vel[i]   = make_double3(vx, vy, vz);
        rad[i]   = r;
    }
    return N;
}

static void writeVTK(const std::string& path,
                     const std::vector<double4>& pos_m,
                     const std::vector<double3>& vel,
                     const std::vector<double>&  rad)
{
    std::ofstream f(path);
    if (!f) { fprintf(stderr, "Cannot write %s\n", path.c_str()); std::exit(1); }
    int N = (int)pos_m.size();
    f << "# vtk DataFile Version 4.0\n";
    f << "MD GPU simulation output\n";
    f << "ASCII\n";
    f << "DATASET UNSTRUCTURED_GRID\n";
    f << "POINTS " << N << " double\n";
    f << std::fixed << std::setprecision(6);
    for (auto& p : pos_m) f << p.x << ' ' << p.y << ' ' << p.z << '\n';
    f << "CELLS 0 0\n";
    f << "CELL_TYPES 0\n";
    f << "POINT_DATA " << N << '\n';
    f << "SCALARS m double\nLOOKUP_TABLE default\n";
    for (auto& p : pos_m) f << p.w << '\n';
    f << "SCALARS radius double\nLOOKUP_TABLE default\n";
    for (auto& r : rad)   f << r << '\n';
    f << "VECTORS v double\n";
    for (auto& v : vel)   f << v.x << ' ' << v.y << ' ' << v.z << '\n';
}

static void mkdirp(const std::string& d) {
    std::string cmd = "mkdir -p '" + d + "'";
    int rc = std::system(cmd.c_str());
    (void)rc;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    Config cfg;
    if (!parseArgs(argc, argv, cfg)) return 1;

    std::vector<double4> h_pos;
    std::vector<double3> h_vel;
    std::vector<double>  h_rad;
    bool isDem = (cfg.phys == "dem");
    int N = loadInput(cfg.input, h_pos, h_vel, h_rad, isDem);

    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    // For DEM the contact range is 2*Rmax; the cell grid is sized to that.
    double Rmax = 0.0;
    for (double r : h_rad) Rmax = (r > Rmax) ? r : Rmax;
    double demCut = 2.0 * Rmax;

    std::cout << "=== MD-GPU ===\n"
              << "device      : " << prop.name << " (cc " << prop.major
              << '.' << prop.minor << ")\n"
              << "input       : " << cfg.input << "\n"
              << "physics     : " << cfg.phys << "\n"
              << "N           : " << N << "\n"
              << "steps       : " << cfg.nSteps << "\n"
              << "dt          : " << cfg.dt << "\n";
    if (isDem) {
        std::cout << "k spring    : " << cfg.kspring << "\n"
                  << "gamma damp  : " << cfg.gamma << "\n"
                  << "gravity     : (0,0," << -cfg.grav << ")\n"
                  << "box walls   : " << (cfg.Lbox > 0 ? "on" : "off")
                  << "  Lbox=" << cfg.Lbox << "\n"
                  << "max radius  : " << Rmax << "\n";
    } else {
        std::cout << "sigma       : " << cfg.sigma << "\n"
                  << "epsilon     : " << cfg.eps << "\n"
                  << "rcut        : " << cfg.rcut << " (0 = none)\n"
                  << "Lbox        : " << cfg.Lbox << " (0 = open / no PBC)\n";
    }
    std::cout << "algo        : " << cfg.algo << "\n"
              << "dump every  : " << cfg.dumpEvery << "\n"
              << "block size  : " << cfg.block << "\n"
              << "out dir     : " << cfg.outDir << "\n"
              << std::flush;

    if (cfg.dumpEvery > 0) mkdirp(cfg.outDir);

    double4 *d_pos;
    double3 *d_vel, *d_acc, *d_acc_new;
    double  *d_rad;
    CUDA_CHECK(cudaMalloc(&d_pos,     N * sizeof(double4)));
    CUDA_CHECK(cudaMalloc(&d_vel,     N * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&d_acc,     N * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&d_acc_new, N * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&d_rad,     N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), N * sizeof(double4),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel.data(), N * sizeof(double3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rad, h_rad.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_acc,     0, N * sizeof(double3)));
    CUDA_CHECK(cudaMemset(d_acc_new, 0, N * sizeof(double3)));

    int blk = cfg.block;
    int grid = (N + blk - 1) / blk;
    size_t shmem = blk * sizeof(double4);
    double rcut2 = cfg.rcut > 0.0 ? cfg.rcut * cfg.rcut : 0.0;
    double Lbox = cfg.Lbox;
    double invL = (Lbox > 0.0) ? 1.0 / Lbox : 0.0;
    bool   pbc  = (Lbox > 0.0) && !isDem;   // DEM uses walls, not PBC
    bool   useWalls = isDem && (Lbox > 0.0);
    double half = 0.5 * Lbox;
    double gz   = -cfg.grav;                // gravity acceleration in z
    bool   useCells = (cfg.algo == "cells");

    // ----- Cell-list setup (only if algo == cells) -----
    int nx = 0;
    int nCells = 0;
    double cellSize = 0.0;
    double invCell = 0.0;
    int *d_cellHead = nullptr;
    int *d_nextOf   = nullptr;
    if (useCells) {
        // Cell edge must be >= the interaction range. For LJ that is rcut;
        // for DEM it is the maximum contact distance 2*Rmax.
        double cut = isDem ? demCut : cfg.rcut;
        if (Lbox <= 0.0 || cut <= 0.0) {
            fprintf(stderr,
                "ERROR: 'cells' requires a box (-L > 0) and a positive "
                "interaction range (rcut for lj, 2*Rmax for dem).\n");
            std::exit(2);
        }
        nx = (int)floor(Lbox / cut);
        if (nx < 3) {
            fprintf(stderr,
                "ERROR: 'cells' needs Lbox >= 3*range "
                "(Lbox=%.4f, range=%.4f, nx=%d)\n", Lbox, cut, nx);
            std::exit(2);
        }
        cellSize = Lbox / nx;
        invCell  = 1.0 / cellSize;
        nCells   = nx * nx * nx;
        CUDA_CHECK(cudaMalloc(&d_cellHead, nCells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_nextOf,   N      * sizeof(int)));
        std::cout << "cells       : " << nx << "^3 = " << nCells
                  << "  cellSize=" << cellSize
                  << "  (range=" << cut << ")\n"
                  << std::flush;
    }

    // Helper: rebuild cell linked lists.
    auto rebuildCells = [&]() {
        int gridC = (nCells + blk - 1) / blk;
        bin_reset_kernel<<<gridC, blk>>>(d_cellHead, nCells);
        bin_build_kernel<<<grid,  blk>>>(d_pos, d_cellHead, d_nextOf, N,
                                          nx, Lbox, invCell);
    };

    // Helper: launch the chosen force kernel into a target acc buffer.
    auto launchForces = [&](double3* d_target) {
        if (isDem) {
            if (useCells) {
                rebuildCells();
                dem_accel_cells_kernel<<<grid, blk>>>(
                    d_pos, d_vel, d_rad, d_target, d_cellHead, d_nextOf,
                    N, nx, cfg.kspring, cfg.gamma, gz, half, Lbox, invCell);
            } else {
                dem_accel_kernel<<<grid, blk>>>(
                    d_pos, d_vel, d_rad, d_target,
                    N, cfg.kspring, cfg.gamma, gz, half, useWalls ? 1 : 0);
            }
        } else if (useCells) {
            rebuildCells();
            lj_accel_cells_kernel<<<grid, blk>>>(
                d_pos, d_target, d_cellHead, d_nextOf,
                N, nx, cfg.sigma, cfg.eps, rcut2, Lbox, invL, invCell);
        } else {
            lj_accel_kernel<<<grid, blk, shmem>>>(
                d_pos, d_target, N, cfg.sigma, cfg.eps, rcut2, Lbox, invL);
        }
    };

    auto dump = [&](int step) {
        if (cfg.dumpEvery <= 0) return;
        CUDA_CHECK(cudaMemcpy(h_pos.data(), d_pos, N * sizeof(double4),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vel.data(), d_vel, N * sizeof(double3),
                              cudaMemcpyDeviceToHost));
        std::ostringstream name;
        name << cfg.outDir << '/' << cfg.outPrefix << '_'
             << std::setw(6) << std::setfill('0') << step << ".vtk";
        writeVTK(name.str(), h_pos, h_vel, h_rad);
    };

    // Wrap initial positions if PBC active (in case input lies outside box).
    if (pbc) {
        wrap_pos_kernel<<<grid, blk>>>(d_pos, N, Lbox, invL);
    }

    // initial acceleration for step 0
    launchForces(d_acc);
    CUDA_CHECK(cudaGetLastError());

    dump(0);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= cfg.nSteps; ++step) {
        // x(t+dt), v(t+dt/2)
        verlet_pos_kernel<<<grid, blk>>>(d_pos, d_vel, d_acc, N, cfg.dt);
        // wrap into periodic box
        if (pbc) {
            wrap_pos_kernel<<<grid, blk>>>(d_pos, N, Lbox, invL);
        }
        // a(t+dt)
        launchForces(d_acc_new);
        // v(t+dt)
        verlet_vel_kernel<<<grid, blk>>>(d_vel, d_acc_new, N, cfg.dt);

        std::swap(d_acc, d_acc_new);

        if (cfg.dumpEvery > 0 && step % cfg.dumpEvery == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            dump(step);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    double mspstep = 1000.0 * secs / cfg.nSteps;
    std::cout << "\nTotal sim time     : " << secs << " s\n"
              << "Mean ms / step     : " << mspstep << "\n"
              << "Steps              : " << cfg.nSteps << "\n"
              << "Particles          : " << N << "\n"
              << "Pair-int / s (eff) : "
              << (double)N * N * cfg.nSteps / secs << "\n";

    cudaFree(d_pos); cudaFree(d_vel);
    cudaFree(d_acc); cudaFree(d_acc_new);
    cudaFree(d_rad);
    if (d_cellHead) cudaFree(d_cellHead);
    if (d_nextOf)   cudaFree(d_nextOf);
    return 0;
}
