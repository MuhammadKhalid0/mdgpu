#!/bin/bash -l
#SBATCH --job-name=md-build
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=md-build-%j.out

set -euo pipefail
module load cuda/12.6.2

cd "$HOME/md-gpu"

echo "=== regenerate inputs ==="
python3 scripts/generate_inputs.py

echo "=== build ==="
make clean && make -j release

echo "=== sanity tests (Part 1 still works) ==="
make tests

echo "=== Part 2 sanity: cells vs naive same lattice ==="
echo "--- naive O(N^2), no PBC ---"
./md -i inputs/bench_001000.in -n 100 -d 0 -t 0.001
echo "--- naive O(N^2), PBC + cutoff ---"
./md -i inputs/bench_001000.in -n 100 -d 0 -t 0.001 \
     -L 11.225 -c 2.5 -A naive
echo "--- cells, PBC + cutoff ---"
./md -i inputs/bench_001000.in -n 100 -d 0 -t 0.001 \
     -L 11.225 -c 2.5 -A cells

echo "=== done ==="
