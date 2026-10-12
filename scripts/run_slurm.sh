#!/bin/bash -l
# Run md-gpu test suite + benchmark on a TinyGPU node.
# Submit with:   sbatch scripts/run_slurm.sh
#
#SBATCH --job-name=md-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=md-gpu-%j.out

set -euo pipefail
module load cuda/12.6.2

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== nvidia-smi ==="
nvidia-smi

echo "=== build ==="
make -j release

echo "=== generate inputs ==="
make inputs

echo "=== sanity tests (2 particles) ==="
make tests

echo "=== scenarios ==="
make scenarios

echo "=== benchmark ==="
make bench | tee bench_results.csv
