#!/bin/bash -l
# Part 2 benchmark: baseline vs cutoff vs cell-list.
# Submit with:   sbatch scripts/run_bench_p2.sh
#
#SBATCH --job-name=md-bench-p2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=00:45:00
#SBATCH --output=md-bench-p2-%j.out

set -euo pipefail
module load cuda/12.6.2

cd "$HOME/md-gpu"

echo "=== regenerate inputs (with .box files) ==="
python3 scripts/generate_inputs.py

echo "=== rebuild ==="
make -j release

echo "=== part-2 benchmark (3 modes) ==="
make bench_p2 | tee bench_p2_results.csv

echo "=== done ==="
cat bench_p2_results.csv
