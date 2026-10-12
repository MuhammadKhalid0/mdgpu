#!/bin/bash -l
# Re-run extended benchmark with larger N.
# Submit with:   sbatch scripts/run_bench.sh
#
#SBATCH --job-name=md-bench
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=md-bench-%j.out

set -euo pipefail
module load cuda/12.6.2

# Always run from the md-gpu project root
cd "$HOME/md-gpu"

echo "=== regenerate inputs ==="
python3 scripts/generate_inputs.py

echo "=== build (in case nothing built) ==="
make -j release

echo "=== extended benchmark ==="
make bench | tee bench_results.csv

echo "=== done ==="
cat bench_results.csv
