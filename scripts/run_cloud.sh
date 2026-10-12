#!/bin/bash -l
# Re-run cloud scenario only.
# Submit with:   sbatch scripts/run_cloud.sh
#
#SBATCH --job-name=md-cloud
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=md-cloud-%j.out

set -euo pipefail
module load cuda/12.6.2

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== removing old cloud output ==="
rm -rf out/cloud

echo "=== running cloud scenario ==="
./md -i inputs/cloud.in -n 3000 -d 50 -t 0.002 -o out/cloud -p p

echo "=== done — $(ls out/cloud | wc -l) VTK files written ==="
