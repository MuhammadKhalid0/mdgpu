#!/bin/bash -l
# Exercise 3: build + run all DEM (granular) scenarios.
# Submit with:   sbatch scripts/run_dem.sh
#
#SBATCH --job-name=md-dem
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=md-dem-%j.out

set -euo pipefail
module load cuda/12.6.2

cd "$HOME/md-gpu"

echo "=== regenerate inputs ==="
python3 scripts/generate_inputs.py

echo "=== build ==="
make -j release

echo "=== verification: single particle drop (energy decay) ==="
# High dump frequency so we can plot kinetic energy vs time.
./md -i inputs/dem_drop.in -P dem -A naive -L 24 -g 9.81 -k 2000 -y 8 \
     -n 8000 -d 20 -t 0.0008 -o out/dem_drop -p p

echo "=== granular scenarios ==="
make dem

echo "=== done ==="
for s in dem_drop dem_pile dem_collapse dem_mix; do
    echo "$s : $(ls out/$s 2>/dev/null | wc -l) VTK files"
done
