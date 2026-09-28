#!/bin/bash
#SBATCH --job-name=eden_multiseed
#SBATCH --output=logs/eden_%j.out
#SBATCH --error=logs/eden_%j.out
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#
# Adaptive RF (EDEN) -- single job, all six datasets x 5 seeds.

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
source ./slurm/_env.sh
activate_env "$EDEN_ENV"

echo "[eden] host=$(hostname) start=$(date)"
python run_eden_multiseed.py --seeds 42 43 44 45 46; rc=$?
echo "[eden] done=$(date) rc=$rc"
exit $rc
