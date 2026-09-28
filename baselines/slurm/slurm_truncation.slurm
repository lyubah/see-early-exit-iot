#!/bin/bash
#SBATCH --job-name=trunc_multiseed
#SBATCH --output=logs/truncation_%j.out
#SBATCH --error=logs/truncation_%j.out
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#
# Static Truncation RF (Table VI) -- single job, all six datasets x 5 seeds x P% sweep.

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
source ./slurm/_env.sh
activate_env "$TRUNC_ENV"

echo "[truncation] host=$(hostname) start=$(date)"
python run_truncation_multiseed.py --seeds 42 43 44 45 46; rc=$?
echo "[truncation] done=$(date) rc=$rc"
exit $rc
