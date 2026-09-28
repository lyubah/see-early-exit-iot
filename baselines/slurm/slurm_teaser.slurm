#!/bin/bash
#SBATCH --job-name=teaser_multiseed
#SBATCH --output=logs/teaser_%j.out
#SBATCH --error=logs/teaser_%j.out
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#
# TEASER -- single job, all six datasets x 5 seeds (RF slave + OneClassSVM master).

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
source ./slurm/_env.sh
activate_env "$TEASER_ENV"

echo "[teaser] host=$(hostname) start=$(date)"
python run_teaser_multiseed.py --seeds 42 43 44 45 46; rc=$?
echo "[teaser] done=$(date) rc=$rc"
exit $rc
