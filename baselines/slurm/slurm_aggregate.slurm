#!/bin/bash
#SBATCH --job-name=baselines_aggregate
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.out
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#
# Final step: collect every *_perseed__*.csv into the paper-ready
# <METHOD>_multiseed.csv files + BASELINES_multiseed.{csv,tex}.
# Submitted with an afterok dependency on the three baseline jobs by submit_all.sh,
# but also safe to run by hand on the login node once the jobs finish:
#     python aggregate.py

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
source ./slurm/_env.sh
activate_env "$EDEN_ENV"

python aggregate.py
