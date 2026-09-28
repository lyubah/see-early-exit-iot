#!/bin/bash
#SBATCH --job-name=nm_multiseed
#SBATCH --output=logs/nonmyopic_%A_%a.out
#SBATCH --error=logs/nonmyopic_%A_%a.out
#SBATCH --array=0-5
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#
# NonMyopic (tslearn) -- one SLURM-array task PER dataset (6 tasks), each runs all
# 5 seeds. This is the heavy baseline (per-prefix RF bank); PAMAP2 (T=512) and SCP
# (T=896) are the long ones, so the array shares a generous wall time.

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
source ./slurm/_env.sh
activate_env "$TSLEARN_ENV"

DATASETS=(Epilepsy PAMAP2 Shoaib WESADchest EMGPhysical SelfRegulationSCP1)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[nonmyopic] task=$SLURM_ARRAY_TASK_ID dataset=$DS host=$(hostname) start=$(date)"

python run_nonmyopic_multiseed.py --dataset "$DS" --seeds 42 43 44 45 46; rc=$?
echo "[nonmyopic] dataset=$DS done=$(date) rc=$rc"
exit $rc
