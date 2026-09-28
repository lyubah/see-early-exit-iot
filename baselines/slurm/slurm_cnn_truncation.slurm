#!/bin/bash
#SBATCH --job-name=cnn_trunc
#SBATCH --output=logs/cnn_trunc_%A_%a.out
#SBATCH --error=logs/cnn_trunc_%A_%a.out
#SBATCH --array=0-5
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#
# Truncation-CNN sweep (100% = Baseline-CNN), one array task per dataset (all 5 seeds).
# CNN1D_extended base. Light-to-moderate: 6 pcts x 5 seeds fresh CNN fits per dataset.

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
source ./slurm/_env.sh
activate_env "${CNN_ENV:-$TSLEARN_ENV}"          # teaser_legacy has torch + tslearn + sktime
export CNN_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -c "import torch; print('[torch]', torch.__version__)" || { echo "ERROR: torch missing in env"; exit 3; }

DATASETS=(Epilepsy PAMAP2 Shoaib WESADchest EMGPhysical SelfRegulationSCP1)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[cnn_trunc] task=$SLURM_ARRAY_TASK_ID dataset=$DS host=$(hostname) start=$(date)"

python run_baseline_cnn_multiseed.py --dataset "$DS" --seeds 42 43 44 45 46 --epochs 20 --outdir "${CNN_OUTDIR:-$(pwd)/results_cnn}"; rc=$?
echo "[cnn_trunc] dataset=$DS done=$(date) rc=$rc"
exit $rc
