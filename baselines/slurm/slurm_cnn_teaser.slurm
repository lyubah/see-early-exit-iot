#!/bin/bash
#SBATCH --job-name=cnn_teaser
#SBATCH --output=logs/cnn_teaser_%A_%a.out
#SBATCH --error=logs/cnn_teaser_%A_%a.out
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#
# TEASER-CNN, one array task per dataset (all 5 seeds). sktime TEASER fits a CNN slave
# per checkpoint (25/50/75%) + an OCSVM master per seed. Moderate-heavy.

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
source ./slurm/_env.sh
activate_env "${CNN_ENV:-$TSLEARN_ENV}"          # teaser_legacy: torch + sktime 0.38.5
export CNN_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -c "import torch, sktime; print('[torch]', torch.__version__, '[sktime]', sktime.__version__)" \
    || { echo "ERROR: torch/sktime missing in env"; exit 3; }

DATASETS=(Epilepsy PAMAP2 Shoaib WESADchest EMGPhysical SelfRegulationSCP1)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[cnn_teaser] task=$SLURM_ARRAY_TASK_ID dataset=$DS host=$(hostname) start=$(date)"

python run_teaser_cnn_multiseed.py --dataset "$DS" --seeds 42 43 44 45 46 --epochs 20 --outdir "${CNN_OUTDIR:-$(pwd)/results_cnn}"; rc=$?
echo "[cnn_teaser] dataset=$DS done=$(date) rc=$rc"
exit $rc
