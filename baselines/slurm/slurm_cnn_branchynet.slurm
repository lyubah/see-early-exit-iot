#!/bin/bash
#SBATCH --job-name=cnn_branchy
#SBATCH --output=logs/cnn_branchy_%A_%a.out
#SBATCH --error=logs/cnn_branchy_%A_%a.out
#SBATCH --array=0-5
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#
# BranchyNet-CNN (CNN1D_extended_EENN, iso-accuracy operating point), one array task
# per dataset (all 5 seeds). Trains ONE multi-exit net per seed, then sweeps entropy
# thresholds at inference -> COMPUTE-savings (see run_branchynet_cnn_multiseed.py AXIS NOTE).

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
source ./slurm/_env.sh
activate_env "${CNN_ENV:-$TSLEARN_ENV}"
export CNN_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -c "import torch; print('[torch]', torch.__version__)" || { echo "ERROR: torch missing in env"; exit 3; }

DATASETS=(Epilepsy PAMAP2 Shoaib WESADchest EMGPhysical SelfRegulationSCP1)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[cnn_branchy] task=$SLURM_ARRAY_TASK_ID dataset=$DS host=$(hostname) start=$(date)"

python run_branchynet_cnn_multiseed.py --dataset "$DS" --seeds 42 43 44 45 46 --epochs 20 --outdir "${CNN_OUTDIR:-$(pwd)/results_cnn}"; rc=$?
echo "[cnn_branchy] dataset=$DS done=$(date) rc=$rc"
exit $rc
