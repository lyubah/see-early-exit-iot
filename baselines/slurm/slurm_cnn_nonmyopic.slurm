#!/bin/bash
#SBATCH --job-name=cnn_nm
#SBATCH --output=logs/cnn_nm_%A_%a.out
#SBATCH --error=logs/cnn_nm_%A_%a.out
#SBATCH --array=0-5
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=16
#
# NonMyopic-CNN, one array task per dataset (all 5 seeds). HEAVIEST baseline: tslearn
# fits one CNN per timestamp (range(min_t, sz+1)), so ~T CNN trainings per seed
# (SCP T=896, PAMAP2 T=512 are the long tails). 48h wall + all cores per CNN.

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
source ./slurm/_env.sh
activate_env "${CNN_ENV:-$TSLEARN_ENV}"
export CNN_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -c "import torch, tslearn; print('[torch]', torch.__version__, '[tslearn]', tslearn.__version__)" \
    || { echo "ERROR: torch/tslearn missing in env"; exit 3; }

DATASETS=(Epilepsy PAMAP2 Shoaib WESADchest EMGPhysical SelfRegulationSCP1)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "[cnn_nm] task=$SLURM_ARRAY_TASK_ID dataset=$DS host=$(hostname) start=$(date)"

python run_nonmyopic_cnn_multiseed.py --dataset "$DS" --seeds 42 43 44 45 46 --epochs 20 --outdir "${CNN_OUTDIR:-$(pwd)/results_cnn}"; rc=$?
echo "[cnn_nm] dataset=$DS done=$(date) rc=$rc"
exit $rc
