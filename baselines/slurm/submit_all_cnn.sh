#!/bin/bash
# submit_all_cnn.sh -- submit the CNN-base baseline sweep (CNN1D_extended base):
#   Truncation-CNN (100% = Baseline-CNN), NonMyopic-CNN, TEASER-CNN.
# Each is a 6-task array (one dataset per task, all 5 seeds inside). aggregate_cnn.py
# runs after all three succeed and writes results_cnn/CNN_BASELINES_multiseed.csv.
#
# Env: defaults to $TSLEARN_ENV from _env.sh (needs torch + tslearn + sktime). Override with
#   CNN_ENV=<env> bash baselines/slurm/submit_all_cnn.sh
set -euo pipefail
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"   # baselines/
# Output dir: default results_cnn/, or set CNN_OUTDIR=/some/dir
export CNN_OUTDIR="${CNN_OUTDIR:-$(pwd)/results_cnn}"
mkdir -p logs "$CNN_OUTDIR"
echo "CNN_OUTDIR=$CNN_OUTDIR"

TR=$(sbatch --parsable slurm/slurm_cnn_truncation.slurm)
NM=$(sbatch --parsable slurm/slurm_cnn_nonmyopic.slurm)
TE=$(sbatch --parsable slurm/slurm_cnn_teaser.slurm)
BR=$(sbatch --parsable slurm/slurm_cnn_branchynet.slurm)
echo "submitted: truncation-cnn=$TR  nonmyopic-cnn=$NM  teaser-cnn=$TE  branchynet-cnn=$BR"

AGG=$(sbatch --parsable --dependency="afterok:${TR}:${NM}:${TE}:${BR}" \
      --job-name=cnn_agg --output=logs/cnn_agg_%j.out --time=00:10:00 --mem=8G \
      --export=ALL,CNN_OUTDIR="$CNN_OUTDIR" --wrap \
      "source ./slurm/_env.sh; activate_env \"\${CNN_ENV:-\$TSLEARN_ENV}\"; python aggregate_cnn.py --outdir \"\$CNN_OUTDIR\"; python make_cnn_paper_table.py --outdir \"\$CNN_OUTDIR\"")
echo "submitted: aggregate=$AGG  (runs after ${TR}, ${NM}, ${TE}, ${BR} all succeed)"
echo
echo "watch:   squeue -u \$USER"
echo "results: $CNN_OUTDIR/  (CNN_BASELINES_multiseed.csv, BN_ACCURACY_TABLE.csv, CNN_PAPER_TABLE.tex)"
