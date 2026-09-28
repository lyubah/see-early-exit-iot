#!/bin/bash
# submit_all.sh -- submit the RF multi-seed baseline jobs (Non-Myopic, TEASER, EDEN,
# static truncation) + the aggregation step, which waits for all of them to finish.
# Run from anywhere:  bash baselines/slurm/submit_all.sh
set -euo pipefail
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"   # baselines/
mkdir -p logs results

NM=$(sbatch --parsable slurm/slurm_nonmyopic.slurm)
TE=$(sbatch --parsable slurm/slurm_teaser.slurm)
ED=$(sbatch --parsable slurm/slurm_eden.slurm)
TR=$(sbatch --parsable slurm/slurm_truncation.slurm)
echo "submitted: nonmyopic(array)=$NM  teaser=$TE  eden=$ED  truncation=$TR"

AGG=$(sbatch --parsable --dependency="afterok:${NM}:${TE}:${ED}:${TR}" slurm/slurm_aggregate.slurm)
echo "submitted: aggregate=$AGG  (runs after ${NM}, ${TE}, ${ED}, ${TR} all succeed)"
echo
echo "watch with:  squeue -u \$USER"
echo "results land in: $(pwd)/results/  (BASELINES_multiseed.{csv,tex})"
