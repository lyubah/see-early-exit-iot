#!/bin/bash
# _env.sh -- shared environment setup sourced by every SLURM script.
# Set the conda env names below (or export EDEN_ENV / TSLEARN_ENV / ... before
# submitting) to match the envs on your cluster.
#
# Requirements per env:
#   * NonMyopic (corrected_tslearn) -> tslearn + scikit-learn + numpy + pandas
#   * TEASER    (run_teaser_corrected) -> sktime + scikit-learn + numpy + pandas
#   * EDEN      (run_eden_baseline) -> eden + bigtree + scikit-learn
#   * Truncation(Partial_sensing_data) -> scikit-learn + numpy + pandas

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true

# Tested with:
#   eden_env      -> sklearn 1.8, eden 1.0, bigtree 1.4.1, numpy 2.4      (EDEN, Truncation)
#   teaser_legacy -> tslearn 0.6.4, sklearn 1.6.1, sktime 0.38.5, numpy 1.26 (NonMyopic, TEASER)
EDEN_ENV="${EDEN_ENV:-eden_env}"
TRUNC_ENV="${TRUNC_ENV:-eden_env}"
TSLEARN_ENV="${TSLEARN_ENV:-teaser_legacy}"
TEASER_ENV="${TEASER_ENV:-teaser_legacy}"

activate_env () {
    local env="$1"
    source activate "$env" 2>/dev/null || conda activate "$env" 2>/dev/null || {
        echo "ERROR: could not activate conda env '$env'" >&2; exit 2; }
    echo "[env] activated '$env' | python=$(which python)" >&2
}

# Keep BLAS/OpenMP from oversubscribing the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
