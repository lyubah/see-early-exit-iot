#!/usr/bin/env python3
"""
run_teaser_multiseed.py
=======================
Multi-seed TEASER baseline for the paper table. Same corrected model as
run_teaser_corrected.py -- plain RandomForest slave (50 trees) + OneClassSVM
master, checkpoints at 25/50/75%, on the CORRECTED triaxial time-major view
(bd.temporal_view, so checkpoints truncate REAL time, not the packed 3*T axis) --
looped over 5 seeds.

PER-SEED protocol (varies BOTH randomness sources):
  split random_state = seed   AND   slave-RF / analyzer random_state = seed.

Split: the shared 60/40 split from mseed_common.load_split -- the SAME matched split
EDEN and NonMyopic use, so the results are comparable.

Sensing savings = mean over TEST windows of window_earliness (= 1 - t_decision/T),
in percent -- TEASER's native early-exit sensing number.

Outputs (results/):
  TEASER_perseed__<dataset>.csv   canonical 7 cols (one per dataset; array-safe)
"""
import argparse
import numpy as np

import mseed_common as mc
import baseline_data as bd
from sklearn.model_selection import train_test_split
from run_teaser_benchmark import ComprehensiveTEASERAnalyzer

RF_N_ESTIMATORS = 50      # same slave-RF size the paper TEASER column used


def run_dataset(dataset, seeds):
    ds = bd.canonical_name(dataset)
    print(f"\n=== TEASER  {ds} ===", flush=True)
    perseed_rows = []
    for seed in seeds:
        # shared 60/40 split by window (same for every baseline and dataset)
        X, y, tr, te = mc.load_split(ds, seed)
        V = np.swapaxes(bd.temporal_view(X, ds), 1, 2)     # (N, T, F) -> (N, F, T)
        T = V.shape[2]
        ckpt = [T // 4, T // 2, 3 * T // 4]
        az = ComprehensiveTEASERAnalyzer(n_estimators=RF_N_ESTIMATORS,
                                         random_state=seed, base_estimator="rf")
        r = az.fit_and_analyze(V[tr], y[tr], V[te], y[te], ckpt, T)
        savings = float(np.asarray(r["window_earliness"], dtype=float).mean()) * 100.0
        perseed_rows.append(mc.make_row(
            ds, seed, float(r["accuracy"]),
            float(r["precision_weighted"]), float(r["recall_weighted"]),
            float(r["f1_weighted"]), savings))
        print(f"  seed={seed}  acc={r['accuracy']*100:5.1f}%  sav={savings:5.1f}%  "
              f"P_w={r['precision_weighted']:.3f}  R_w={r['recall_weighted']:.3f}  "
              f"F1_w={r['f1_weighted']:.3f}  (n_test={len(te)})", flush=True)

    mc.write_perseed("TEASER", perseed_rows, dataset=ds)
    return perseed_rows


def main():
    ap = argparse.ArgumentParser(description="Multi-seed TEASER baseline")
    ap.add_argument("--datasets", nargs="*", default=mc.DATASET_ORDER)
    ap.add_argument("--dataset", default=None,
                    help="single dataset (SLURM-array convenience; overrides --datasets)")
    ap.add_argument("--seeds", nargs="*", type=int, default=mc.SEEDS)
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else args.datasets
    print(f"TEASER multi-seed | seeds={args.seeds} | datasets={datasets}", flush=True)
    for ds in datasets:
        run_dataset(ds, args.seeds)
    print("\nDONE. Per-dataset TEASER_perseed__*.csv written to results/.")


if __name__ == "__main__":
    main()
