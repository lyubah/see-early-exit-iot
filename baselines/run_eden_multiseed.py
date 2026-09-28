#!/usr/bin/env python3
"""
run_eden_multiseed.py
=====================
Multi-seed Adaptive RF (EDEN) baseline for the paper table. Same model as
run_eden_baseline.py / run_eden_wesad_match.py -- RF (50 trees, depth 30) handed
to EDEN with 8->16->32-bit quantization fallback, headline = the operating point
with the FEWEST mean trees whose accuracy is within EPSILON_ACC of EDEN's
full-ensemble accuracy -- looped over 5 seeds, all six datasets.

PER-SEED protocol (varies BOTH randomness sources):
  split random_state = seed   AND   RF random_state = seed (set on the
  run_eden_baseline module global so eden_adaptive_eval picks it up).

EDEN reads the FULL sensor window (sensing savings = 0) and saves only COMPUTE by
stopping early in the tree ensemble. Its reported "savings" is TOTAL-energy savings
= (1-SENSING_SHARE)*compute_savings -- the SEE-RF-comparable axis (compute is only
~30% of the budget, so EDEN's headline cannot exceed ~30%). Raw compute_savings and
mean_trees are kept in the detail CSV for reference.

Split: the shared 60/40 split from mseed_common.load_split -- the SAME matched split
TEASER and NonMyopic use, so the results are comparable.

Outputs (results/):
  EDEN_perseed__<dataset>.csv   canonical 7 cols (savings = total_energy_savings)
  EDEN_detail__<dataset>.csv    mean_trees / total_savings / bits / full-ens acc
"""
import argparse
import csv
import numpy as np

import mseed_common as mc
import baseline_data as bd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import run_eden_baseline as reb
from run_eden_baseline import (eden_adaptive_eval, N_ESTIMATORS, MAX_DEPTH,
                               INPUT_BITS, EPSILON_ACC)


def _eval_one(name, seed):
    """One (dataset, seed) EDEN evaluation on the shared 60/40 split. Returns
    (headline_row, full_acc, bits_used)."""
    # shared 60/40 split by window (same for every baseline and dataset)
    X, y, tr, te = mc.load_split(name, seed)
    Xtr = bd.flatten_full(X[tr], name)
    Xte = bd.flatten_full(X[te], name)
    ytr, yte = y[tr], y[te]
    window_len = int(bd.temporal_view(X[te], name).shape[1])   # real T

    # float reference RF (== Baseline-RF accuracy on this split)
    ref = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                 random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    float_acc = accuracy_score(yte, ref.predict(Xte))

    bits_used, rows, full_acc = None, None, None
    for bits in (INPUT_BITS, 16, 32):
        rows, _fp, full_acc, _rec = eden_adaptive_eval(
            Xtr, ytr, Xte, yte, bits, name=name, test_idx=te, window_len=window_len)
        bits_used = bits
        if full_acc >= float_acc - EPSILON_ACC:
            break

    eligible = [r for r in rows if r["accuracy"] >= full_acc - EPSILON_ACC]
    headline = (min(eligible, key=lambda r: r["mean_trees_used"])
                if eligible else max(rows, key=lambda r: r["threshold"]))
    return headline, float_acc, full_acc, bits_used, len(te)


def run_dataset(dataset, seeds):
    ds = bd.canonical_name(dataset)
    print(f"\n=== EDEN (Adaptive RF)  {ds} ===", flush=True)
    perseed_rows, detail = [], []
    for seed in seeds:
        reb.RANDOM_STATE = seed                  # vary the RF random_state
        h, float_acc, full_acc, bits, n_te = _eval_one(ds, seed)
        # EDEN reads the FULL window (sensing savings = 0) and saves only the
        # compute slice. Report TOTAL-energy savings -- the SEE-RF-comparable axis
        # (= (1-SENSING_SHARE)*compute_savings) -- as the headline, NOT raw compute.
        compute_sav = float(h["compute_energy_savings"])
        total_sav = float(h["total_energy_savings"])
        perseed_rows.append(mc.make_row(
            ds, seed, float(h["accuracy"]),
            float(h["precision_weighted"]), float(h["recall_weighted"]),
            float(h["f1_weighted"]), total_sav))
        detail.append({
            "dataset": ds, "seed": seed, "accuracy_pct": round(h["accuracy"] * 100, 4),
            "compute_savings": round(compute_sav, 4),
            "total_energy_savings": round(total_sav, 4),
            "mean_trees_used": round(float(h["mean_trees_used"]), 4),
            "threshold": int(h["threshold"]), "input_bits": int(bits),
            "full_ensemble_acc": round(full_acc * 100, 4),
            "float_rf_acc": round(float_acc * 100, 4), "n_test": n_te,
        })
        print(f"  seed={seed}  acc={h['accuracy']*100:5.1f}%  total_E_sav={total_sav:5.1f}%  "
              f"(compute_sav={compute_sav:.1f}%)  mean_trees={h['mean_trees_used']:.1f}/{N_ESTIMATORS}"
              f"  bits={bits}  F1_w={h['f1_weighted']:.3f}", flush=True)
    reb.RANDOM_STATE = 42                          # restore module default
    mc.write_perseed("EDEN", perseed_rows, dataset=ds)
    _write_detail(ds, detail)
    return perseed_rows


def _write_detail(ds, detail):
    p = mc.results_dir() / f"EDEN_detail__{ds}.csv"
    cols = ["dataset", "seed", "accuracy_pct", "compute_savings",
            "total_energy_savings", "mean_trees_used", "threshold", "input_bits",
            "full_ensemble_acc", "float_rf_acc", "n_test"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in detail:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Multi-seed Adaptive RF (EDEN) baseline")
    ap.add_argument("--datasets", nargs="*", default=mc.DATASET_ORDER)
    ap.add_argument("--dataset", default=None,
                    help="single dataset (SLURM-array convenience; overrides --datasets)")
    ap.add_argument("--seeds", nargs="*", type=int, default=mc.SEEDS)
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else args.datasets
    print(f"EDEN multi-seed | seeds={args.seeds} | datasets={datasets}", flush=True)
    for ds in datasets:
        run_dataset(ds, args.seeds)
    print("\nDONE. Per-dataset EDEN_perseed__*.csv written to results/.")


if __name__ == "__main__":
    main()
