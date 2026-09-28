#!/usr/bin/env python3
"""
run_truncation_multiseed.py
===========================
Multi-seed static-truncation RF baseline: classify from only the first P% of the
window, with no early exit.

For each data percentage P in {10,20,30,40,50,100}: train AND test a fresh RF
(50 trees, depth 30) on the first P% of the REAL sensor timesteps -- triaxial-safe
via bd.truncate_flatten (reshape to (N, T, features) first, keep first P% of T).
savings = (1 - P/100)*100 (deterministic; std 0). It's a SWEEP, not one cell.

SPLIT: to keep all methods comparable on a MATCHED split, every (dataset, seed)
uses the shared 60/40 split from mseed_common.load_split, and the RF random_state =
seed too, so BOTH the split and the RF vary per seed.

Outputs (results/):
  TRUNCATION_perseed__<dataset>.csv   dataset, seed, pct + weighted metrics + savings
"""
import argparse
import numpy as np

import mseed_common as mc
import baseline_data as bd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1-1 BY CONSTRUCTION: the ablation RF reads the canonical Baseline-RF/GB tree size
# from the shared module (mseed_common) instead of hardcoding it, so it can never drift
# from the methods it ablates. Verified against EDEN's reference RF when eden is importable.
N_ESTIMATORS = mc.BASELINE_RF_N_ESTIMATORS      # == Baseline-RF / GB / EDEN reference RF
MAX_DEPTH = mc.BASELINE_RF_MAX_DEPTH
try:
    from run_eden_baseline import N_ESTIMATORS as _EDEN_N, MAX_DEPTH as _EDEN_D
    assert (N_ESTIMATORS, MAX_DEPTH) == (_EDEN_N, _EDEN_D), (
        f"truncation RF ({N_ESTIMATORS},{MAX_DEPTH}) != canonical Baseline-RF "
        f"({_EDEN_N},{_EDEN_D}); the ablation would no longer be 1-1.")
except ImportError:
    pass   # eden not installed (lightweight local run) -> mc values are authoritative


def _split_6040(name, seed):
    # shared 60/40 split by window (same for every baseline and dataset)
    X, y, tr, te = mc.load_split(name, seed)
    return X[tr], y[tr], X[te], y[te]


def run_dataset(dataset, seeds):
    ds = bd.canonical_name(dataset)
    print(f"\n=== Static Truncation  {ds}  pcts={mc.TRUNC_PERCENTAGES} ===", flush=True)
    rows = []
    for seed in seeds:
        Xtr_raw, ytr, Xte_raw, yte = _split_6040(ds, seed)
        for pct in mc.TRUNC_PERCENTAGES:
            Xtr, cut = bd.truncate_flatten(Xtr_raw, ds, pct)   # triaxial-safe, real time
            Xte, _ = bd.truncate_flatten(Xte_raw, ds, pct)
            model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                           random_state=seed, n_jobs=-1).fit(Xtr, ytr)
            yp = model.predict(Xte)
            acc = accuracy_score(yte, yp)
            m = bd.prf(yte, yp)                                 # weighted P/R/F1 (0-1)
            savings = (1.0 - pct / 100.0) * 100.0
            rows.append(mc.make_trunc_row(
                ds, seed, pct, acc, m["precision_weighted"], m["recall_weighted"],
                m["f1_weighted"], savings))
        accs = {r["pct"]: r["accuracy"] for r in rows if r["seed"] == seed}
        print(f"  seed={seed}  " + "  ".join(f"{p}%:{accs[p]:.1f}" for p in mc.TRUNC_PERCENTAGES)
              + f"  (n_test={len(yte)})", flush=True)

    mc.write_trunc_perseed(rows, dataset=ds)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Multi-seed static-truncation RF baseline")
    ap.add_argument("--datasets", nargs="*", default=mc.DATASET_ORDER)
    ap.add_argument("--dataset", default=None,
                    help="single dataset (SLURM-array convenience; overrides --datasets)")
    ap.add_argument("--seeds", nargs="*", type=int, default=mc.SEEDS)
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else args.datasets
    print(f"Static-truncation multi-seed | seeds={args.seeds} | datasets={datasets}", flush=True)
    for ds in datasets:
        run_dataset(ds, args.seeds)
    print("\nDONE. Per-dataset TRUNCATION_perseed__*.csv written to results/.")


if __name__ == "__main__":
    main()
