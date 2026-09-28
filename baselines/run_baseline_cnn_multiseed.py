#!/usr/bin/env python3
"""
run_baseline_cnn_multiseed.py
=============================
CNN counterpart of run_truncation_multiseed.py: static-truncation sweep with the
CNN1D_extended base (cnn_base.SklearnCNN) instead of RandomForest. The P=100% cell IS
Baseline-CNN (full window, no early exit) -- the CNN accuracy ceiling.

MATCHED to the RF truncation cell-for-cell:
  * split      = mseed_common.load_split (single 60/40, no stratify, rs=seed)
  * cut        = bd.truncate_flatten's timestep count (triaxial-safe REAL time)
  * pcts       = mseed_common.TRUNC_PERCENTAGES {10,20,30,40,50,100}
  * savings    = (1 - pct/100)*100  (deterministic; std 0)
  * metrics    = weighted P/R/F1 via bd.prf; recall_weighted == accuracy
The ONLY difference vs run_truncation_multiseed.py is base = CNN, and the CNN is fed
the (N, cut, F) time-major prefix (not the flattened vector).

Outputs (results_cnn/):  TRUNCATION_CNN_perseed__<dataset>.csv
(Separate dir + filename so it never clobbers the RF TRUNCATION_perseed__*.csv.)
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import mseed_common as mc
import baseline_data as bd
from cnn_base import SklearnCNN, DEFAULT_EPOCHS
from sklearn.metrics import accuracy_score


def run_dataset(dataset, seeds, epochs, outdir):
    ds = bd.canonical_name(dataset)
    print(f"\n=== Truncation-CNN  {ds}  pcts={mc.TRUNC_PERCENTAGES}  epochs={epochs} ===", flush=True)
    rows = []
    for seed in seeds:
        X, y, tr, te = mc.load_split(ds, seed)
        Vtr_full = np.asarray(bd.temporal_view(X[tr], ds), dtype=np.float32)   # (Ntr, T, F)
        Vte_full = np.asarray(bd.temporal_view(X[te], ds), dtype=np.float32)
        ytr, yte = y[tr], y[te]
        nfeat = Vtr_full.shape[2]
        for pct in mc.TRUNC_PERCENTAGES:
            _, cut = bd.truncate_flatten(X[tr], ds, pct)      # same cut the RF baseline uses
            cut = int(max(1, cut))
            Vtr, Vte = Vtr_full[:, :cut, :], Vte_full[:, :cut, :]
            clf = SklearnCNN(n_channels=nfeat, random_state=seed, n_epochs=epochs).fit(Vtr, ytr)
            yp = clf.predict(Vte)
            acc = accuracy_score(yte, yp)
            m = bd.prf(yte, yp)
            savings = (1.0 - pct / 100.0) * 100.0
            rows.append(mc.make_trunc_row(
                ds, seed, pct, acc, m["precision_weighted"], m["recall_weighted"],
                m["f1_weighted"], savings))
        accs = {r["pct"]: r["accuracy"] for r in rows if r["seed"] == seed}
        print(f"  seed={seed}  " + "  ".join(f"{p}%:{accs[p]:.1f}" for p in mc.TRUNC_PERCENTAGES)
              + f"  (n_test={len(yte)})", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"TRUNCATION_CNN_perseed__{ds}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=mc.TRUNC_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in mc.TRUNC_FIELDS})
    print(f"  -> wrote {out}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Truncation-CNN sweep (100% = Baseline-CNN)")
    ap.add_argument("--datasets", nargs="*", default=mc.DATASET_ORDER)
    ap.add_argument("--dataset", default=None, help="single dataset (SLURM-array); overrides --datasets")
    ap.add_argument("--seeds", nargs="*", type=int, default=mc.SEEDS)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--outdir", default=str(mc.HERE / "results_cnn"))
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else args.datasets
    outdir = Path(args.outdir)
    print(f"Truncation-CNN | seeds={args.seeds} | epochs={args.epochs} | datasets={datasets}", flush=True)
    for ds in datasets:
        run_dataset(ds, args.seeds, args.epochs, outdir)
    print(f"\nDONE. TRUNCATION_CNN_perseed__*.csv in {outdir}")


if __name__ == "__main__":
    main()
