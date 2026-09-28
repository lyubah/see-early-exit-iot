#!/usr/bin/env python3
"""
run_nonmyopic_cnn_multiseed.py
==============================
CNN counterpart of run_nonmyopic_multiseed.py: tslearn NonMyopicEarlyClassifier with
the CNN1D_extended base (cnn_base.SklearnCNN) swapped in for the RandomForest.

IDENTICAL protocol to the RF version (so the cell is apples-to-apples):
  * split       = mseed_common.load_split -> bd.temporal_view  (N, T, F)
  * seeds       = 42-46 (split rs == model rs == seed)
  * n_clusters  = nonmyopic_config.CONFIG[ds]['nc'] with descending fallback
  * FIXED rule  = alpha = c/T, c = 0.02 (UNTUNED adaptive-regime floor; NOT the c-sweep
                  that origin/baselines NM_run_cnn.py used -- we match the matched sweep)
  * savings     = mean_TEST(1 - t_decision / T) * 100   (SEE-RF parity)
  * metrics     = weighted P/R/F1 via bd.prf; recall_weighted == accuracy

The ONLY change vs run_nonmyopic_multiseed.py is base = CNN1D_extended. tslearn fits
one base classifier PER timestamp (early_classification.py:163, range(min_t, sz+1)),
so this trains ~T CNNs per (dataset, seed) -- the heaviest baseline; keep --epochs low
for a smoke test, use a cluster for the full sweep. The CNN takes tslearn's 3D (n, t, F) prefix
slices directly (NO FlatWrap; FlatWrap exists only to 2D-flatten for the RF/1-NN base).

Outputs (results_cnn/):  NONMYOPIC_CNN_perseed__<dataset>.csv  (canonical 7-col schema)
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import mseed_common as mc
import baseline_data as bd
from cnn_base import SklearnCNN, DEFAULT_EPOCHS
from sklearn.metrics import accuracy_score
from tslearn.early_classification import NonMyopicEarlyClassifier
from tslearn.utils import to_time_series_dataset

# n_clusters per dataset (structural; NOT chosen on accuracy/savings) -- same source the
# RF version uses. Only CONFIG is imported (FlatWrap is RF-only, unused for the CNN).
from nonmyopic_config import CONFIG

C_TIME = 0.02          # identical fixed constant as run_nonmyopic_multiseed.py


def _split_view(name, seed):
    X, y, tr, te = mc.load_split(name, seed)
    V = np.asarray(bd.temporal_view(X, name), dtype=np.float32)   # (N, T, F)
    return V[tr], y[tr], V[te], y[te]


def _savings(dt, T):
    dt = np.clip(np.asarray(dt).astype(int), 1, T)
    return float(np.mean(1.0 - dt / float(T))) * 100.0


def _fit_nonmyopic(Vtr, ytr, nc_pref, alpha, seed, epochs):
    """Fit NonMyopic (CNN base) at the fixed alpha=c/T, with the same n_clusters
    fallback the RF version uses (a TimeSeriesKMeans cluster may be too small to
    stratify tslearn's internal 50/50 split)."""
    Xts = to_time_series_dataset(Vtr)
    nfeat = int(Vtr.shape[2])
    for nc in range(nc_pref, 0, -1):
        try:
            clf = NonMyopicEarlyClassifier(
                base_classifier=SklearnCNN(n_channels=nfeat, random_state=seed, n_epochs=epochs),
                n_clusters=nc, cost_time_parameter=alpha, random_state=seed)
            clf.fit(Xts, ytr)
            return clf, nc
        except ValueError as e:
            if ("least populated class" in str(e)
                    or "minimum number of groups" in str(e)) and nc > 1:
                continue
            raise
    raise RuntimeError("NonMyopic (CNN) fit failed at all n_clusters")


def run_dataset(dataset, seeds, c_time, epochs, outdir):
    ds = bd.canonical_name(dataset)
    cfg = CONFIG[ds]
    print(f"\n=== NonMyopic-CNN  {ds}  nc={cfg['nc']}  alpha=c/T (c={c_time:g})  "
          f"epochs={epochs} ===", flush=True)
    rows = []
    for seed in seeds:
        Vtr, ytr, Vte, yte = _split_view(ds, seed)
        T = int(Vtr.shape[1])
        alpha = c_time / float(T)
        clf, nc_used = _fit_nonmyopic(Vtr, ytr, cfg["nc"], alpha, seed, epochs)
        clf.cost_time_parameter = alpha
        yp, dt = clf.predict_class_and_earliness(to_time_series_dataset(Vte))
        acc = accuracy_score(yte, yp)
        m = bd.prf(yte, yp)
        savings = _savings(dt, T)
        exits_vary = len(np.unique(np.clip(np.asarray(dt).astype(int), 1, T))) > 1
        rows.append(mc.make_row(ds, seed, acc, m["precision_weighted"],
                                m["recall_weighted"], m["f1_weighted"], savings))
        if nc_used != cfg["nc"]:
            print(f"  [note] seed={seed}: n_clusters {cfg['nc']}->{nc_used}", flush=True)
        print(f"  seed={seed}  alpha={alpha:.3g} (T={T})  acc={acc*100:5.1f}%  "
              f"sav={savings:5.1f}%  F1w={m['f1_weighted']:.3f}  n_test={len(yte)}  "
              f"exit_varies={exits_vary}", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"NONMYOPIC_CNN_perseed__{ds}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=mc.PERSEED_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in mc.PERSEED_FIELDS})
    print(f"  -> wrote {out}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Multi-seed NonMyopic-CNN (CNN1D_extended; fixed alpha=c/T)")
    ap.add_argument("--datasets", nargs="*", default=mc.DATASET_ORDER)
    ap.add_argument("--dataset", default=None, help="single dataset (SLURM-array); overrides --datasets")
    ap.add_argument("--seeds", nargs="*", type=int, default=mc.SEEDS)
    ap.add_argument("--c-time", type=float, default=C_TIME)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--outdir", default=str(mc.HERE / "results_cnn"))
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else args.datasets
    outdir = Path(args.outdir)
    print(f"NonMyopic-CNN | seeds={args.seeds} | c={args.c_time:g} | epochs={args.epochs} | "
          f"datasets={datasets}", flush=True)
    for ds in datasets:
        run_dataset(ds, args.seeds, args.c_time, args.epochs, outdir)
    print(f"\nDONE. NONMYOPIC_CNN_perseed__*.csv in {outdir}")


if __name__ == "__main__":
    main()
