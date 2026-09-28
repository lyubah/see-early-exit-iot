#!/usr/bin/env python3
"""
run_teaser_cnn_multiseed.py
===========================
CNN counterpart of run_teaser_multiseed.py: sktime TEASER with a CNN1D_extended slave
(cnn_base.SklearnCNN) + OneClassSVM master, instead of the RandomForest slave.

MATCHED to the RF TEASER cell (run_teaser_multiseed.py -> ComprehensiveTEASERAnalyzer):
  * engine      = sktime TEASER + OneClassSVM(rbf, gamma=scale), nu grid {0.1,0.3}
  * split       = mseed_common.load_split -> bd.temporal_view -> (N, F, T) (n,channels,time)
  * checkpoints = [T//4, T//2, 3T//4]  (25/50/75%), n_data = T   (identical to RF)
  * earliness   = 1 - decision_times / T   (state_info[:,3]); savings = mean * 100
  * metrics     = weighted P/R/F1 via bd.prf; recall_weighted == accuracy
  * seeds       = 42-46 (split rs == slave rs == master rs == seed)

The ONLY change vs the RF TEASER is the slave classifier -> CNN1D_extended. The shared
TEASER engine (run_teaser_benchmark.py) is deliberately NOT edited; the earliness/metric
formulas here are copied verbatim from ComprehensiveTEASERAnalyzer.fit_and_analyze so the
CNN cell is computed identically to the RF cell.

Outputs (results_cnn/):  TEASER_CNN_perseed__<dataset>.csv  (canonical 7-col schema)
"""
import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import mseed_common as mc
import baseline_data as bd
from cnn_base import SklearnCNN, DEFAULT_EPOCHS
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from sktime.classification.early_classification import TEASER

warnings.filterwarnings("ignore")


class CNNTimeSeriesSlave(BaseEstimator, ClassifierMixin):
    """CNN1D_extended slave for sktime TEASER. Accepts the data formats sktime hands a
    sub-estimator (nested DataFrame or 3D numpy (n, channels, timepoints)), converts to
    (n, timepoints, channels) = (n, T, F), and delegates to cnn_base.SklearnCNN. Mirrors
    SimpleTimeSeriesClassifier's data handling (run_teaser_benchmark.py:49) but feeds a
    CNN instead of flattening for a tree."""

    def __init__(self, n_epochs=DEFAULT_EPOCHS, random_state=42):
        self.n_epochs = n_epochs
        self.random_state = random_state

    def get_params(self, deep=True):
        return {"n_epochs": self.n_epochs, "random_state": self.random_state}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @staticmethod
    def _to_ntf(X):
        """-> (n, T, F) float32. Handles nested df (cells = pd.Series) and 3D numpy."""
        if isinstance(X, pd.DataFrame):
            n, nch = len(X), len(X.columns)
            nt = len(X.iloc[0, 0])
            arr = np.zeros((n, nch, nt), dtype=np.float32)
            for i in range(n):
                for j, col in enumerate(X.columns):
                    arr[i, j, :] = np.asarray(X.iloc[i, j], dtype=np.float32)
            return np.transpose(arr, (0, 2, 1))            # (n, T, F)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:                                    # (n, channels, timepoints)
            return np.transpose(X, (0, 2, 1))              # (n, T, F)
        if X.ndim == 2:                                    # (n, timepoints) single channel
            return X[:, :, None]
        raise ValueError(f"CNNTimeSeriesSlave: unexpected input ndim {X.ndim}")

    def fit(self, X, y):
        V = self._to_ntf(X)
        self._cnn = SklearnCNN(n_channels=V.shape[2], random_state=self.random_state,
                               n_epochs=self.n_epochs).fit(V, np.asarray(y))
        self.classes_ = self._cnn.classes_
        return self

    def predict(self, X):
        return self._cnn.predict(self._to_ntf(X))

    def predict_proba(self, X):
        return self._cnn.predict_proba(self._to_ntf(X))


def run_dataset(dataset, seeds, epochs, outdir):
    ds = bd.canonical_name(dataset)
    print(f"\n=== TEASER-CNN  {ds}  epochs={epochs} ===", flush=True)
    rows = []
    for seed in seeds:
        X, y, tr, te = mc.load_split(ds, seed)
        V = np.swapaxes(np.asarray(bd.temporal_view(X, ds)), 1, 2)   # (N, F, T) = (n,chan,time)
        T = V.shape[2]
        ckpt = [T // 4, T // 2, 3 * T // 4]                          # 25/50/75% (== RF)
        Xtr = _nested(V[tr]); Xte = _nested(V[te])
        ytr, yte = y[tr], y[te]

        teaser = TEASER(
            estimator=CNNTimeSeriesSlave(n_epochs=epochs, random_state=seed),
            classification_points=ckpt,
            one_class_classifier=OneClassSVM(kernel="rbf", gamma="scale"),
            one_class_param_grid={"nu": [0.1, 0.3]},
            random_state=seed, n_jobs=1)
        teaser.fit(Xtr, ytr)

        result = teaser.predict(Xte)
        yp = result[0] if isinstance(result, tuple) else result
        decision_times = teaser.state_info[:, 3].astype(int)
        savings = float((1.0 - decision_times / float(T)).mean()) * 100.0   # earliness (== RF)
        acc = accuracy_score(yte, yp)
        m = bd.prf(yte, yp)
        rows.append(mc.make_row(ds, seed, acc, m["precision_weighted"],
                                m["recall_weighted"], m["f1_weighted"], savings))
        print(f"  seed={seed}  ckpt={ckpt} (T={T})  acc={acc*100:5.1f}%  sav={savings:5.1f}%  "
              f"F1w={m['f1_weighted']:.3f}  n_test={len(yte)}", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"TEASER_CNN_perseed__{ds}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=mc.PERSEED_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in mc.PERSEED_FIELDS})
    print(f"  -> wrote {out}", flush=True)
    return rows


def _nested(V):
    """(n, channels, timepoints) numpy -> sktime nested DataFrame."""
    n, nch, _ = V.shape
    return pd.DataFrame({f"dim_{c}": [pd.Series(V[i, c, :]) for i in range(n)]
                         for c in range(nch)})


def main():
    ap = argparse.ArgumentParser(description="Multi-seed TEASER-CNN (CNN1D_extended slave + OCSVM)")
    ap.add_argument("--datasets", nargs="*", default=mc.DATASET_ORDER)
    ap.add_argument("--dataset", default=None, help="single dataset (SLURM-array); overrides --datasets")
    ap.add_argument("--seeds", nargs="*", type=int, default=mc.SEEDS)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--outdir", default=str(mc.HERE / "results_cnn"))
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else args.datasets
    outdir = Path(args.outdir)
    print(f"TEASER-CNN | seeds={args.seeds} | epochs={args.epochs} | datasets={datasets}", flush=True)
    for ds in datasets:
        run_dataset(ds, args.seeds, args.epochs, outdir)
    print(f"\nDONE. TEASER_CNN_perseed__*.csv in {outdir}")


if __name__ == "__main__":
    main()
