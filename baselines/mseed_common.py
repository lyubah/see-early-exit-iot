#!/usr/bin/env python3
"""
mseed_common.py
===============
Shared multi-seed config + metric aggregation + table formatting for the
baselines:

  * NonMyopic (tslearn NonMyopicEarlyClassifier, RF base)  -> run_nonmyopic_multiseed.py
  * TEASER    (RF slave + OneClassSVM master)              -> run_teaser_multiseed.py
  * Adaptive RF / EDEN                                     -> run_eden_multiseed.py

Design goals (match the paper's tab:model_comparison exactly):
  * 5 seeds [42,43,44,45,46]; EACH seed varies BOTH the train/test split
    random_state AND the model random_state.
  * weighted precision / recall / F1 (so recall_weighted == accuracy).
  * std = SAMPLE std (ddof=1) -> statistics.stdev.
  * accuracy & savings: percent, 1 decimal     e.g. "92.8 (1.1)"
  * precision/recall/f1:  decimals, 3 places    e.g. "0.928 (0.010)"

Data is read from Datasets/ in this folder (a link to the repo-level Datasets/);
the shared helper modules (baseline_data, seg_layout, run_teaser_benchmark,
run_eden_baseline, nonmyopic_config) live in methods/.
"""
import os
import sys
import csv
import glob
import statistics
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = HERE.parent
SEEN_DIR = REPO_ROOT / "seen_cnn"   # CNN1D early-exit network used by the BranchyNet baseline
METHODS_DIR = HERE / "methods"      # shared helper modules

# Resolution order (front of sys.path wins): THIS folder, then methods/, then the
# repo root as a fallback.
_paths = [str(REPO_ROOT)]
if METHODS_DIR.is_dir():
    _paths.append(str(METHODS_DIR))
_paths.append(str(HERE))
for p in _paths:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

import baseline_data as bd  # noqa: E402

# Point the shared data loader at this folder's Datasets/.
# _dataset_path looks in PROJECT_ROOT/Datasets first.
if (HERE / "Datasets").is_dir():
    bd.PROJECT_ROOT = HERE

# ── multi-seed protocol ──────────────────────────────────────────────────────
SEEDS = [42, 43, 44, 45, 46]

# canonical pickle base name -> display label used in the paper table
DATASET_ORDER = ["Epilepsy", "PAMAP2", "Shoaib",
                 "WESADchest", "EMGPhysical", "SelfRegulationSCP1"]
DISPLAY = {
    "Epilepsy": "Epilepsy", "PAMAP2": "PAMAP2", "Shoaib": "Shoaib",
    "WESADchest": "WESAD", "EMGPhysical": "EMG", "SelfRegulationSCP1": "SCP",
}

# WESAD uses the same 7421-window build (classes 4003:2152:1266) as the SEE-RF
# models and is split by window like every other dataset. That build carries no
# subject IDs, so a subject-out split is not possible from it.


def load_split(name, seed):
    """(X_raw, y, train_idx, test_idx) for one (dataset, seed), identical across all
    baselines so the results stay matched.

    Every dataset uses <name>_dataLabels.pkl with a single random 60/40 train/test
    split by window, no stratification, random_state = seed (seeds 42-46 give the
    multi-seed spread; seed 42 matches the SEE-vRF split). WESAD is handled like
    every other dataset.
    """
    cn = bd.canonical_name(name)
    X, y = bd.load_raw(cn)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.40, random_state=seed)   # 60 / 40, no stratify
    return X, y, np.asarray(tr), np.asarray(te)


# single-operating-point methods (one table cell per dataset)
METHODS = ["NONMYOPIC", "TEASER", "EDEN"]
METHOD_LABEL = {"NONMYOPIC": "NonMyopic (tslearn)",
                "TEASER": "TEASER", "EDEN": "Adaptive RF (EDEN)",
                "TRUNCATION": "Static Truncation (RF)"}

# static truncation is a P%-SWEEP (Table VI), handled separately from the 3 above.
TRUNC_PERCENTAGES = [10, 20, 30, 40, 50, 100]

# ── canonical Baseline-RF / GB tree config (paper Table VII: 50 estimators, depth 30) ──
# SINGLE source of truth for the RF tree size shared by every RF-based method in this
# study: Baseline-RF, the GB baseline, EDEN's reference RF (run_eden_baseline), and the
# static-truncation ablation. The ablation MUST be 1-1 with the methods it ablates, so it
# reads these instead of hardcoding -- this constant can never silently drift from them.
BASELINE_RF_N_ESTIMATORS = 50
BASELINE_RF_MAX_DEPTH = 30

RESULTS_DIR = HERE / "results"
PERSEED_FIELDS = ["dataset", "seed", "accuracy",
                  "precision_weighted", "recall_weighted", "f1_weighted", "savings"]


# ── per-seed row helpers ─────────────────────────────────────────────────────
def make_row(dataset, seed, acc01, prec_w, rec_w, f1_w, savings_pct):
    """Build one (dataset, seed) record in the canonical units:
    accuracy & savings as PERCENT, P/R/F1 as DECIMALS (0-1)."""
    return {
        "dataset": dataset,
        "seed": int(seed),
        "accuracy": round(float(acc01) * 100.0, 4),
        "precision_weighted": round(float(prec_w), 6),
        "recall_weighted": round(float(rec_w), 6),
        "f1_weighted": round(float(f1_w), 6),
        "savings": round(float(savings_pct), 4),
    }


def results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def perseed_path(method, dataset=None):
    d = results_dir()
    if dataset:
        return d / f"{method}_perseed__{dataset}.csv"
    return d / f"{method}_perseed.csv"


def write_perseed(method, rows, dataset=None):
    """Write the canonical 7-column per-seed CSV (one file per dataset when
    `dataset` is given -- the SLURM-array-friendly layout)."""
    p = perseed_path(method, dataset)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PERSEED_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in PERSEED_FIELDS})
    return p


# ── aggregation + formatting ─────────────────────────────────────────────────
def _stdev(xs):
    xs = [float(x) for x in xs]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0   # ddof=1 sample std


def _mean(xs):
    return statistics.mean(float(x) for x in xs)


def fmt_pct(xs):
    """accuracy / savings -> 'mean (std)' in percent, 1 decimal."""
    return f"{_mean(xs):.1f} ({_stdev(xs):.1f})"


def fmt_dec(xs):
    """precision / recall / f1 -> 'mean (std)' as decimal, 3 places."""
    return f"{_mean(xs):.3f} ({_stdev(xs):.3f})"


def load_perseed(method):
    """Read every <method>_perseed__*.csv (and a combined one if present) and
    return {dataset: [row, ...]} keyed by canonical dataset name."""
    rows = []
    seen = set()
    files = sorted(glob.glob(str(results_dir() / f"{method}_perseed__*.csv")))
    combined = perseed_path(method)
    if combined.exists():
        files = files + [str(combined)]   # per-dataset files WIN; combined only fills gaps
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["dataset"], int(r["seed"]))
                if key in seen:          # combined file wins over per-dataset dupes
                    continue
                seen.add(key)
                rows.append(r)
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    return by_ds


def combine_perseed(method):
    """Concatenate per-dataset per-seed files into <method>_perseed.csv (dataset
    order, seed order) and return the path + the {dataset: rows} map."""
    by_ds = load_perseed(method)
    out = perseed_path(method)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PERSEED_FIELDS)
        w.writeheader()
        for ds in DATASET_ORDER:
            for r in sorted(by_ds.get(ds, []), key=lambda x: int(x["seed"])):
                w.writerow({k: r[k] for k in PERSEED_FIELDS})
    return out, by_ds


def multiseed_table(method):
    """Return list of per-dataset dicts with formatted 'mean (std)' cells +
    the raw means (for the .tex). Also sanity-checks recall_weighted==accuracy."""
    by_ds = load_perseed(method)
    table = []
    for ds in DATASET_ORDER:
        rs = by_ds.get(ds, [])
        if not rs:
            continue
        acc = [r["accuracy"] for r in rs]
        pw = [r["precision_weighted"] for r in rs]
        rw = [r["recall_weighted"] for r in rs]
        f1 = [r["f1_weighted"] for r in rs]
        sv = [r["savings"] for r in rs]
        # recall_weighted (decimal) * 100 must equal accuracy (percent).
        rec_acc_gap = abs(_mean(rw) * 100.0 - _mean(acc))
        table.append({
            "dataset": ds, "display": DISPLAY[ds], "n_seeds": len(rs),
            "accuracy": fmt_pct(acc), "precision_weighted": fmt_dec(pw),
            "recall_weighted": fmt_dec(rw), "f1_weighted": fmt_dec(f1),
            "savings": fmt_pct(sv),
            "_acc_mean": _mean(acc), "_acc_std": _stdev(acc),
            "_pw_mean": _mean(pw), "_rw_mean": _mean(rw), "_f1_mean": _mean(f1),
            "_sv_mean": _mean(sv), "_sv_std": _stdev(sv),
            "_recall_acc_gap": rec_acc_gap,
        })
    return table


def write_multiseed_csv(method):
    """Write <method>_multiseed.csv: one row per dataset of formatted cells."""
    table = multiseed_table(method)
    out = results_dir() / f"{method}_multiseed.csv"
    cols = ["dataset", "n_seeds", "accuracy", "precision_weighted",
            "recall_weighted", "f1_weighted", "savings"]
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for t in table:
            w.writerow([t["display"], t["n_seeds"], t["accuracy"],
                        t["precision_weighted"], t["recall_weighted"],
                        t["f1_weighted"], t["savings"]])
    return out, table


# ── static truncation (P%-sweep) helpers ─────────────────────────────────────
TRUNC_FIELDS = ["dataset", "seed", "pct", "accuracy",
                "precision_weighted", "recall_weighted", "f1_weighted", "savings"]


def make_trunc_row(dataset, seed, pct, acc01, prec_w, rec_w, f1_w, savings_pct):
    return {"dataset": dataset, "seed": int(seed), "pct": int(pct),
            "accuracy": round(float(acc01) * 100.0, 4),
            "precision_weighted": round(float(prec_w), 6),
            "recall_weighted": round(float(rec_w), 6),
            "f1_weighted": round(float(f1_w), 6),
            "savings": round(float(savings_pct), 4)}


def trunc_perseed_path(dataset=None):
    d = results_dir()
    return d / (f"TRUNCATION_perseed__{dataset}.csv" if dataset
                else "TRUNCATION_perseed.csv")


def write_trunc_perseed(rows, dataset=None):
    p = trunc_perseed_path(dataset)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TRUNC_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in TRUNC_FIELDS})
    return p


def load_trunc_perseed():
    """{(dataset, pct): [rows]} across all TRUNCATION_perseed__*.csv."""
    rows, seen = [], set()
    files = sorted(glob.glob(str(results_dir() / "TRUNCATION_perseed__*.csv")))
    comb = trunc_perseed_path()
    if comb.exists():
        files = files + [str(comb)]   # per-dataset files WIN; combined only fills gaps
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["dataset"], int(r["seed"]), int(r["pct"]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], int(r["pct"])), []).append(r)
    return by


def trunc_multiseed_table():
    """Per (dataset, pct): formatted mean(std) cells, dataset+pct ordered."""
    by = load_trunc_perseed()
    out = []
    for ds in DATASET_ORDER:
        for pct in TRUNC_PERCENTAGES:
            rs = by.get((ds, pct))
            if not rs:
                continue
            out.append({
                "dataset": ds, "display": DISPLAY[ds], "pct": pct, "n_seeds": len(rs),
                "accuracy": fmt_pct([r["accuracy"] for r in rs]),
                "precision_weighted": fmt_dec([r["precision_weighted"] for r in rs]),
                "recall_weighted": fmt_dec([r["recall_weighted"] for r in rs]),
                "f1_weighted": fmt_dec([r["f1_weighted"] for r in rs]),
                "savings": fmt_pct([r["savings"] for r in rs]),
            })
    return out
