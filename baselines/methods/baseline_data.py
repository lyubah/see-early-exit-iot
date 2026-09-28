#!/usr/bin/env python3
"""
baseline_data.py
================
Single source of truth for data loading / splitting / flattening used by ALL
comparative baselines (static truncation, TEASER, Non-Myopic, EDEN, GB,
Baseline-RF).

It mirrors the SEE-hRF loader ``see_rf/hrf/ReadFile.py`` so that the baselines
are a FAIR comparison to the SEE-RF models. Three things it guarantees:

1. Flattening order. Triaxial sensors must be flattened **time-major with the
   three axes grouped per timestep** (xyz_1, xyz_2, ...), NOT axis-major
   (x_1..x_n, y_1..y_n, z_1..z_n). For Shoaib and PAMAP2 the three axes are
   packed inside the timestamp axis (Traw = 3*T), so we reshape
   (N, sensors, 3, T) -> (N, T, sensors, 3) before flattening. For every other
   dataset the time-major order is just swapaxes(1, 2).reshape.
   For a plain RF/GB/EDEN this reorder is permutation-invariant (no accuracy
   change), but it is essential so that the static-truncation and
   early-classification baselines truncate the REAL time axis (T), not the
   packed 3*T axis.

2. Split. 60/20/20 via two nested train_test_split calls (test_size=0.20 then
   0.25) with random_state=42, exactly as ReadFile.py. Baselines train on the
   60% TRAIN split only and report on the 20% TEST split.

3. No PAMAP2 class-3 augmentation (the SEE-RF loader has none). Controlled by the
   ``AUGMENT_PAMAP2`` toggle below -- flip to True only if the proposed-side
   final run augments, so the baselines stay matched.

WESAD uses the same plain random 60/20/20 split as every other dataset, matching
the SEE-RF loader.
"""

import os
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Shared, self-verifying SEG_SIZE-axis layout resolver. temporal_view / flatten_full /
# truncate_flatten now route through it so the baselines and the proposed-side loader
# (see_rf/hrf/ReadFile.py) use the IDENTICAL, data-verified layout.
import seg_layout


# ── Configuration ──────────────────────────────────────────────────────────

# Datasets whose 3 axes are packed inside the timestamp axis (Traw = 3 * T).
TRIAXIAL = {"Shoaib", "PAMAP2"}

# Match the SEE-RF loader (no augmentation). Flip to True ONLY if the proposed-side
# final run re-introduces PAMAP2 class-3 augmentation, so baselines stay matched.
AUGMENT_PAMAP2 = False

# 60 / 20 / 20 split with this seed, exactly as see_rf/hrf/ReadFile.py.
SPLIT_RANDOM_STATE = 42

# WESAD windows are stored in subject-contiguous blocks (15 subjects). Used to build
# a leakage-free Leave-Subjects-Out split: the default random by-window split puts the
# SAME subject in train and test and inflates WESAD to ~100% (memorized physiology,
# not generalization). See wesad_subject_ids / split_subject_grouped / split_indices.
WESAD_N_SUBJECTS = 15

# Single folder where EVERY comparative-baseline output (results + plot CSVs) lands.
# Honors the BASELINE_RESULTS_DIR env var so a run can be redirected without editing code.
RESULTS_DIR = os.environ.get("BASELINE_RESULTS_DIR", "results")

# Sensing share of TOTAL system energy (the paper states sensing is 60-80% of the
# budget). Used to put methods on ONE comparable total-energy axis:
#     total_energy_used = SENSING_SHARE*window_fraction + (1-SENSING_SHARE)*compute_fraction
#     total_energy_savings = (1 - total_energy_used) * 100
# Early-exit methods (SEE-RF, static truncation, TEASER, Non-Myopic) cut sensing
# AND compute together, so their window-fraction savings already approximate total
# savings. EDEN is the exception: it reads the FULL window (window_fraction=1) and
# only reduces trees, so it saves only the compute slice -> (1-SENSING_SHARE)*(...).
# Change this one number to match your paper's assumed split (e.g. 0.6 or 0.8).
SENSING_SHARE = 0.7

# Canonical six health datasets (pickle base names).
DATASETS = ["Epilepsy", "PAMAP2", "Shoaib", "WESADchest",
            "EMGPhysical", "SelfRegulationSCP1"]

# Friendly aliases used by some baseline scripts -> canonical pickle name.
NAME_ALIASES = {
    "WESAD": "WESADchest",
    "SCP": "SelfRegulationSCP1",
    "SR-SCP1": "SelfRegulationSCP1",
    "EMG": "EMGPhysical",
}

PROJECT_ROOT = Path(__file__).resolve().parent


# ── Name + path resolution ─────────────────────────────────────────────────

def canonical_name(name):
    """Map a friendly alias (WESAD, SCP, EMG, ...) to its pickle base name."""
    return NAME_ALIASES.get(name, name)


def ensure_results_dir(path=None):
    """Create (if needed) and return the results folder; defaults to RESULTS_DIR
    (results/) at the project root. All baselines write here."""
    p = Path(path) if path else (PROJECT_ROOT / RESULTS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dataset_path(name):
    """Locate <name>_dataLabels.pkl, tolerating Datasets/ and Datasets/Datasets/."""
    name = canonical_name(name)
    candidates = [
        PROJECT_ROOT / "Datasets" / f"{name}_dataLabels.pkl",
        PROJECT_ROOT / "Datasets" / "Datasets" / f"{name}_dataLabels.pkl",
        Path("Datasets") / f"{name}_dataLabels.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Cannot find {name}_dataLabels.pkl under {PROJECT_ROOT/'Datasets'}")


# ── Loading ────────────────────────────────────────────────────────────────

def load_raw(name):
    """Return (X, y) with X shape (N, C, Traw), y shape (N,).

    Applies PAMAP2 class-3 augmentation ONLY if AUGMENT_PAMAP2 is True
    (default False, matching the SEE-RF loader). No scaling / relabeling.
    """
    name = canonical_name(name)
    with open(_dataset_path(name), "rb") as f:
        d = pickle.load(f)
    X = np.asarray(d["data"], dtype=np.float64)
    y = np.asarray(d["labels"]).ravel()

    if AUGMENT_PAMAP2 and name == "PAMAP2":
        mask = (y == 3)
        X = np.append(X, X[mask], axis=0)
        y = np.append(y, y[mask])

    return X, y


def load_subjects(name):
    """Return explicit per-window subject IDs from the pickle if present, else None.

    The rebuilt WESAD pickle (re-windowed from the raw per-subject recordings) carries
    a 'subjects' array so a real leave-one-subject-out / subject-disjoint split is
    possible. Returns None for datasets/pickles without subject IDs.
    """
    with open(_dataset_path(canonical_name(name)), "rb") as f:
        d = pickle.load(f)
    s = d.get("subjects") if isinstance(d, dict) else None
    return None if s is None else np.asarray(s).ravel()


# ── Flattening (mirrors ReadFile.py) ─────────────────────────────

def temporal_view(X, name):
    """Canonical time-major view (N, T_real, F); axis 1 = earliest -> latest time.

    Delegates to seg_layout.temporal_view, which DETECTS + verifies the SEG_SIZE-axis
    layout from the data (axis_major / interleaved / time) rather than assuming it.
    For triaxial datasets (Shoaib, PAMAP2) that means the 3 axes are unpacked
    correctly so axis 1 is real time; for everything else it is swapaxes(1, 2).

    Flattening this view (reshape(N, -1)) yields per-timestep groups (xyz_1, xyz_2,
    ...) and slicing axis 1 gives a CORRECT contiguous temporal truncation. This is
    bit-identical to ReadFile.py's _flatten because both now call seg_layout.
    """
    return seg_layout.temporal_view(X, canonical_name(name))


def flatten_full(X, name):
    """Full-window time-major flatten -> (N, F_total). == ReadFile.py _flatten."""
    return seg_layout.flatten_full(X, canonical_name(name))


def truncate_flatten(X, name, pct):
    """Keep the first ``pct`` percent of REAL timesteps, then flatten.

    Returns (X_flat, cut) where cut is the number of timesteps kept. Operates on
    the canonical time-major view, so for triaxial data it truncates real time (T),
    NOT the packed 3*T axis.
    """
    return seg_layout.truncate_flatten(X, canonical_name(name), pct)


# ── Splitting (mirrors ReadFile.py) ──────────────────────────────

def split_60_20_20(n, y, rs=SPLIT_RANDOM_STATE):
    """Return (train_idx, val_idx, test_idx) as int arrays.

    Two nested splits exactly as ReadFile.SplitData: first 20% test, then 25%
    of the remaining 80% as validation -> 60/20/20. No stratification.
    """
    lst = list(range(n))
    tmp_idx, test_idx, l_tmp, _ = train_test_split(
        lst, y, test_size=0.20, random_state=rs)
    train_idx, val_idx, _, _ = train_test_split(
        tmp_idx, l_tmp, test_size=0.25, random_state=rs)
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def wesad_subject_ids(n_samples, n_subjects=WESAD_N_SUBJECTS):
    """Infer per-window subject IDs for WESAD from its subject-contiguous block
    layout (standard WESAD preprocessing; 15 subjects). Mirrors inspect_wesad.py so
    the leakage-free split here matches the documented subject-out benchmark."""
    per = n_samples // n_subjects
    ids = np.repeat(np.arange(n_subjects), per)
    if len(ids) < n_samples:                       # trailing remainder -> last subject
        ids = np.concatenate([ids, np.full(n_samples - len(ids), n_subjects - 1)])
    return ids


def split_subject_grouped(subject_ids, rs=SPLIT_RANDOM_STATE):
    """Subject-DISJOINT 60/20/20 (by windows) so NO subject crosses train/val/test --
    the leakage-free analogue of split_60_20_20. Two nested GroupShuffleSplits
    (test 20%, then val 25% of the remaining), grouping on subject_ids."""
    idx = np.arange(len(subject_ids))
    tmp, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.20,
                                           random_state=rs).split(idx, groups=subject_ids))
    sub_tr, sub_val = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                             random_state=rs).split(tmp, groups=subject_ids[tmp]))
    return tmp[sub_tr], tmp[sub_val], test_idx


def window_order_is_shuffled(y, thresh=0.9):
    """True if the window ORDER looks randomly shuffled -- the number of label runs is
    ~ the random-shuffle expectation 1 + (n-1)(1 - sum p_k^2). When True, any
    subject/session grouping has been destroyed and CANNOT be reconstructed from this
    array (so a leakage-free subject-out split is impossible from the pickle alone)."""
    y = np.asarray(y); n = len(y)
    if n < 8:
        return False
    runs = int(np.sum(np.diff(y) != 0) + 1)
    p = np.array([np.mean(y == c) for c in np.unique(y)], dtype=float)
    exp_shuffled = 1.0 + (n - 1) * (1.0 - float(np.sum(p ** 2)))
    return exp_shuffled > 0 and (runs / exp_shuffled) > thresh


def split_indices(name, y, rs=SPLIT_RANDOM_STATE, wesad_loso=False):
    """(train, val, test) indices. Random 60/20/20 by default (matched across methods,
    leaky on WESAD). With wesad_loso=True, WESAD uses a subject-DISJOINT split.

    FAIL-CLOSED: the subject-out split only works if windows are stored in
    subject-contiguous blocks. WESADchest as shipped here is SHUFFLED (verified: label
    runs ~= the random-shuffle expectation; LeaveOneGroupOut on inferred blocks = 100%),
    so subjects cannot be recovered and a "subject-out" split would SILENTLY still leak
    (the random-split ~100% is unfixable downstream). We raise instead of faking it. The
    only honest fix is to re-window from the RAW WESAD recordings with real subject IDs."""
    if wesad_loso and canonical_name(name) == "WESADchest":
        subj = load_subjects(name)
        if subj is not None and len(subj) == len(y):
            # Rebuilt pickle carries REAL per-window subject IDs -> honest subject-out.
            return split_subject_grouped(subj, rs)
        if window_order_is_shuffled(y):
            raise ValueError(
                "WESAD windows are SHUFFLED in this pickle -- subject grouping is "
                "unrecoverable, so a leakage-free subject-out split is IMPOSSIBLE here "
                "(the random-split ~100% cannot be de-leaked downstream). Re-window from "
                "the raw WESAD recordings (subjects S2..S17) with real subject IDs, then "
                "split by subject. Do NOT report the WESAD random-split number as accuracy.")
        return split_subject_grouped(wesad_subject_ids(len(y)), rs)
    return split_60_20_20(len(y), y, rs)


# ── Convenience: full-window train/test ready for sklearn ──────────────────

def get_train_test(name, return_index=False, wesad_loso=False):
    """Load + split + flatten. Baselines train on TRAIN (60%), test on TEST (20%).

    Returns (X_train, y_train, X_test, y_test) with flattened features. With
    return_index=True also returns (test_idx, window_len): the original-dataset
    indices of the test rows (the cross-baseline join key for per-sample records)
    and the real time-window length T (timesteps), so callers can populate the
    unified records schema without reloading.
    """
    X, y = load_raw(name)
    train_idx, _, test_idx = split_indices(name, y, wesad_loso=wesad_loso)
    X_train = flatten_full(X[train_idx], name)
    X_test = flatten_full(X[test_idx], name)
    if return_index:
        window_len = int(temporal_view(X[test_idx], name).shape[1])
        return X_train, y[train_idx], X_test, y[test_idx], test_idx, window_len
    return X_train, y[train_idx], X_test, y[test_idx]


def get_train_test_view(name, return_index=False, wesad_loso=False):
    """Like get_train_test but returns (N, T, F) time-series views.

    For tslearn / TEASER (which consume (samples, timesteps, features)) and for
    temporal truncation. Returns (V_train, y_train, V_test, y_test). With
    return_index=True also returns test_idx (original-dataset indices of the test
    rows) -- the cross-baseline join key for per-sample records. wesad_loso=True
    swaps WESAD's random split for the subject-disjoint one (leakage-free).
    """
    X, y = load_raw(name)
    train_idx, _, test_idx = split_indices(name, y, wesad_loso=wesad_loso)
    V_train = temporal_view(X[train_idx], name)
    V_test = temporal_view(X[test_idx], name)
    if return_index:
        return V_train, y[train_idx], V_test, y[test_idx], test_idx
    return V_train, y[train_idx], V_test, y[test_idx]


# ── Metrics helper ─────────────────────────────────────────────────────────

def decision_path_work(clf, X):
    """Mean number of tree nodes visited per inference, summed across ALL trees
    in the ensemble -- a direct, MEASURED proxy for inference compute (the real
    number of node comparisons). Uses the model's decision_path. Returns nodes
    per sample; divide two models' values to get a compute_fraction.
    """
    dp = clf.decision_path(X)
    indicator = dp[0] if isinstance(dp, tuple) else dp
    n = X.shape[0]
    return float(indicator.sum()) / n if n else 0.0


def decision_path_nodes_per_sample(clf, X):
    """PER-SAMPLE tree-nodes visited, summed across all trees (MEASURED compute).

    Like decision_path_work but returns the (n_samples,) vector instead of the
    mean -- the per-row `compute_units_used` for the records schema. Falls back to
    per-estimator summation for ensembles without a top-level decision_path (e.g.
    GradientBoostingClassifier, whose estimators_ is a 2-D array of regressors).
    """
    if hasattr(clf, "decision_path"):
        try:
            dp = clf.decision_path(X)
            indicator = dp[0] if isinstance(dp, tuple) else dp
            return np.asarray(indicator.sum(axis=1)).ravel()
        except Exception:
            pass
    ests = getattr(clf, "estimators_", None)
    if ests is None:
        return np.zeros(X.shape[0])
    total = np.zeros(X.shape[0])
    for e in np.ravel(np.asarray(ests, dtype=object)):
        if hasattr(e, "decision_path"):
            total += np.asarray(e.decision_path(X).sum(axis=1)).ravel()
    return total


def harmonic_mean(accuracy, earliness_saved):
    """Harmonic mean of accuracy and earliness -- the early-TSC operating-point
    metric (as used by TEASER/Mori). `earliness_saved` is the fraction of the
    window NOT needed (= 1 - decision_time/T); both args are in [0,1]. Returns
    2*a*e/(a+e), or 0 if both are 0. Used to select each method's headline point."""
    a, e = float(accuracy), float(earliness_saved)
    return (2.0 * a * e / (a + e)) if (a + e) > 0 else 0.0


def ensemble_node_count(clf):
    """Total nodes across every tree in the ensemble = the full-traversal
    `compute_units_total` for the 'nodes' compute axis. Handles RF (1-D
    estimators_) and GB (2-D estimators_). Returns None if not tree-based."""
    ests = getattr(clf, "estimators_", None)
    if ests is None:
        return None
    total = 0
    for e in np.ravel(np.asarray(ests, dtype=object)):
        tree = getattr(e, "tree_", None)
        if tree is not None:
            total += int(tree.node_count)
    return int(total)


# ── Unified per-sample records schema ───────────────────────────────────────
# One row per TEST sample, per operating point, for EVERY comparative baseline.
# These are the raw atoms: accuracy / P-R-F1 / confusion / per-class / ECE /
# entropy / sensing-savings / compute-savings are all re-derivable from them
# WITHOUT re-running the model. Keep alongside (not instead of) the aggregate CSVs.
SAMPLE_RECORD_COLUMNS = [
    "dataset", "method", "config_id", "random_state", "sample_id",
    "y_true", "y_pred", "y_proba", "proba_classes",
    "sensing_axis", "sensing_used", "sensing_total", "sensing_fraction",
    "compute_units_used", "compute_units_total", "compute_kind",
]


def _py(v):
    """numpy scalar -> python scalar (so CSVs are clean)."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    return v


def proba_to_str(row):
    """Serialize a class-probability vector as a ';'-joined string (CSV-safe; no
    commas). Pair with proba_classes to know the label order."""
    if row is None:
        return ""
    return ";".join(f"{float(p):.6g}" for p in np.asarray(row).ravel())


def build_sample_records(*, dataset, method, config_id, random_state, sample_ids,
                         y_true, y_pred, y_proba=None, classes=None,
                         sensing_axis, sensing_used, sensing_total,
                         compute_units_used, compute_units_total, compute_kind):
    """Assemble unified per-sample records (list of dicts in SAMPLE_RECORD_COLUMNS
    order). sensing_used / compute_units_used may be scalar (broadcast to every
    sample) or a per-sample array; sensing_total / compute_units_total are scalar.
    classes is the predict_proba column order (for proba_classes)."""
    n = len(y_true)

    def _col(v):
        if v is None or np.isscalar(v):
            return [v] * n
        arr = np.asarray(v)
        return arr if arr.shape[0] == n else [v] * n

    s_used = _col(sensing_used)
    c_used = _col(compute_units_used)
    c_tot = _col(compute_units_total)
    cls_str = ";".join(str(c) for c in classes) if classes is not None else ""
    s_tot = float(sensing_total) if sensing_total not in (None, 0) else None
    has_proba = y_proba is not None
    rows = []
    for i in range(n):
        su = _py(s_used[i])
        rows.append({
            "dataset": dataset,
            "method": method,
            "config_id": config_id,
            "random_state": _py(random_state),
            "sample_id": int(sample_ids[i]),
            "y_true": _py(y_true[i]),
            "y_pred": _py(y_pred[i]),
            "y_proba": proba_to_str(y_proba[i]) if has_proba else "",
            "proba_classes": cls_str,
            "sensing_axis": sensing_axis,
            "sensing_used": su,
            "sensing_total": _py(sensing_total),
            "sensing_fraction": round(float(su) / s_tot, 6) if (s_tot and su is not None) else None,
            "compute_units_used": _py(c_used[i]),
            "compute_units_total": _py(c_tot[i]),
            "compute_kind": compute_kind,
        })
    return rows


def write_sample_records(rows, path):
    """Write per-sample records to CSV in canonical column order. Appends if the
    file exists (so multi-dataset / multi-seed runs accumulate one records file)."""
    if not rows:
        return None
    import pandas as pd
    df = pd.DataFrame(rows)
    for c in SAMPLE_RECORD_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[SAMPLE_RECORD_COLUMNS]
    header = not os.path.exists(str(path))
    df.to_csv(path, mode="a" if not header else "w", header=header, index=False)
    return path


def prf(y_true, y_pred):
    """Macro + weighted Precision/Recall/F1 (0-1 scale, zero_division=0).

    Returns a flat dict with keys precision_macro, recall_macro, f1_macro,
    precision_weighted, recall_weighted, f1_weighted.
    """
    return {
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def confusion_aggregates(y_true, y_pred, labels=None):
    """Confusion-matrix totals + macro rates for the main results table.

    Per-class one-vs-rest TP/FP/FN/TN are summed across classes. NOTE for
    single-label multi-class: sum(FP) == sum(FN) == number of misclassified
    samples (the per-class FP/FN differ — see per_class_metrics). TN is the
    summed per-class true-negative count. specificity/fpr/fnr are macro-averaged
    over classes (0-1).
    """
    if labels is None:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    tp = np.diag(cm).astype(np.int64)
    fp = cm.sum(axis=0).astype(np.int64) - tp
    fn = cm.sum(axis=1).astype(np.int64) - tp
    tn = total - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
        fpr = np.where((fp + tn) > 0, fp / (fp + tn), 0.0)
        fnr = np.where((fn + tp) > 0, fn / (fn + tp), 0.0)
    return {
        "support": total,
        "n_classes": len(labels),
        "TP": int(tp.sum()),          # == correct predictions
        "FP": int(fp.sum()),          # == FN == #errors for single-label multi-class
        "FN": int(fn.sum()),
        "TN": int(tn.sum()),
        "specificity_macro": float(spec.mean()),
        "fpr_macro": float(fpr.mean()),
        "fnr_macro": float(fnr.mean()),
    }


def per_class_metrics(y_true, y_pred, labels=None):
    """One-vs-rest TP/FP/FN/TN + precision/recall/f1/specificity PER class.

    Returns a list of dicts (one per class). This is where FP and FN are
    genuinely informative (they differ per class, unlike the multi-class totals).
    """
    if labels is None:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    rows = []
    for i, c in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rows.append({
            "class": c, "support": support,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec, "f1": f1, "specificity": spec,
        })
    return rows


# ── Unified plot-ready output (mirrors the SEE-RF records schema for graphing) ──
#
# The SEE-RF experiments save per-configuration rows (see_rf/hrf/
# RecordsFolder/<ds>_records.csv) with DataSet_Name, Num_Trees_per_RF,
# num_of_exits, Total_acc_Test (0-1), and an energy ratio. To recreate the paper
# figures (accuracy-vs-estimators, energy-savings, and the accuracy-energy Pareto
# Fig 8/9) with baselines OVERLAID, every baseline also emits a <name>_plot.csv in
# this single common schema. One row per evaluated point (full sweeps for static
# truncation / EDEN / Non-Myopic so curves can be drawn; is_headline marks the
# single point that goes in the comparison table).
#
# Scales (kept consistent with the SEE-RF records): accuracy and all P/R/F1 are
# 0-1; energy_savings is a percentage 0-100; energy_ratio = 1 - energy_savings/100
# (the fraction of the resource used, == the SEE-RF "E_ratio").

PLOT_COLUMNS = [
    "Dataset", "Method", "config_id", "is_headline",
    "n_estimators", "num_exits",
    "accuracy", "energy_savings", "energy_ratio", "energy_type",
    # --- raw energy ingredients saved for EVERY method, so any energy model can
    # --- be derived later WITHOUT recomputing the run. window_fraction_used is
    # --- the SAME quantity SEE-CNN / SEE-RF report (their E_ratio): the mean
    # --- fraction of the sensor window the prediction depends on. Then:
    #       sensing_energy_savings = (1 - window_fraction_used) * 100   (the SEE metric)
    #       compute_energy_savings = (1 - compute_fraction_used) * 100
    #       total_energy_savings   = (1 - (s*window_frac + (1-s)*compute_frac)) * 100, s=SENSING_SHARE
    "window_fraction_used", "compute_fraction_used", "compute_fraction_source",
    "sensing_energy_savings", "compute_energy_savings", "total_energy_savings",
    "savings_kind",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
]

_PRF_KEYS = ["precision_macro", "recall_macro", "f1_macro",
             "precision_weighted", "recall_weighted", "f1_weighted"]


def plot_row(dataset, method, accuracy, energy_savings, *, energy_type="sensing",
             n_estimators=None, num_exits=1, config_id="", is_headline=True,
             prf_dict=None, acc_scale=1.0, prf_scale=1.0,
             window_fraction_used=None, compute_fraction_used=None,
             compute_fraction_source=None, savings_kind=None, sensing_share=None):
    """Build one unified plot-ready row.

    accuracy is divided by acc_scale and prf_dict values by prf_scale so the
    output is always 0-1. energy_savings (legacy column) is the method's headline
    percentage (kept unchanged for backward compat).

    RAW ENERGY INGREDIENTS (the part that lets us derive any energy model later):
    pass window_fraction_used (fraction of the sensor window the prediction needs
    -- the SAME quantity SEE-CNN/SEE-RF report) and compute_fraction_used
    (fraction of full model work). From those we derive, with NO assumption baked
    in beyond the SENSING_SHARE convenience:
        sensing_energy_savings = (1 - window_fraction_used) * 100   (the SEE metric)
        compute_energy_savings = (1 - compute_fraction_used) * 100
        total_energy_savings   = (1 - (s*window + (1-s)*compute)) * 100, s=SENSING_SHARE
    savings_kind in {'measured','real','projected','compute_only','none'} flags how
    the saving is obtained (measured on HW / real reduction / optimistic projection /
    compute-only / no saving) -- it changes NO number.
    """
    es = float(energy_savings)
    s = SENSING_SHARE if sensing_share is None else float(sensing_share)
    row = OrderedDict()
    row["Dataset"] = canonical_name(dataset)
    row["Method"] = method
    row["config_id"] = config_id
    row["is_headline"] = bool(is_headline)
    row["n_estimators"] = n_estimators
    row["num_exits"] = num_exits
    row["accuracy"] = float(accuracy) / acc_scale
    row["energy_savings"] = es
    row["energy_ratio"] = 1.0 - es / 100.0
    row["energy_type"] = energy_type
    # raw ingredients + derived per-axis savings (None if fractions not given)
    row["window_fraction_used"] = window_fraction_used
    row["compute_fraction_used"] = compute_fraction_used
    row["compute_fraction_source"] = compute_fraction_source  # measured/tree_count/full/proxy
    if window_fraction_used is not None and compute_fraction_used is not None:
        wf, cf = float(window_fraction_used), float(compute_fraction_used)
        row["sensing_energy_savings"] = (1.0 - wf) * 100.0
        row["compute_energy_savings"] = (1.0 - cf) * 100.0
        row["total_energy_savings"] = (1.0 - (s * wf + (1.0 - s) * cf)) * 100.0
    else:
        row["sensing_energy_savings"] = None
        row["compute_energy_savings"] = None
        row["total_energy_savings"] = None
    row["savings_kind"] = savings_kind
    for k in _PRF_KEYS:
        row[k] = (float(prf_dict[k]) / prf_scale) if (prf_dict and prf_dict.get(k) is not None) else None
    return row


def write_plot_csv(rows, path):
    """Write unified plot rows to `path` with the fixed PLOT_COLUMNS order."""
    import pandas as pd
    df = pd.DataFrame(list(rows))
    for c in PLOT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df[PLOT_COLUMNS].to_csv(path, index=False)
    return path
