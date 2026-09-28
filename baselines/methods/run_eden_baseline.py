#!/usr/bin/env python3
"""
EDEN baseline (recent energy-efficient decision-tree-ensemble method).

EDEN (Daghero et al., "Dynamic Decision Tree Ensembles for Energy-Efficient
Inference on IoT Edge Nodes", github.com/eml-eda/eden) takes a trained sklearn
RandomForestClassifier, quantizes it, and runs ADAPTIVE inference that early-
stops evaluating trees once a confidence margin is met. It reduces COMPUTE
energy (fewer trees) while reading the FULL sensor window -- so it does not
reduce sensing energy.

We compare it on the same fair footing as the other RF baselines via
baseline_data.py: no PAMAP2 augmentation, 60/20/20 split (rs=42), train
on the 60% train / test on the 20% test, time-major triaxial-aware flatten,
WESAD on the standard split (no LOSO). RF config matches GB/Baseline-RF
(50 estimators, max_depth 30).

Energy is computed with the SAME methodology as the other early-classification
baselines (tslearn/TEASER): savings = 1 - mean(resource_used / resource_total).
EDEN's adaptive resource is the number of trees, so
    compute_energy_savings = (1 - mean(trees_used) / n_estimators) * 100.
We also record sensing_energy_savings = 0.0 (EDEN reads the full window) for an
honest same-axis note, analogous to the paper's footnote that TEASER/Non-Myopic
do not physically control sensor power.

A margin-threshold sweep produces an accuracy-vs-compute-energy Pareto; the
"headline" operating point (is_headline=True) is the fewest mean trees whose
accuracy stays within EPSILON_ACC of the full-ensemble accuracy.

Outputs run_eden_results.csv. Run `python run_eden_baseline.py --smoke` for an
iris plumbing test before touching real data.

Install (git only; PyPI 'eden' is unrelated):
    pip install "git+https://github.com/eml-eda/eden.git"
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bigtree import preorder_iter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from eden.transform.quantization import (quantize, quantize_leaves,
                                         quantize_pre_training_alphas)
from eden.frontend.sklearn.ensemble import parse_random_forest
from eden.inference import predict_adaptive, score_margin

import baseline_data as bd


N_ESTIMATORS = 50
MAX_DEPTH = 30
RANDOM_STATE = 42
INPUT_BITS = 8            # default precision; fall back to 16 then 32 if accuracy degrades
EPSILON_ACC = 0.005       # 0.5 pt tolerance for the headline operating point
N_THRESHOLDS = 12

PROJECT_ROOT = Path(__file__).resolve().parent


def _squeeze_leaves(ensemble):
    """sklearn >=1.x stores classifier leaves as (1, n_classes); EDEN's
    predict_raw sizes by values.shape[0]. Flatten every node's values to
    (n_classes,) so predict_raw / leaf_length work."""
    for tree in ensemble.flat_trees:
        for node in preorder_iter(tree):
            v = np.asarray(node.values)
            if v.ndim > 1:
                node.values = v.reshape(-1)
    return ensemble


def eden_adaptive_eval(X_train, y_train, X_test, y_test, input_bits,
                       name=None, test_idx=None, window_len=None) -> Tuple[List[Dict], float, float, List[Dict]]:
    """Train an RF, hand it to EDEN, sweep margin thresholds.

    Returns (rows, sklearn_fp_accuracy, full_ensemble_accuracy, sample_records).
    """
    mn, mx = float(X_train.min()), float(X_train.max())
    Xtr_q, _, _ = quantize(data=X_train, min_val=mn, max_val=mx, precision=input_bits)
    Xte_q, _, _ = quantize(data=X_test, min_val=mn, max_val=mx, precision=input_bits)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(Xtr_q, y_train)
    fp_acc = accuracy_score(y_test, model.predict(Xte_q))

    em = _squeeze_leaves(parse_random_forest(model=model))
    em = quantize_leaves(em, precision=input_bits)
    em = quantize_pre_training_alphas(em, precision=input_bits, min_val=mn, max_val=mx)

    # Per-tree quantized logits -> (n_trees, n_samples, n_classes); widen before cumsum.
    qpred = em.predict_raw(Xte_q).transpose(1, 0, 2).astype(np.int64)
    acc_logits = np.cumsum(qpred, axis=0)
    classes = model.classes_   # argmax gives a class INDEX; map back to labels
    full_acc = accuracy_score(y_test, classes[acc_logits[-1].argmax(-1)])

    early_scores = score_margin(acc_logits)                  # (n_trees, n_samples)
    hi = float(np.percentile(early_scores[-1], 99))
    thresholds = np.unique(np.linspace(0, max(hi, 1.0), N_THRESHOLDS).astype(int))
    adaptive_pred, classifiers_used = predict_adaptive(
        predictions=acc_logits, thresholds=thresholds, early_scores=early_scores)

    rows = []
    sample_records = []
    sample_ids = test_idx if test_idx is not None else np.arange(len(y_test))
    wlen = window_len if window_len else X_test.shape[1]
    for i, th in enumerate(thresholds):
        y_pred = classes[adaptive_pred[i].argmax(-1)]
        mean_trees = float(classifiers_used[i].mean())
        # Per-sample records: full window (sensing_axis='none'); compute = trees
        # actually executed before the score-margin early-stop (classifiers_used).
        ap = np.asarray(adaptive_pred[i], dtype=float)
        rsum = ap.sum(axis=1, keepdims=True)
        rsum[rsum == 0] = 1.0
        sample_records.extend(bd.build_sample_records(
            dataset=name, method="EDEN", config_id=f"th={int(th)}",
            random_state=RANDOM_STATE, sample_ids=sample_ids,
            y_true=y_test, y_pred=y_pred, y_proba=ap / rsum, classes=classes,
            sensing_axis="none", sensing_used=wlen, sensing_total=wlen,
            compute_units_used=classifiers_used[i], compute_units_total=N_ESTIMATORS,
            compute_kind="trees"))
        m = bd.prf(y_test, y_pred)
        conf = bd.confusion_aggregates(y_test, y_pred)
        per_class = [{"config_id": f"th={int(th)}", **pc}
                     for pc in bd.per_class_metrics(y_test, y_pred)]
        compute_frac = mean_trees / N_ESTIMATORS          # window_frac is 1.0 (full window)
        rows.append({
            "threshold": int(th),
            "input_bits": int(input_bits),
            "accuracy": accuracy_score(y_test, y_pred),
            **m,
            **conf,
            "mean_trees_used": mean_trees,
            "compute_energy_savings": (1.0 - compute_frac) * 100.0,
            "sensing_energy_savings": 0.0,   # EDEN reads the full sensor window
            # TOTAL-energy savings on the shared axis: only the compute slice is saved.
            "total_energy_savings": (1.0 - bd.SENSING_SHARE) * (1.0 - compute_frac) * 100.0,
            "_per_class": per_class,
        })
    return rows, float(fp_acc), float(full_acc), sample_records


def evaluate_eden(name) -> List[Dict]:
    X_train, y_train, X_test, y_test, test_idx, window_len = bd.get_train_test(
        name, return_index=True)
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))
    print(f"  Train {X_train.shape}  Test {X_test.shape}  Classes {n_classes}")

    # Float reference RF (no quantization) == the Baseline-RF accuracy.
    ref = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    ref.fit(X_train, y_train)
    float_acc = accuracy_score(y_test, ref.predict(X_test))

    # Quantization-precision fallback: 8 -> 16 -> 32 until EDEN's full ensemble
    # recovers the float RF accuracy (EDEN's input quantization can lose accuracy;
    # higher precision restores it, mirroring its "<1% drop" deployment story).
    bits_used = None
    records = []
    for bits in (INPUT_BITS, 16, 32):
        rows, fp_q, full_acc, records = eden_adaptive_eval(
            X_train, y_train, X_test, y_test, bits,
            name=name, test_idx=test_idx, window_len=window_len)
        bits_used = bits
        if full_acc >= float_acc - EPSILON_ACC:
            break
        print(f"  {bits}-bit quantization lost accuracy "
              f"(full={full_acc:.3f} < float RF={float_acc:.3f}); retrying higher precision")

    # Headline operating point: fewest mean trees within EPSILON of EDEN's own
    # (quantized) full-ensemble accuracy ceiling.
    eligible = [r for r in rows if r["accuracy"] >= full_acc - EPSILON_ACC]
    headline = min(eligible, key=lambda r: r["mean_trees_used"]) if eligible else \
        max(rows, key=lambda r: r["threshold"])

    for r in rows:
        r["Dataset"] = name
        r["eval_method"] = "random_split"
        r["n_estimators"] = N_ESTIMATORS
        r["max_depth"] = MAX_DEPTH
        r["input_bits"] = bits_used
        r["full_ensemble_accuracy"] = full_acc
        r["float_rf_accuracy"] = float_acc
        r["sklearn_fp_accuracy"] = fp_q
        r["is_headline"] = (r is headline)
        for pc in r.get("_per_class", []):
            pc["Dataset"] = name
            pc["Method"] = "EDEN"
            pc["is_headline"] = (r is headline)

    print(f"  float RF acc={float_acc:.3f} | EDEN full-ensemble acc={full_acc:.3f} "
          f"({bits_used}-bit, sklearn-on-quant={fp_q:.3f})")
    print(f"  headline: acc={headline['accuracy']:.3f}  "
          f"mean_trees={headline['mean_trees_used']:.1f}  "
          f"compute_savings={headline['compute_energy_savings']:.1f}%  "
          f"TOTAL_savings={headline['total_energy_savings']:.1f}% "
          f"(SENSING_SHARE={bd.SENSING_SHARE})")
    if rows:
        rows[0]["_records"] = records   # drained in main(); written to run_eden_records.csv
    return rows


COLUMNS = ["Dataset", "eval_method", "n_estimators", "max_depth", "input_bits",
           "threshold", "is_headline", "accuracy",
           "precision_macro", "recall_macro", "f1_macro",
           "precision_weighted", "recall_weighted", "f1_weighted",
           "TP", "FP", "FN", "TN",
           "specificity_macro", "fpr_macro", "fnr_macro", "support", "n_classes",
           "mean_trees_used", "compute_energy_savings", "sensing_energy_savings",
           "total_energy_savings",
           "full_ensemble_accuracy", "float_rf_accuracy", "sklearn_fp_accuracy"]


def main():
    parser = argparse.ArgumentParser(description="EDEN adaptive-RF baseline")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a tiny iris plumbing test and exit.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset (default: all 6). Accepts aliases.")
    args = parser.parse_args()

    if args.smoke:
        _smoke()
        return

    datasets = [bd.canonical_name(args.dataset)] if args.dataset else bd.DATASETS

    print("EDEN adaptive-RF baseline")
    print(f"  RF: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
          f"random_state={RANDOM_STATE}, input_bits={INPUT_BITS} (fallback 16/32)")
    print(f"  Preprocessing: baseline_data.py | split 60/20/20 rs="
          f"{bd.SPLIT_RANDOM_STATE} | AUGMENT_PAMAP2={bd.AUGMENT_PAMAP2}")
    print(f"  Energy: compute_energy_savings=(1-mean_trees/n_est)*100; "
          f"sensing_energy_savings=0 (full window)\n")

    all_rows = []
    for ds in datasets:
        print(f"Dataset: {ds}")
        try:
            all_rows.extend(evaluate_eden(ds))
        except FileNotFoundError as e:
            print(f"  SKIP {ds}: {e}")
        except Exception as e:
            print(f"  FAILED on {ds}: {e}")
            import traceback
            traceback.print_exc()

    if not all_rows:
        print("\nNo results (all datasets failed/missing).")
        sys.exit(1)

    # Pull per-class + per-sample records out before tabulating.
    per_class_rows = []
    record_rows = []
    for r in all_rows:
        per_class_rows.extend(r.pop("_per_class", []))
        record_rows.extend(r.pop("_records", []))

    df = pd.DataFrame(all_rows)
    df = df[[c for c in COLUMNS if c in df.columns]]
    results_dir = bd.ensure_results_dir()
    out_path = results_dir / "run_eden_results.csv"
    df.to_csv(out_path, index=False)
    if per_class_rows:
        pd.DataFrame(per_class_rows).to_csv(results_dir / "run_eden_per_class.csv", index=False)
    if record_rows:
        rec_path = results_dir / "run_eden_records.csv"
        bd.write_sample_records(record_rows, rec_path)
        print(f"Per-sample records -> {rec_path}")

    # Unified plot-ready rows: full threshold sweep. energy_savings here is the
    # TOTAL-energy savings (comparable to SEE-RF/EENN) -- EDEN reads the full window
    # so it only saves the compute slice. (compute_energy_savings is kept in the
    # detailed results CSV for reference.)
    plot_rows = [
        bd.plot_row(r["Dataset"], "EDEN", r["accuracy"], r["total_energy_savings"],
                    energy_type="total", n_estimators=r["n_estimators"], num_exits=1,
                    config_id=f"th={r['threshold']},{r['input_bits']}bit",
                    is_headline=bool(r["is_headline"]),
                    window_fraction_used=1.0,
                    compute_fraction_used=r["mean_trees_used"] / r["n_estimators"],
                    compute_fraction_source="tree_count", savings_kind="compute_only",
                    prf_dict={k: r[k] for k in bd._PRF_KEYS})
        for r in all_rows
    ]
    bd.write_plot_csv(plot_rows, results_dir / "run_eden_plot.csv")

    print(f"\nHeadline operating points (Table-V comparison) | SENSING_SHARE={bd.SENSING_SHARE}:")
    head = df[df["is_headline"]].copy()
    disp = head[["Dataset", "accuracy", "f1_macro", "mean_trees_used",
                 "compute_energy_savings", "total_energy_savings", "input_bits"]].copy()
    disp["accuracy"] = (disp["accuracy"] * 100).round(2)
    disp["f1_macro"] = (disp["f1_macro"] * 100).round(2)
    disp["compute_energy_savings"] = disp["compute_energy_savings"].round(1)
    disp["total_energy_savings"] = disp["total_energy_savings"].round(1)
    print(disp.to_string(index=False))
    print(f"\nSaved full sweep (Pareto) to: {out_path}")


def _smoke():
    """Iris plumbing test: asserts EDEN full-ensemble accuracy ~= sklearn."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    rows, fp, full = eden_adaptive_eval(Xtr, ytr, Xte, yte, INPUT_BITS)
    assert abs(full - fp) < 0.02, f"full={full} vs fp={fp}"
    aggressive = min(rows, key=lambda r: r["threshold"])
    conservative = max(rows, key=lambda r: r["threshold"])
    print(f"EDEN smoke OK | fp={fp:.3f} full={full:.3f}")
    print(f"  aggressive th={aggressive['threshold']}: acc={aggressive['accuracy']:.3f} "
          f"trees={aggressive['mean_trees_used']:.1f}")
    print(f"  full     th={conservative['threshold']}: acc={conservative['accuracy']:.3f} "
          f"trees={conservative['mean_trees_used']:.1f}")


if __name__ == "__main__":
    main()
