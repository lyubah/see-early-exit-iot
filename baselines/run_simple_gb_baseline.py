#!/usr/bin/env python3
"""
Full-window Gradient Boosting baseline.

Config matches the paper / Baseline-RF: 50 estimators, max_depth 30,
random_state 42. Fair preprocessing via baseline_data.py:
  - time-major flatten with xyz grouped per timestep (triaxial-aware for
    Shoaib/PAMAP2),
  - 60/20/20 split (rs=42); train on the 60% train split, test on the 20% test,
  - NO PAMAP2 augmentation,
  - WESAD on the standard random split (no LOSO) -- matches the proposed method.

Reports Accuracy + macro/weighted Precision/Recall/F1 on the test split.

Outputs simple_gb_results.csv with columns:
  Dataset, n_estimators, max_depth, accuracy,
  precision_macro, recall_macro, f1_macro,
  precision_weighted, recall_weighted, f1_weighted,
  n_train, n_test, n_features, n_classes

WESAD is split like every other dataset, the same split the SEE-RF models use.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "methods"))
import baseline_data as bd  # noqa: E402


N_ESTIMATORS = 50
MAX_DEPTH = 30
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent
bd.PROJECT_ROOT = PROJECT_ROOT   # read Datasets/ (and write results/) in this folder


def evaluate_gb(name: str) -> Dict:
    X_train, y_train, X_test, y_test, test_idx, window_len = bd.get_train_test(
        name, return_index=True)
    n_classes = len(np.unique(np.concatenate([y_train, y_test])))

    print(f"  Train: {X_train.shape}  Test: {X_test.shape}  Classes: {n_classes}")

    clf = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    m = bd.prf(y_test, y_pred)
    conf = bd.confusion_aggregates(y_test, y_pred)
    per_class = [{"Dataset": name, "Method": "GB", "config_id": "full", **pc}
                 for pc in bd.per_class_metrics(y_test, y_pred)]

    # Unified per-sample records: full window (sensing_axis='none'); compute =
    # MEASURED nodes (summed over GB's regressor trees via the helper fallback).
    records = bd.build_sample_records(
        dataset=name, method="GB", config_id="full",
        random_state=RANDOM_STATE, sample_ids=test_idx,
        y_true=y_test, y_pred=y_pred, y_proba=y_proba, classes=clf.classes_,
        sensing_axis="none", sensing_used=window_len, sensing_total=window_len,
        compute_units_used=bd.decision_path_nodes_per_sample(clf, X_test),
        compute_units_total=bd.ensemble_node_count(clf), compute_kind="nodes")

    return {
        "Dataset": name,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "accuracy": acc,
        **m,
        **conf,
        "n_train": int(len(y_train)),
        "n_features": int(X_train.shape[1]),
        "_per_class": per_class,
        "_records": records,
    }


def main() -> None:
    results: List[Dict] = []

    print("Gradient Boosting baseline (full data, Table IV)")
    print(f"  n_estimators = {N_ESTIMATORS}, max_depth = {MAX_DEPTH}, "
          f"random_state = {RANDOM_STATE}")
    print(f"  Preprocessing: baseline_data.py | split 60/20/20 rs="
          f"{bd.SPLIT_RANDOM_STATE} | AUGMENT_PAMAP2={bd.AUGMENT_PAMAP2}")
    print("")

    for ds in bd.DATASETS:
        print(f"Dataset: {ds}")
        try:
            res = evaluate_gb(ds)
            print(f"  Accuracy: {res['accuracy']*100:.2f}  "
                  f"F1(macro/weighted): {res['f1_macro']*100:.2f} / "
                  f"{res['f1_weighted']*100:.2f}")
            results.append(res)
        except FileNotFoundError as e:
            print(f"  SKIP {ds}: {e}")
        except Exception as e:
            print(f"  FAILED on {ds}: {e}")

    if not results:
        print("\nNo results (all datasets failed/missing).")
        sys.exit(1)

    results_dir = bd.ensure_results_dir()

    # Pull per-class + per-sample records out of the result dicts before tabulating.
    per_class_rows = []
    record_rows = []
    for r in results:
        per_class_rows.extend(r.pop("_per_class", []))
        record_rows.extend(r.pop("_records", []))

    df = pd.DataFrame(results)
    out_path = results_dir / "simple_gb_results.csv"
    df.to_csv(out_path, index=False)
    if per_class_rows:
        pd.DataFrame(per_class_rows).to_csv(results_dir / "simple_gb_per_class.csv", index=False)
    if record_rows:
        rec_path = results_dir / "simple_gb_records.csv"
        bd.write_sample_records(record_rows, rec_path)
        print(f"Per-sample records -> {rec_path}")

    # Unified plot-ready rows (full-data baseline -> 0% energy savings).
    plot_rows = [
        bd.plot_row(r["Dataset"], "GB", r["accuracy"], 0.0,
                    energy_type="full", n_estimators=r["n_estimators"], num_exits=1,
                    config_id="full", is_headline=True,
                    window_fraction_used=1.0, compute_fraction_used=1.0,
                    compute_fraction_source="full", savings_kind="none",
                    prf_dict={k: r[k] for k in bd._PRF_KEYS})
        for r in results
    ]
    bd.write_plot_csv(plot_rows, results_dir / "simple_gb_plot.csv")

    print("\nSummary (accuracy / F1 as %):")
    disp = df[["Dataset", "accuracy", "f1_macro", "f1_weighted",
               "FP", "FN", "TN", "specificity_macro", "support"]].copy()
    for c in ["accuracy", "f1_macro", "f1_weighted"]:
        disp[c] = (disp[c] * 100).round(2)
    disp["specificity_macro"] = disp["specificity_macro"].round(3)
    print(disp.to_string(index=False))
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
