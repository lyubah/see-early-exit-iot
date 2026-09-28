#!/usr/bin/env python3
"""
aggregate_cnn.py
================
Aggregate the CNN-base baseline per-seed CSVs (in results_cnn/) into paper-style
mean(std) tables, reusing mseed_common's formatters so the numbers are formatted
identically to the RF sweep.

Reads (results_cnn/):
  NONMYOPIC_CNN_perseed__<ds>.csv   TEASER_CNN_perseed__<ds>.csv   (7-col schema)
  TRUNCATION_CNN_perseed__<ds>.csv  (8-col TRUNC schema; pct=100 IS Baseline-CNN)

Writes (results_cnn/):
  CNN_BASELINES_multiseed.csv   one row per (method, dataset): acc(std) sav(std) P/R/F1
  TRUNCATION_CNN_multiseed.csv  one row per (dataset, pct)
"""
import argparse
import csv
import glob
from pathlib import Path

import mseed_common as mc


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _agg_single(outdir, method):
    """Aggregate a 7-col single-operating-point method (NONMYOPIC_CNN / TEASER_CNN)."""
    by_ds = {}
    for f in sorted(glob.glob(str(outdir / f"{method}_perseed__*.csv"))):
        for r in _read(f):
            by_ds.setdefault(r["dataset"], []).append(r)
    table = []
    for ds in mc.DATASET_ORDER:
        rs = by_ds.get(ds, [])
        if not rs:
            continue
        table.append({
            "method": method, "dataset": mc.DISPLAY.get(ds, ds), "n_seeds": len(rs),
            "accuracy": mc.fmt_pct([r["accuracy"] for r in rs]),
            "savings": mc.fmt_pct([r["savings"] for r in rs]),
            "precision_weighted": mc.fmt_dec([r["precision_weighted"] for r in rs]),
            "recall_weighted": mc.fmt_dec([r["recall_weighted"] for r in rs]),
            "f1_weighted": mc.fmt_dec([r["f1_weighted"] for r in rs]),
        })
    return table


def _agg_branchynet(outdir):
    """BranchyNet per-#exits (paper Table 3 'BN' column). Groups by (dataset, num_exits);
    num_exits=0 is the full-net 'Baseline' reference. Returns long-form rows + a wide
    accuracy matrix (dataset x {Baseline, BN@1..BN@4})."""
    by = {}
    for f in sorted(glob.glob(str(outdir / "BRANCHYNET_perseed__*.csv"))):
        for r in _read(f):
            by.setdefault((r["dataset"], int(r["num_exits"])), []).append(r)
    long_rows, wide = [], {}
    n_seen = set()
    for (dsname, n), rs in by.items():
        n_seen.add(n)
    for ds in mc.DATASET_ORDER:
        for n in sorted(n_seen):
            rs = by.get((ds, n))
            if not rs:
                continue
            role = "Baseline (full-net)" if n == 0 else f"BN {n}-exit"
            long_rows.append({
                "dataset": mc.DISPLAY.get(ds, ds), "num_exits": n, "role": role,
                "n_seeds": len(rs),
                "accuracy": mc.fmt_pct([r["accuracy"] for r in rs]),
                "exit1_accuracy": mc.fmt_pct([r["exit1_accuracy"] for r in rs]),
                "compute_savings": mc.fmt_pct([r["savings"] for r in rs]),
                "precision_weighted": mc.fmt_dec([r["precision_weighted"] for r in rs]),
                "recall_weighted": mc.fmt_dec([r["recall_weighted"] for r in rs]),
                "f1_weighted": mc.fmt_dec([r["f1_weighted"] for r in rs]),
            })
            wide.setdefault(mc.DISPLAY.get(ds, ds), {})[n] = mc.fmt_pct([r["accuracy"] for r in rs])
    return long_rows, wide, sorted(n_seen)


def _agg_trunc(outdir):
    """Aggregate the Truncation-CNN P%-sweep (pct=100 == Baseline-CNN)."""
    by = {}
    for f in sorted(glob.glob(str(outdir / "TRUNCATION_CNN_perseed__*.csv"))):
        for r in _read(f):
            by.setdefault((r["dataset"], int(r["pct"])), []).append(r)
    out = []
    for ds in mc.DATASET_ORDER:
        for pct in mc.TRUNC_PERCENTAGES:
            rs = by.get((ds, pct))
            if not rs:
                continue
            out.append({
                "dataset": mc.DISPLAY.get(ds, ds), "pct": pct, "n_seeds": len(rs),
                "role": "Baseline-CNN" if pct == 100 else "Truncation-CNN",
                "accuracy": mc.fmt_pct([r["accuracy"] for r in rs]),
                "savings": mc.fmt_pct([r["savings"] for r in rs]),
                "precision_weighted": mc.fmt_dec([r["precision_weighted"] for r in rs]),
                "recall_weighted": mc.fmt_dec([r["recall_weighted"] for r in rs]),
                "f1_weighted": mc.fmt_dec([r["f1_weighted"] for r in rs]),
            })
    return out


def main():
    ap = argparse.ArgumentParser(description="Aggregate CNN-base baseline per-seed CSVs")
    ap.add_argument("--outdir", default=str(mc.HERE / "results_cnn"))
    args = ap.parse_args()
    outdir = Path(args.outdir)

    single = []
    for method in ("NONMYOPIC_CNN", "TEASER_CNN"):    # sensing baselines (single op point)
        single += _agg_single(outdir, method)
    cols = ["method", "dataset", "n_seeds", "accuracy", "savings",
            "precision_weighted", "recall_weighted", "f1_weighted"]
    p1 = outdir / "CNN_BASELINES_multiseed.csv"
    with open(p1, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(single)

    # BranchyNet: per-#exits (paper Table 3 BN column) -> long form + wide accuracy matrix
    bn_long, bn_wide, bn_ns = _agg_branchynet(outdir)
    pbn = outdir / "BRANCHYNET_multiseed.csv"
    bcols = ["dataset", "num_exits", "role", "n_seeds", "accuracy", "exit1_accuracy",
             "compute_savings", "precision_weighted", "recall_weighted", "f1_weighted"]
    with open(pbn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=bcols)
        w.writeheader()
        w.writerows(bn_long)
    pbn2 = outdir / "BN_ACCURACY_TABLE.csv"     # Table-3-style: dataset x {Baseline, BN@n}
    hdr = ["dataset"] + (["Baseline"] if 0 in bn_ns else []) + [f"BN_{n}exit" for n in bn_ns if n > 0]
    with open(pbn2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for ds, cells in bn_wide.items():
            row = [ds] + ([cells.get(0, "")] if 0 in bn_ns else []) + [cells.get(n, "") for n in bn_ns if n > 0]
            w.writerow(row)

    trunc = _agg_trunc(outdir)
    tcols = ["dataset", "pct", "role", "n_seeds", "accuracy", "savings",
             "precision_weighted", "recall_weighted", "f1_weighted"]
    p2 = outdir / "TRUNCATION_CNN_multiseed.csv"
    with open(p2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=tcols)
        w.writeheader()
        w.writerows(trunc)

    print(f"wrote {p1}  ({len(single)} rows)")
    print(f"wrote {pbn}  ({len(bn_long)} rows)  + {pbn2}")
    print(f"wrote {p2}  ({len(trunc)} rows)")
    print("  --- sensing baselines (single operating point) ---")
    for t in single:
        print(f"  {t['method']:14s} {t['dataset']:8s} acc={t['accuracy']:>12s}  sav={t['savings']:>12s}")
    print("  --- BranchyNet accuracy per #exits (paper Table 3 'BN') ---")
    print("  " + "  ".join(f"{h:>12s}" for h in hdr))
    for ds, cells in bn_wide.items():
        line = [f"{ds:>12s}"] + ([f"{cells.get(0,''):>12s}"] if 0 in bn_ns else []) \
               + [f"{cells.get(n,''):>12s}" for n in bn_ns if n > 0]
        print("  " + "  ".join(line))
    print("  --- Baseline-CNN (Truncation pct=100) ---")
    for t in trunc:
        if t["pct"] == 100:
            print(f"  Baseline-CNN   {t['dataset']:8s} acc={t['accuracy']:>12s}")


if __name__ == "__main__":
    main()
