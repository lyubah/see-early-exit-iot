#!/usr/bin/env python3
"""
make_cnn_paper_table.py
=======================
Assemble ALL CNN-base baselines into one paper-ready comparison table (CSV + Markdown),
reading the per-seed CSVs in results_cnn/. This is the drop-in baseline block for the
SEEN paper: the stated baselines (Baseline-CNN, BranchyNet) plus the additional
early-classification baselines we are adding (Truncation / NonMyopic / TEASER on the
CNN1D_extended = "Model 1 CNN" backbone).

Each cell is `acc% (sav%)`, mean over seeds. SAVINGS AXIS differs by method and is
labeled in the header:
  * Baseline-CNN   : full window, no exit            -> sav = 0 (reference)
  * BranchyNet(BN) : depth early-exit, FULL window   -> sav = COMPUTE (best #exits by acc)
  * Truncation-CNN : partial window (iso-accuracy)   -> sav = SENSING
  * NonMyopic-CNN  : partial window (fixed c=0.02)    -> sav = SENSING
  * TEASER-CNN     : partial window (OCSVM master)    -> sav = SENSING
SEEN (the proposed method) is NOT produced here -- fill its column from the SEEN runs.

Usage:  python make_cnn_paper_table.py [--outdir results_cnn] [--iso-eps 1.0]
Outputs (results_cnn/):  CNN_PAPER_TABLE.csv, CNN_PAPER_TABLE.md
"""
import argparse
import csv
import glob
from pathlib import Path

import mseed_common as mc


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else float("nan")


def _by_dataset(pattern, outdir):
    d = {}
    for f in sorted(glob.glob(str(outdir / pattern))):
        for r in _read(f):
            d.setdefault(r["dataset"], []).append(r)
    return d


def _cell(acc, sav=None):
    if acc != acc:                       # NaN
        return "-"
    return f"{acc:.1f} ({sav:.1f})" if sav is not None else f"{acc:.1f}"


def build(outdir, iso_eps):
    trunc = _by_dataset("TRUNCATION_CNN_perseed__*.csv", outdir)
    nm = _by_dataset("NONMYOPIC_CNN_perseed__*.csv", outdir)
    te = _by_dataset("TEASER_CNN_perseed__*.csv", outdir)
    bn = _by_dataset("BRANCHYNET_perseed__*.csv", outdir)

    table = []
    for ds in mc.DATASET_ORDER:
        row = {"dataset": mc.DISPLAY.get(ds, ds)}

        # Baseline-CNN + Truncation-CNN @ iso-accuracy (from the P%-sweep)
        base_acc = float("nan"); trunc_cell = "-"
        if ds in trunc:
            by_pct = {}
            for r in trunc[ds]:
                by_pct.setdefault(int(r["pct"]), []).append(r)
            accs = {p: _mean([r["accuracy"] for r in rs]) for p, rs in by_pct.items()}
            savs = {p: _mean([r["savings"] for r in rs]) for p, rs in by_pct.items()}
            base_acc = accs.get(100, float("nan"))
            # iso-accuracy: smallest pct (max sensing savings) within iso_eps of Baseline-CNN
            elig = [p for p in sorted(by_pct) if p < 100 and accs[p] >= base_acc - iso_eps]
            p_iso = min(elig) if elig else 100
            trunc_cell = _cell(accs[p_iso], savs[p_iso])
        row["Baseline-CNN"] = _cell(base_acc)
        row["Truncation-CNN (sens)"] = trunc_cell

        # BranchyNet: best accuracy over #exits>=1 (compute savings)
        bn_cell = "-"
        if ds in bn:
            by_n = {}
            for r in bn[ds]:
                by_n.setdefault(int(r["num_exits"]), []).append(r)
            cand = [(n, _mean([r["accuracy"] for r in rs]), _mean([r["savings"] for r in rs]))
                    for n, rs in by_n.items() if n >= 1]
            if cand:
                n_best, a_best, s_best = max(cand, key=lambda t: t[1])
                bn_cell = f"{a_best:.1f} ({s_best:.1f}) @{n_best}ex"
        row["BranchyNet (comp)"] = bn_cell

        # NonMyopic-CNN / TEASER-CNN (sensing)
        row["NonMyopic-CNN (sens)"] = (
            _cell(_mean([r["accuracy"] for r in nm[ds]]), _mean([r["savings"] for r in nm[ds]]))
            if ds in nm else "-")
        row["TEASER-CNN (sens)"] = (
            _cell(_mean([r["accuracy"] for r in te[ds]]), _mean([r["savings"] for r in te[ds]]))
            if ds in te else "-")

        table.append(row)
    return table


def main():
    ap = argparse.ArgumentParser(description="Assemble the CNN baseline paper table")
    ap.add_argument("--outdir", default=str(mc.HERE / "results_cnn"))
    ap.add_argument("--iso-eps", type=float, default=1.0,
                    help="iso-accuracy tolerance (pts) for the Truncation-CNN operating point")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    table = build(outdir, args.iso_eps)

    cols = ["dataset", "Baseline-CNN", "BranchyNet (comp)", "Truncation-CNN (sens)",
            "NonMyopic-CNN (sens)", "TEASER-CNN (sens)"]
    pcsv = outdir / "CNN_PAPER_TABLE.csv"
    with open(pcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(table)

    pmd = outdir / "CNN_PAPER_TABLE.md"
    with open(pmd, "w") as fh:
        fh.write("# CNN baseline comparison (backbone = CNN1D_extended = paper 'Model 1 CNN')\n\n")
        fh.write("Each cell: `accuracy% (savings%)`, mean over seeds 42-46. Savings axis: "
                 "BranchyNet=COMPUTE, Truncation/NonMyopic/TEASER=SENSING, Baseline=none. "
                 "SEEN (proposed) column to be filled from SEEN runs.\n\n")
        fh.write("| " + " | ".join(cols) + " | SEEN (proposed) |\n")
        fh.write("|" + "|".join(["---"] * (len(cols) + 1)) + "|\n")
        for r in table:
            fh.write("| " + " | ".join(r.get(c, "-") for c in cols) + " | _fill_ |\n")

    # LaTeX (booktabs) -- paste directly into Overleaf. `_esc` guards the (%) cells.
    def _esc(s):
        return str(s).replace("%", "\\%").replace("_", "\\_")
    hdr = ["Dataset", "Baseline-CNN", "BranchyNet", "Truncation-CNN",
           "NonMyopic-CNN", "TEASER-CNN", "SEEN"]
    ptex = outdir / "CNN_PAPER_TABLE.tex"
    with open(ptex, "w") as fh:
        fh.write("% CNN baseline comparison. Backbone = CNN1D_extended (paper 'Model 1 CNN').\n")
        fh.write("% Cells: accuracy (savings). BranchyNet savings = COMPUTE; "
                 "Truncation/NonMyopic/TEASER = SENSING. Fill SEEN from the proposed-method runs.\n")
        fh.write("\\begin{table}[t]\n\\centering\n")
        fh.write("\\caption{CNN baseline comparison (accuracy \\% (energy savings \\%)), "
                 "seeds 42--46. BranchyNet savings are compute; Truncation/NonMyopic/TEASER are sensing.}\n")
        fh.write("\\label{tab:cnn_baselines}\n")
        fh.write("\\begin{tabular}{l" + "c" * (len(hdr) - 1) + "}\n\\toprule\n")
        fh.write(" & ".join(hdr) + " \\\\\n\\midrule\n")
        for r in table:
            cells = [r["dataset"], r["Baseline-CNN"], r["BranchyNet (comp)"],
                     r["Truncation-CNN (sens)"], r["NonMyopic-CNN (sens)"],
                     r["TEASER-CNN (sens)"]]
            fh.write(" & ".join(_esc(c) for c in cells) + " & TBD \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"wrote {pcsv}\nwrote {pmd}\nwrote {ptex}\n")
    print("| " + " | ".join(cols) + " |")
    for r in table:
        print("| " + " | ".join(r.get(c, "-") for c in cols) + " |")


if __name__ == "__main__":
    main()
