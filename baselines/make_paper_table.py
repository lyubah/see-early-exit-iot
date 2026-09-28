#!/usr/bin/env python3
"""
make_paper_table.py
===================
Turn the multi-seed per-seed CSVs into TWO copy/paste-ready artifacts:

  1) results/FINAL_acc_energy_table.{csv,tex,md}
       ONE table -- rows = datasets, columns = each method's Accuracy and Energy as
       `mean (std)` (the "parenthesis thing"). CSV for Sheets/Excel, .tex for the
       paper, .md for a quick look.

  2) results/barplot_data.csv
       Tidy/long, NUMERIC, one row per (method, dataset):
         method, method_label, dataset, energy_type,
         accuracy_mean, accuracy_std, energy_mean, energy_std
       -> drop-in for grouped bar plots of accuracy and energy (with error bars).

Energy axes differ by method (flagged in both outputs):
  TEASER / NonMyopic / Static-Truncation -> SENSING savings (cut the time window)
  Adaptive RF (EDEN)                     -> COMPUTE savings (full window, fewer trees)

Static truncation is a P%-sweep; it enters the single-cell table/barplot at ONE
operating point, --trunc-pct (default 50).

Usage:  python make_paper_table.py [--trunc-pct 50]
"""
import argparse
import csv

import mseed_common as mc

# table column order + energy semantics
SINGLE = [("TEASER", "sensing"), ("NONMYOPIC", "sensing"), ("EDEN", "compute")]
E_LABEL = {"sensing": "E_sens", "compute": "E_comp"}


def _stats(rows, col):
    xs = [float(r[col]) for r in rows]
    return mc._mean(xs), mc._stdev(xs)


def _row_stats(rows, etype, label):
    """All per-(method,dataset) stats: acc/energy (+P/R/F1) mean & std."""
    am, asd = _stats(rows, "accuracy")
    em, esd = _stats(rows, "savings")
    pm, psd = _stats(rows, "precision_weighted")
    rm, rsd = _stats(rows, "recall_weighted")
    fm, fsd = _stats(rows, "f1_weighted")
    return dict(acc_mean=am, acc_std=asd, e_mean=em, e_std=esd, e_type=etype, label=label,
                prec_mean=pm, prec_std=psd, rec_mean=rm, rec_std=rsd, f1_mean=fm, f1_std=fsd)


def collect(trunc_pct):
    """-> {method_key: {dataset: stats dict}} with acc/energy/P/R/F1 mean & std."""
    out = {}
    for mkey, etype in SINGLE:
        by = mc.load_perseed(mkey)
        out[mkey] = {ds: _row_stats(by[ds], etype, mc.METHOD_LABEL[mkey])
                     for ds in mc.DATASET_ORDER if by.get(ds)}
    # static truncation at the chosen operating point
    by_t = mc.load_trunc_perseed()
    out["TRUNCATION"] = {ds: _row_stats(by_t[(ds, trunc_pct)], "sensing",
                                        f"Static Trunc @{trunc_pct}%")
                         for ds in mc.DATASET_ORDER if by_t.get((ds, trunc_pct))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunc-pct", type=int, default=50,
                    help="static-truncation operating point for the single-cell table")
    args = ap.parse_args()
    pct = args.trunc_pct
    methods = ["TEASER", "NONMYOPIC", "EDEN", "TRUNCATION"]
    data = collect(pct)
    rd = mc.results_dir()

    def acc_cell(s):
        return f"{s['acc_mean']:.1f} ({s['acc_std']:.1f})"

    def e_cell(s):
        return f"{s['e_mean']:.1f} ({s['e_std']:.1f})"

    # ── 1a) wide CSV (parenthesis strings) ──────────────────────────────────
    hdr = ["Dataset"]
    for m in methods:
        lab = (f"StaticTrunc@{pct}%" if m == "TRUNCATION" else mc.METHOD_LABEL[m])
        etype = "compute" if m == "EDEN" else "sensing"
        hdr += [f"{lab} Acc%", f"{lab} Energy%({etype})"]
    csv_path = rd / "FINAL_acc_energy_table.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for ds in mc.DATASET_ORDER:
            row = [mc.DISPLAY[ds]]
            for m in methods:
                s = data[m].get(ds)
                row += ([acc_cell(s), e_cell(s)] if s else ["--", "--"])
            w.writerow(row)

    # ── 1b) LaTeX (booktabs, multicolumn per method) ────────────────────────
    tex = [
        "% FINAL_acc_energy_table.tex -- mean (std) over seeds 42-46 (std ddof=1).",
        "% Acc & Energy in %, 1 decimal. E_sens=sensing savings; E_comp=compute savings (EDEN).",
        r"\begin{tabular}{l" + "cc" * len(methods) + "}",
        r"\toprule",
        " & " + " & ".join(
            r"\multicolumn{2}{c}{%s}" % (f"Static Trunc @{pct}" + r"\%" if m == "TRUNCATION"
                                          else mc.METHOD_LABEL[m].replace("&", r"\&"))
            for m in methods) + r" \\",
        "Dataset & " + " & ".join(
            "Acc & " + (E_LABEL["compute"] if m == "EDEN" else E_LABEL["sensing"])
            for m in methods) + r" \\",
        r"\midrule",
    ]
    for ds in mc.DATASET_ORDER:
        cells = []
        for m in methods:
            s = data[m].get(ds)
            cells += ([acc_cell(s), e_cell(s)] if s else ["--", "--"])
        tex.append(f"{mc.DISPLAY[ds]} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    tex_path = rd / "FINAL_acc_energy_table.tex"
    tex_path.write_text("\n".join(tex) + "\n")

    # ── 1c) markdown preview ────────────────────────────────────────────────
    md = ["| Dataset | " + " | ".join(
        (f"StaticTrunc@{pct}% Acc | E" if m == "TRUNCATION"
         else f"{mc.METHOD_LABEL[m]} Acc | E") for m in methods) + " |"]
    md.append("|" + "---|" * (1 + 2 * len(methods)))
    for ds in mc.DATASET_ORDER:
        row = [mc.DISPLAY[ds]]
        for m in methods:
            s = data[m].get(ds)
            row += ([acc_cell(s), e_cell(s)] if s else ["--", "--"])
        md.append("| " + " | ".join(row) + " |")
    md_path = rd / "FINAL_acc_energy_table.md"
    md_path.write_text("\n".join(md) + "\n")

    # ── 2) tidy numeric barplot CSV ─────────────────────────────────────────
    bar_path = rd / "barplot_data.csv"
    with open(bar_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "method_label", "dataset", "energy_type",
                    "accuracy_mean", "accuracy_std", "energy_mean", "energy_std"])
        for m in methods:
            for ds in mc.DATASET_ORDER:
                s = data[m].get(ds)
                if not s:
                    continue
                w.writerow([m, s["label"], mc.DISPLAY[ds], s["e_type"],
                            f"{s['acc_mean']:.4f}", f"{s['acc_std']:.4f}",
                            f"{s['e_mean']:.4f}", f"{s['e_std']:.4f}"])

    # ── 3) classification-metrics table (Acc% / Prec / Recall / F1, weighted) ──
    def pct_cell(s):
        return f"{s['acc_mean']:.1f} ({s['acc_std']:.1f})"

    def dec_cell(mean, std):
        return f"{mean:.3f} ({std:.3f})"

    SUBM = [("Acc%", None), ("Prec", ("prec_mean", "prec_std")),
            ("Recall", ("rec_mean", "rec_std")), ("F1", ("f1_mean", "f1_std"))]

    def metric_cells(s):
        cells = []
        for _, keys in SUBM:
            cells.append(pct_cell(s) if keys is None else dec_cell(s[keys[0]], s[keys[1]]))
        return cells

    # 3a) wide CSV
    mhdr = ["Dataset"]
    for m in methods:
        lab = (f"StaticTrunc@{pct}%" if m == "TRUNCATION" else mc.METHOD_LABEL[m])
        mhdr += [f"{lab} {sub}" for sub, _ in SUBM]
    mcsv = rd / "FINAL_classification_table.csv"
    with open(mcsv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(mhdr)
        for ds in mc.DATASET_ORDER:
            row = [mc.DISPLAY[ds]]
            for m in methods:
                s = data[m].get(ds)
                row += (metric_cells(s) if s else ["--"] * len(SUBM))
            w.writerow(row)

    # 3b) LaTeX (multicolumn{4} per method)
    mtex = [
        "% FINAL_classification_table.tex -- mean (std) over seeds 42-46 (std ddof=1).",
        "% Acc in %, 1 decimal; Precision/Recall/F1 weighted, 3 decimals. recall_weighted == accuracy.",
        r"\begin{tabular}{l" + "cccc" * len(methods) + "}",
        r"\toprule",
        " & " + " & ".join(
            r"\multicolumn{4}{c}{%s}" % (f"Static Trunc @{pct}" + r"\%" if m == "TRUNCATION"
                                          else mc.METHOD_LABEL[m].replace("&", r"\&"))
            for m in methods) + r" \\",
        "Dataset & " + " & ".join(" & ".join(sub for sub, _ in SUBM).replace("Acc%", r"Acc\%")
                                  for m in methods) + r" \\",
        r"\midrule",
    ]
    for ds in mc.DATASET_ORDER:
        cells = []
        for m in methods:
            s = data[m].get(ds)
            cells += (metric_cells(s) if s else ["--"] * len(SUBM))
        mtex.append(f"{mc.DISPLAY[ds]} & " + " & ".join(cells) + r" \\")
    mtex += [r"\bottomrule", r"\end{tabular}"]
    mtex_path = rd / "FINAL_classification_table.tex"
    mtex_path.write_text("\n".join(mtex) + "\n")

    # 3c) markdown preview
    mmd = ["| Dataset | " + " | ".join(
        (f"StaticTrunc@{pct}% " if m == "TRUNCATION" else f"{mc.METHOD_LABEL[m]} ")
        + " | ".join(sub for sub, _ in SUBM) for m in methods) + " |"]
    mmd.append("|" + "---|" * (1 + len(SUBM) * len(methods)))
    for ds in mc.DATASET_ORDER:
        row = [mc.DISPLAY[ds]]
        for m in methods:
            s = data[m].get(ds)
            row += (metric_cells(s) if s else ["--"] * len(SUBM))
        mmd.append("| " + " | ".join(row) + " |")
    mmd_path = rd / "FINAL_classification_table.md"
    mmd_path.write_text("\n".join(mmd) + "\n")

    print("wrote:")
    for p in (csv_path, tex_path, md_path, bar_path, mcsv, mtex_path, mmd_path):
        print(f"  results/{p.name}")
    print("\n--- FINAL_classification_table.md (Acc% / Prec / Recall / F1, weighted) ---")
    print("\n".join(mmd))


if __name__ == "__main__":
    main()
