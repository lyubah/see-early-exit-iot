"""
process_power_energy.py -- turn a folder of board runs into per-run latency / accuracy / energy.

Pairs every run's latency CSV (written by Main_board_cnn.py: t_start, t1..t4, total,
true_label, prediction, correctness, exit_taken, data%) with its power trace
(<Dataset>_<Backbone>_<Variant>_power.csv, or <Dataset>_shared_Power.csv as written by
data_logger.py: wall-clock time, power mW)
and integrates measured power over each window's active interval
[t_start, t_start + total]  ->  energy in mJ (mW x s).

Usage:
    python3 process_power_energy.py                      # reads ./board_results
    python3 process_power_energy.py --results_dir DIR --out energy_summary.csv
    python3 process_power_energy.py --active_power_mw 900   # fallback when no _power.csv

Output: one row per run: mean/std exec time (ms), accuracy, mean energy (mJ), exit usage.
"""
import argparse, csv, glob, os
import numpy as np

DATASETS = {"Epilepsy", "Shoaib", "PAMAP2", "WESADchest", "EMGPhysical", "SelfRegulationSCP1"}


def read_latency_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t_start": float(r["t_start"]),
                    "total": float(r["total"]),
                    "correct": str(r["correctness"]).strip().lower() == "true",
                    "exit_taken": int(float(r["exit_taken"])),
                    "data_pct": float(r.get("data%", 100) or 100),
                })
            except (KeyError, ValueError):
                continue
    return rows


def read_power_csv(path):
    t, p = [], []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 2:
                continue
            try:
                t.append(float(r[0])); p.append(float(r[1]))
            except ValueError:
                continue  # header line
    return np.asarray(t), np.asarray(p)


def window_energy_mj(t0, dur_s, pt, pp):
    """Integral of power (mW) over [t0, t0+dur_s] seconds -> mJ."""
    if dur_s <= 0 or len(pt) < 2:
        return np.nan
    mask = (pt >= t0) & (pt <= t0 + dur_s)
    if mask.sum() >= 2:
        _integrate = getattr(np, "trapezoid", None) or np.trapz   # numpy >= 2.0 renamed trapz
        return float(_integrate(pp[mask], pt[mask]))
    # window shorter than the sampling gap: nearest-sample power x duration
    i = int(np.argmin(np.abs(pt - (t0 + dur_s / 2.0))))
    if abs(pt[i] - t0) > 5.0:
        return np.nan  # power log doesn't cover this run
    return float(pp[i] * dur_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="board_results")
    ap.add_argument("--out", default="energy_summary.csv")
    ap.add_argument("--active_power_mw", type=float, default=None,
                    help="fallback estimate when a run has no _power.csv")
    a = ap.parse_args()

    out_rows = []
    for path in sorted(glob.glob(os.path.join(a.results_dir, "*.csv"))):
        base = os.path.basename(path)
        if base.endswith("_power.csv") or base == os.path.basename(a.out):
            continue
        stem = base[:-4] if base.lower().endswith(".csv") else base
        parts = stem.split("_")
        if len(parts) >= 5 and parts[1] == "accuracy" and parts[2] == "results":
            ds, backbone, variant = parts[0], parts[3], parts[4]   # <ds>_accuracy_results_<bb>_<var>.csv (Main_board_cnn.py)
        elif len(parts) >= 3:
            ds, backbone, variant = parts[0], parts[1], parts[2]   # legacy <ds>_<bb>_<var>.csv
        else:
            continue
        if ds not in DATASETS:
            continue
        rows = read_latency_csv(path)
        if not rows:
            print(f"[skip] {base}: no parsable latency rows")
            continue

        ms = np.array([r["total"] * 1000.0 for r in rows])
        acc = 100.0 * np.mean([r["correct"] for r in rows])
        dpct = float(np.mean([r["data_pct"] for r in rows]))
        exits = {}
        for r in rows:
            exits[r["exit_taken"]] = exits.get(r["exit_taken"], 0) + 1
        exit_hist = " ".join(f"e{k}:{v}" for k, v in sorted(exits.items()))

        power_path = os.path.join(a.results_dir, f"{ds}_{backbone}_{variant}_power.csv")
        if not os.path.exists(power_path):
            # data_logger.py's default output name
            alt = os.path.join(a.results_dir, f"{ds}_shared_Power.csv")
            if os.path.exists(alt):
                power_path = alt
        if os.path.exists(power_path):
            pt, pp = read_power_csv(power_path)
            e = np.array([window_energy_mj(r["t_start"], r["total"], pt, pp) for r in rows])
            energy, src = float(np.nanmean(e)), "INA219"
            if np.all(np.isnan(e)):
                energy, src = np.nan, "power-log-mismatch"
        elif a.active_power_mw:
            energy, src = float(ms.mean() * a.active_power_mw / 1000.0), f"estimate@{a.active_power_mw:.0f}mW"
        else:
            energy, src = np.nan, "no-power-log"

        out_rows.append([ds, backbone, variant, len(rows), f"{ms.mean():.2f}", f"{ms.std():.2f}",
                         f"{acc:.1f}", f"{dpct:.0f}", f"{energy:.2f}", src, exit_hist])
        print(f"{ds:20s} {backbone:8s} {variant:9s}  {ms.mean():7.2f} ms +/- {ms.std():5.2f}   "
              f"acc {acc:5.1f}%   energy {energy:8.2f} mJ  [{src}]  {exit_hist}")

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Backbone", "Variant", "windows", "mean_ms", "std_ms",
                    "acc_pct", "mean_data_pct", "energy_mJ", "energy_source", "exit_hist"])
        w.writerows(out_rows)
    print(f"\n[done] wrote {a.out}  ({len(out_rows)} runs)")


if __name__ == "__main__":
    main()
