#!/usr/bin/env python3
"""Energy from a single INA219 power log (data_logger.py output).

    python compute_energy.py <power_log.csv>
        -> total energy (mJ / J), duration, mean power for the whole log
    python compute_energy.py <power_log.csv> --latency_csv <ds>_accuracy_results_<bb>_<var>.csv
        -> also per-window energy: integrates power over each window's
           [t_start, t_start + total] interval from the Main_board_cnn.py output

To process a whole folder of runs at once (one row per run, latency + power paired
automatically) use process_power_energy.py instead:
    python process_power_energy.py --results_dir board_results
"""
import argparse
import numpy as np
from process_power_energy import read_power_csv, read_latency_csv, window_energy_mj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("power_log", help="CSV from data_logger.py: time s, power mW")
    ap.add_argument("--latency_csv", help="per-window latency CSV written by Main_board_cnn.py")
    a = ap.parse_args()

    t, p = read_power_csv(a.power_log)
    if len(t) < 2:
        raise SystemExit(f"{a.power_log}: fewer than 2 power samples")
    integrate = getattr(np, "trapezoid", None) or np.trapz
    total_mj = float(integrate(p, t))
    dur = float(t[-1] - t[0])
    print(f"samples          : {len(t)}")
    print(f"duration         : {dur:.3f} s (mean sample interval {dur / (len(t) - 1) * 1000:.2f} ms)")
    print(f"mean power       : {p.mean():.1f} mW")
    print(f"TOTAL ENERGY     : {total_mj:.1f} mJ  ({total_mj / 1000.0:.3f} J)")

    if a.latency_csv:
        rows = read_latency_csv(a.latency_csv)
        e = np.array([window_energy_mj(r["t_start"], r["total"], t, p) for r in rows])
        ok = e[~np.isnan(e)]
        print(f"windows          : {len(rows)} ({len(ok)} covered by the power log)")
        if len(ok):
            print(f"PER-WINDOW ENERGY: {ok.mean():.3f} mJ mean, {ok.std():.3f} mJ std")
        else:
            print("PER-WINDOW ENERGY: none computable -- power log does not overlap the run timestamps")


if __name__ == "__main__":
    main()
