#!/usr/bin/env python3
"""
run_nonmyopic_multiseed.py
==========================
Multi-seed Non-Myopic baseline for the paper table, built on tslearn's
NonMyopicEarlyClassifier. Run with ONE fixed rule applied identically to every
dataset -- the same discipline as the other baselines (TEASER, truncation). No
alpha tuning, no validation split, no per-dataset selection, no peeking at
accuracy/savings to pick anything.

PINNED DESIGN (all enforced here):
  * Base classifier : RandomForest 50x30 for EVERY dataset (forced in `make_base`,
    for every dataset).
  * Split           : the shared 60/40 split on the corrected time-major view
    (bd.temporal_view, triaxial-correct); split random_state = seed.
  * WESAD           : split like every other dataset (mseed_common.load_split),
    identical to TEASER/EDEN.
  * Seeds           : 42-46 (split random_state == model random_state == seed).
  * n_clusters      : per-dataset, from nonmyopic_config.CONFIG['nc'] (structural
    cluster count, with the documented small-cluster fallback). This is the only
    remaining per-dataset hyperparameter; it is NOT chosen from accuracy/savings.

TIME-COST RULE (FIXED, single constant; NOT tuned on accuracy/savings):
  tslearn's cost-of-time is f(t) = cost_time_parameter * t (verified in 0.8.1
  source: _cost_time, predict-only). To apply ONE identical rule across datasets
  with different window lengths T, we hold a single constant c fixed for all six
  datasets and set

        cost_time_parameter (alpha) = c / T          (T = dataset window length)

  so the time penalty for waiting the FULL window, alpha * T = c, is identical
  everywhere.

  c = 0.02. CORRECTNESS FLOOR, not a performance tune: tslearn's docstring regime
  (cost_time_parameter=0.1 on length-6 series -> c = alpha*T = 0.6) is too large
  for these datasets -- at c~0.6 the time penalty dominates and NonMyopic COLLAPSES
  to a single fixed exit (every test window stops at the same early timestep,
  exit_varies=False), i.e. it stops behaving as a per-sample early classifier at
  all. We therefore use c=0.02, which keeps NonMyopic in its ADAPTIVE regime
  (per-sample-varying exit times). c was checked ONLY against the exit_varies
  correctness signal (is the method functioning?) -- NEVER against accuracy or
  savings. Same c across all six datasets.

  This is still an UNTUNED operating point and may give modest savings. That is
  intended -- apples-to-apples against the other fixed-rule baselines; its
  performance is NOT to be "improved."

Sensing savings (same definition as SEE-RF):
    savings = mean_TEST(1 - t_decision / T_real) * 100,  t in REAL timesteps.
  tslearn returns the raw decision TIMESTAMP t (predict_class_and_earliness); it has
  NO native savings metric (its only native scorer is early_classification_cost =
  (1-acc) + alpha*mean(t)). The 1 - t/T normalization is ours, a monotone transform
  of tslearn's own t -- the decision logic itself is untouched.

Outputs (results/):
  NONMYOPIC_perseed__<dataset>.csv   canonical 7 cols (matched schema)
"""
import argparse
from pathlib import Path
import numpy as np

import mseed_common as mc
import baseline_data as bd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tslearn.early_classification import NonMyopicEarlyClassifier
from tslearn.utils import to_time_series_dataset

# n_clusters is taken from nonmyopic_config; FlatWrap is the
# (n, t, F) -> (n, t*F) flattening wrapper around the RF base. alpha is NOT imported --
# it is the fixed c/T rule below.
from nonmyopic_config import CONFIG, FlatWrap

# Fixed time-cost constant. c=0.02 is a CORRECTNESS FLOOR: the tslearn docstring
# regime (c~0.6) collapses NonMyopic to a single fixed exit (exit_varies=False) on
# these datasets; c=0.02 keeps it adaptive (per-sample-varying exits). Chosen ONLY
# against exit_varies, NEVER against accuracy/savings. Held identical for all six
# datasets; alpha = C_TIME / T per dataset.
C_TIME = 0.02


def make_base(seed, base="rf5030"):
    """Base classifier, wrapped by FlatWrap to flatten (n, t, F) -> (n, t*F).
      rf5030 = RandomForest 50x30 (default, same size as every other RF baseline).
      1nn    = literal 1-NN (the paper's STATED base) -- lets you run the 1-NN
               version under THIS exact regime (fixed alpha=c/T, 60/40, one seed),
               i.e. apples-to-apples with the RF run, NOT the papermatch 60/20/20
               harmonic-mean-grid setup."""
    if base == "1nn":
        return FlatWrap(lambda: KNeighborsClassifier(n_neighbors=1))
    return FlatWrap(lambda: RandomForestClassifier(
        n_estimators=50, max_depth=30, random_state=seed))


def _split_6040_view(name, seed):
    """Shared 60/40 split (mseed_common.load_split) on the corrected time-major view
    (N, T_real, F)."""
    X, y, tr, te = mc.load_split(name, seed)
    V = bd.temporal_view(X, name)
    return V[tr], y[tr], V[te], y[te]


def _savings(dt, T):
    dt = np.clip(np.asarray(dt).astype(int), 1, T)
    return float(np.mean(1.0 - dt / float(T))) * 100.0


def _fit_nonmyopic(Vtr, ytr, nc_pref, alpha, seed, base="rf5030"):
    """Fit NonMyopic at the FIXED alpha = c/T. tslearn's fit stratifies an internal
    50/50 split on the cluster labels, which fails if a TimeSeriesKMeans cluster gets
    <2 members (small train + higher nc). Fall back to fewer clusters for that seed,
    returning nc_used. (alpha is predict-only in tslearn, so the value passed here is
    irrelevant to fit; we set it anyway for clarity.)"""
    Xts = to_time_series_dataset(Vtr)
    for nc in range(nc_pref, 0, -1):
        try:
            c = NonMyopicEarlyClassifier(
                base_classifier=make_base(seed, base), n_clusters=nc,
                cost_time_parameter=alpha, random_state=seed)
            c.fit(Xts, ytr)
            return c, nc
        except ValueError as e:
            if ("least populated class" in str(e)
                    or "minimum number of groups" in str(e)) and nc > 1:
                continue
            raise
    raise RuntimeError("NonMyopic fit failed at all n_clusters")


def run_dataset(dataset, seeds, c_time, base="rf5030"):
    ds = bd.canonical_name(dataset)
    cfg = CONFIG[ds]                                  # n_clusters only
    print(f"\n=== NonMyopic  {ds}  base={base} nc={cfg['nc']}  "
          f"alpha = c/T  (c={c_time:g}, fixed; adaptive-regime floor) ===", flush=True)
    perseed_rows = []
    for seed in seeds:
        Vtr, ytr, Vte, yte = _split_6040_view(ds, seed)
        T = int(Vtr.shape[1])
        alpha = c_time / float(T)                     # the one fixed rule

        clf, nc_used = _fit_nonmyopic(Vtr, ytr, cfg["nc"], alpha, seed, base)
        clf.cost_time_parameter = alpha
        yp, dt = clf.predict_class_and_earliness(to_time_series_dataset(Vte))

        acc = accuracy_score(yte, yp)
        m = bd.prf(yte, yp)                           # weighted P/R/F1 (0-1)
        savings = _savings(dt, T)
        exits_vary = len(np.unique(np.clip(np.asarray(dt).astype(int), 1, T))) > 1
        perseed_rows.append(mc.make_row(
            ds, seed, acc, m["precision_weighted"], m["recall_weighted"],
            m["f1_weighted"], savings))
        if nc_used != cfg["nc"]:
            print(f"  [note] seed={seed}: n_clusters {cfg['nc']}->{nc_used} "
                  f"(a cluster was too small to stratify)", flush=True)
        print(f"  seed={seed}  alpha={alpha:.3g} (c/T, T={T})  acc={acc*100:5.1f}%  "
              f"sav={savings:5.1f}%  F1w={m['f1_weighted']:.3f}  n_test={len(yte)}  "
              f"exit_varies={exits_vary}", flush=True)

    mc.write_perseed("NONMYOPIC", perseed_rows, dataset=ds)
    return perseed_rows


def main():
    ap = argparse.ArgumentParser(
        description="Multi-seed Non-Myopic (RF 50x30; fixed alpha = c/T, untuned)")
    ap.add_argument("--datasets", nargs="*", default=mc.DATASET_ORDER)
    ap.add_argument("--dataset", default="WESADchest",
                    help="single dataset (SLURM-array convenience; overrides --datasets)")
    ap.add_argument("--seeds", nargs="*", type=int, default=mc.SEEDS)
    ap.add_argument("--c-time", type=float, default=C_TIME,
                    help="fixed time-cost constant c (alpha=c/T). Default 0.02 "
                         "(adaptive-regime correctness floor); do NOT tune on results.")
    ap.add_argument("--base", default="rf5030", choices=["rf5030", "1nn"],
                    help="base classifier: rf5030 (canonical RF 50x30, default) or "
                         "1nn (literal 1-NN, the paper's stated base). Both run under "
                         "the SAME fixed alpha=c/T, 60/40, single-seed regime.")
    ap.add_argument("--outdir", default=None,
                    help="write per-seed CSVs here instead of the canonical results/ "
                         "dir. USE THIS for ad-hoc 1nn / c-time experiments so you do "
                         "NOT overwrite the canonical results/NONMYOPIC_perseed__*.csv.")
    args = ap.parse_args()
    if args.outdir:
        mc.RESULTS_DIR = Path(args.outdir)
        print(f"[outdir] writing per-seed CSVs to {mc.RESULTS_DIR} "
              f"(canonical results/ left untouched)", flush=True)
    datasets = [args.dataset] if args.dataset else args.datasets
    print(f"NonMyopic multi-seed (base={args.base}, fixed alpha=c/T, c={args.c_time:g}) | "
          f"seeds={args.seeds} | datasets={datasets}", flush=True)
    for ds in datasets:
        run_dataset(ds, args.seeds, args.c_time, args.base)
    print("\nDONE. Per-dataset NONMYOPIC_perseed__*.csv written to results/.")


if __name__ == "__main__":
    main()
