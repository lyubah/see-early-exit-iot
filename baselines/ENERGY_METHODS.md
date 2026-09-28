# Energy-savings accounting

## Total-energy model
We score every method on a single **total-energy** axis so that early-exit
(sensing-reducing) and adaptive-inference (compute-reducing) methods are directly
comparable. The per-inference energy budget is split into a **sensing** slice
(acquiring the input window) and a **compute** slice (running the classifier):

  E_total = E_sense + E_compute,   s := E_sense / E_total

Embedded sensing systems are sensing-dominated; we take s = 0.70 (sensing is
60–80% of the budget in our platform). A method that senses a fraction `w ∈ [0,1]`
of the window and executes a fraction `c ∈ [0,1]` of the model uses normalized
energy

  e = s·w + (1 − s)·c,

and its **energy savings** is

  E_save = (1 − e)·100% = (1 − [s·w + (1 − s)·c])·100%.

The fractions are computed **per test window** (with `w` measured in real
timesteps, after the verified triaxial layout correction) and averaged over the
test set; we then report mean ± standard deviation over 5 seeds.

## Per-method instantiation
- **Static truncation, TEASER (early-exit/sensing methods, compute scales):** these
  shorten the sensing window (`w < 1`) and the compute scales with the truncated
  input (`c ≈ w`), so `e ≈ w` and `E_save ≈ (1 − w)·100%`. For these methods we
  report the realized sensing-window savings, `1 − w`, which equals the
  total-energy savings under `c ≈ w`. `w` is the realized fraction sensed:
  `w = t_exit / T` for TEASER (the per-window decision time) and
  `w = P/100` for static truncation.
- **Non-Myopic (early-exit/sensing method, compute does NOT scale):** Non-Myopic
  shortens the sensing window (`w = t_exit / T < 1`) but its base classifier (RF or
  1-NN) still executes in **full** on the prefix — tree-inference cost is set by
  `n_estimators × depth` (and 1-NN by the train-set size), **not** by window length —
  so `c ≈ 1`, not `c ≈ w`. Substituting `c = 1`:

  E_save(Non-Myopic) = (1 − [s·w + (1 − s)·1])·100% = s·(1 − w)·100% = **s × (sensing savings)**.

  With `s = 0.70`, Non-Myopic's total-energy savings is `0.70 ×` its realized
  sensing-window savings, and is upper-bounded by `s = 70%` — it cannot save the
  compute slice. Reporting raw sensing savings (e.g. 88% on WESAD) would overstate
  the system-level benefit; on the total-energy axis it is 0.70×88% ≈ 61.6%. This is
  the sensing-side mirror of the EDEN `(1 − s)` compute discount below.
- **Adaptive RF (EDEN, compute method):** EDEN reads the **full** window
  (`w = 1`, so sensing savings = 0) and reduces only the model by stopping early
  in the tree ensemble, `c = (mean trees evaluated) / N_trees`. Substituting
  `w = 1`:

  E_save(EDEN) = (1 − s)·(1 − c)·100% = (1 − s)·(compute savings).

  With `s = 0.70`, EDEN's total-energy savings equals **0.30 × its compute
  savings**, and is therefore upper-bounded by `(1 − s) = 30%` — it cannot save
  any of the sensing slice. Reporting raw compute savings (e.g. 86% on PAMAP2)
  would overstate EDEN's system-level benefit; on the total-energy axis it is
  25.8%. We therefore report total-energy savings for all methods.

## Multi-seed reporting
Each baseline is run over 5 seeds [42–46]; every seed independently varies the
60/40 train/test split (`random_state`) and the model `random_state`.
We report weighted Precision/Recall/F1 (so weighted recall equals accuracy) and
sample standard deviation (ddof = 1). All methods share the same split for a given
seed, so the results are directly comparable.

> Sensitivity note: `s` is a platform constant. The qualitative ordering is
> insensitive to `s ∈ [0.6, 0.8]`; only EDEN's ceiling moves (40% → 20%).
