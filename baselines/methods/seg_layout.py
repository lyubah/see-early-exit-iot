#!/usr/bin/env python3
"""
seg_layout.py
=============
Single source of truth for how the raw SEG_SIZE (timestamp) axis of each dataset
is laid out in memory, and for turning a raw ``(N, C, SEG_SIZE)`` array into the
canonical time-major view ``(N, T, F)`` (axes grouped per timestep -> xyz_1, xyz_2,
...). Every loader that flattens windows (SEE-hRF ``ReadFile.py``, the SEEN CNN
pipeline, and the comparison baselines) imports from here, so they are GUARANTEED to
use the identical layout -- comparisons stay apples-to-apples.

Why this module exists
----------------------
For "triaxial" datasets (Shoaib, PAMAP2) the 3 axes (x/y/z) are packed *inside* the
SEG_SIZE axis, SEG_SIZE = 3 * T. There are two ways that can be packed:

    axis_major : [ x_1..x_T , y_1..y_T , z_1..z_T ]        (3 blocks; the OLD assumption)
    interleaved: [ x_1 y_1 z_1 , x_2 y_2 z_2 , ... ]       (per-timestep groups)

and some datasets whose SEG_SIZE merely happens to be divisible by 3 are not packed
at all:

    time       : SEG_SIZE IS the real time axis (not triaxial)

The old code hard-coded the ``axis_major`` reshape ``reshape(N, C, 3, T)`` for
Shoaib/PAMAP2 *without verifying it*. If the raw is actually ``interleaved`` (or
``time``) that reshape SCRAMBLES time, so the static-truncation baseline's "first
P% of timesteps" silently samples across the WHOLE window -> temporal leakage ->
truncation looks far too good (PAMAP2 was ~97.6% from just 10% of the window).

Rather than swap one unverified guess for another, this module DETECTS the layout
from the data (temporal-autocorrelation + variance fingerprints), VERIFIES it, and
fails loudly if the signal is ambiguous (so a wrong guess can never pass silently).
You can pin a layout explicitly via the ``LAYOUT_OVERRIDE`` dict or the
``SEG_LAYOUT_<DATASET>`` environment variable once you've confirmed it.
"""

import os
import numpy as np

LAYOUTS = ("axis_major", "interleaved", "time")

# Explicit per-dataset pin (highest precedence after env var). Fill in once the
# layout is confirmed, e.g. {"PAMAP2": "interleaved"}.
# Leave empty to auto-detect.
#
# PINNED: PAMAP2 & Shoaib were confirmed AXIS_MAJOR on the preprocessed
# pickles via the autocorrelation fingerprint (PAMAP2 join_lag1 .962->-.417,
# thirds_cov .32; Shoaib .943->-.441, thirds_cov .30; truncation drops sensibly).
# Pinning makes the layout deterministic + fail-safe for the medical runs so a
# data refresh can never silently re-detect a different (scrambling) layout.
# Re-confirm with detect_seg_layout() before changing. Non-triaxial datasets are unaffected (still auto-detect -> "time").
LAYOUT_OVERRIDE = {"PAMAP2": "axis_major", "Shoaib": "axis_major"}

# Detection sensitivity. If no layout wins by at least these margins the resolver
# RAISES (fail-closed) instead of guessing -- you then pin it via override.
_INTERLEAVE_MARGIN = 0.05    # lag3 must beat lag1 by this to call "interleaved"
_BOUNDARY_MARGIN = 0.12      # interior lag1 minus across-block-join lag1 to call "axis_major"
_THIRDS_COV = 0.25           # CoV of the 3 contiguous-thirds variances (axis_major hint)
_PERIOD3_COV = 0.25          # CoV of the 3 phase variances (interleaved hint)

_cache = {}                  # name -> (layout, info)


# ── fingerprints ─────────────────────────────────────────────────────────────

def _zscore_rows(seqs):
    s = seqs - seqs.mean(axis=1, keepdims=True)
    std = s.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return s / std


def _autocorr(seqs, max_lag=6):
    """Mean normalized autocorrelation at lags 0..max_lag over rows (each z-scored)."""
    M, L = seqs.shape
    s = seqs - seqs.mean(axis=1, keepdims=True)
    denom = (s * s).sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    out = np.zeros(max_lag + 1)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        num = (s[:, : L - lag] * s[:, lag:]).sum(axis=1, keepdims=True)
        out[lag] = float(np.mean(num / denom))
    return out


def _boundary_drop(seqs, L):
    """axis_major => 3 concatenated blocks of length T=L/3; lag-1 correlation stays
    high inside a block but collapses across the two joins (T-1->T, 2T-1->2T).
    Returns (interior_lag1, boundary_lag1) or None if L not divisible by 3."""
    if L % 3 != 0:
        return None
    T = L // 3
    s = _zscore_rows(seqs)
    interior = [i for i in range(L - 1) if (i + 1) % T != 0]
    joins = [T - 1, 2 * T - 1]
    a = float(np.mean(s[:, interior] * s[:, [i + 1 for i in interior]]))
    b = float(np.mean(s[:, joins] * s[:, [i + 1 for i in joins]]))
    return a, b


def _phase3_cov(seqs, L):
    """interleaved => positions {0,3,..},{1,4,..},{2,5,..} are 3 different axes with
    typically different variance. CoV of the 3 phase variances (high => interleaved)."""
    if L % 3 != 0:
        return None
    v = [float(seqs[:, p::3].var()) for p in range(3)]
    m = np.mean(v)
    return (np.std(v) / m) if m else 0.0


def _thirds_cov(seqs, L):
    """axis_major => the 3 contiguous thirds are different axes with different
    variance. CoV of the 3 block variances (high => axis_major)."""
    if L % 3 != 0:
        return None
    T = L // 3
    v = [float(seqs[:, b * T:(b + 1) * T].var()) for b in range(3)]
    m = np.mean(v)
    return (np.std(v) / m) if m else 0.0


# ── detection ────────────────────────────────────────────────────────────────

def detect_seg_layout(X, name="<data>", max_windows=400):
    """Decide the SEG_SIZE layout of raw X (N, C, L) from the data itself.

    Returns (layout, info). layout in LAYOUTS. info carries the fingerprint values
    and a human-readable reason. Raises ValueError if L%3==0 but no layout wins by
    the configured margins (ambiguous -> pin via override)."""
    N, C, L = X.shape
    if L % 3 != 0:
        return "time", {"reason": f"SEG_SIZE={L} not divisible by 3 -> cannot pack 3 axes",
                        "L": L, "divisible_by_3": False}

    rng = np.random.RandomState(0)
    idx = rng.choice(N, size=min(N, max_windows), replace=False)
    seqs = X[idx].reshape(len(idx) * C, L).astype(np.float64)

    ac = _autocorr(seqs, max_lag=6)
    ac1, ac3 = ac[1], ac[3]
    interior, boundary = _boundary_drop(seqs, L)
    p3 = _phase3_cov(seqs, L)
    th = _thirds_cov(seqs, L)
    info = {"L": L, "T": L // 3, "ac1": round(ac1, 4), "ac3": round(ac3, 4),
            "lag1_interior": round(interior, 4), "lag1_join": round(boundary, 4),
            "phase3_cov": round(p3, 4), "thirds_cov": round(th, 4),
            "divisible_by_3": True}

    interleaved = (ac3 - ac1 > _INTERLEAVE_MARGIN) or (p3 > _PERIOD3_COV and ac3 > ac1)
    axis_major = (interior - boundary > _BOUNDARY_MARGIN) or (th > _THIRDS_COV and ac1 >= ac3)
    pure_time = (ac1 > 0.3 and ac3 <= ac1 and (interior - boundary) <= _BOUNDARY_MARGIN
                 and th <= _THIRDS_COV and p3 <= _PERIOD3_COV)

    # Resolve, preferring the strongest unambiguous signal.
    if interleaved and not axis_major:
        info["reason"] = f"lag3({ac3:.3f}) > lag1({ac1:.3f}) and/or phase-3 CoV {p3:.3f} -> xyz interleaved"
        return "interleaved", info
    if axis_major and not interleaved:
        info["reason"] = (f"lag1 collapses at block joins ({interior:.3f}->{boundary:.3f}) "
                          f"and/or thirds CoV {th:.3f} -> axis-major blocks")
        return "axis_major", info
    if pure_time and not interleaved and not axis_major:
        info["reason"] = f"smooth lag1-dominant ({ac1:.3f}), no period-3, no block joins -> real time axis"
        return "time", info

    # Ambiguous or conflicting signals -> do NOT guess.
    raise ValueError(
        f"seg_layout: AMBIGUOUS layout for {name} (L={L}). "
        f"Fingerprints: ac1={ac1:.3f} ac3={ac3:.3f} interior_lag1={interior:.3f} "
        f"join_lag1={boundary:.3f} phase3_cov={p3:.3f} thirds_cov={th:.3f}. "
        f"Inspect the data and pin the layout via "
        f"LAYOUT_OVERRIDE['{name}']='axis_major'|'interleaved'|'time' or env SEG_LAYOUT_{name.upper()}=...")


def _override_for(name):
    env = os.environ.get(f"SEG_LAYOUT_{str(name).upper()}")
    if env:
        if env not in LAYOUTS:
            raise ValueError(f"SEG_LAYOUT_{name.upper()}={env!r} not in {LAYOUTS}")
        return env, "env"
    if name in LAYOUT_OVERRIDE:
        v = LAYOUT_OVERRIDE[name]
        if v not in LAYOUTS:
            raise ValueError(f"LAYOUT_OVERRIDE[{name!r}]={v!r} not in {LAYOUTS}")
        return v, "override"
    return None, None


def resolve_seg_layout(X, name, verbose=True):
    """Return the layout for (X, name): env var > LAYOUT_OVERRIDE > auto-detect.
    Caches per name and logs the decision ONCE so it shows up in run logs."""
    if name in _cache:
        return _cache[name][0]
    forced, src = _override_for(name)
    if forced is not None:
        layout, info = forced, {"reason": f"pinned via {src}"}
    else:
        layout, info = detect_seg_layout(X, name)
    _cache[name] = (layout, info)
    if verbose:
        print(f"[seg_layout] {name}: SEG_SIZE layout = {layout.upper()}  ({info.get('reason','')})")
    return layout


def clear_cache():
    _cache.clear()


# ── canonical views ──────────────────────────────────────────────────────────

def temporal_view(X, name, verbose=True):
    """Raw (N, C, L) -> canonical time-major (N, T, F); axis 1 = earliest..latest
    real time, so slicing axis 1 is a correct contiguous temporal truncation, and
    flattening (reshape(N,-1)) yields per-timestep groups (xyz_1, xyz_2, ...)."""
    N, C, L = X.shape
    layout = resolve_seg_layout(X, name, verbose=verbose)
    if layout == "time":
        return np.swapaxes(X, 1, 2)                                   # (N, L, C)
    T = L // 3
    if layout == "axis_major":      # [x_1..x_T, y_1..y_T, z_1..z_T]
        return X.reshape(N, C, 3, T).transpose(0, 3, 1, 2).reshape(N, T, C * 3)
    if layout == "interleaved":     # [x_1 y_1 z_1, x_2 y_2 z_2, ...]
        return X.reshape(N, C, T, 3).transpose(0, 2, 1, 3).reshape(N, T, C * 3)
    raise ValueError(f"unknown layout {layout!r} for {name}")


def flatten_full(X, name, verbose=True):
    """Full-window time-major flatten -> (N, F_total)."""
    v = temporal_view(X, name, verbose=verbose)
    return v.reshape(v.shape[0], -1)


def truncate_flatten(X, name, pct, verbose=True):
    """Keep the first ``pct`` percent of REAL timesteps, then flatten.
    Returns (X_flat, cut) with cut = timesteps kept. Truncates real time T, never
    the packed 3*T axis."""
    v = temporal_view(X, name, verbose=verbose)
    T = v.shape[1]
    cut = T if pct >= 100 else max(1, int(T * pct / 100))
    return v[:, :cut, :].reshape(v.shape[0], -1), cut
