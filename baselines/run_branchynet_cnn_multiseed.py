#!/usr/bin/env python3
"""
run_branchynet_cnn_multiseed.py
===============================
BranchyNet (BN) baseline reproduced to the SEEN ESWEEK paper's protocol -- the "BN"
column of Table 3 (Model 1 CNN). BranchyNet = CNN1D_extended_EENN (EENN_functions.py:139),
the SEEN backbone WITHOUT sensor-aware partial sampling: it reads the FULL window and
exits early in DEPTH (§5.1.5: "utilizes full sensor data in a window at all exits...
unable to save sensor energy"). Savings axis is COMPUTE, not sensing.

PAPER PROTOCOL (Table 2 / §5.1.3 / §5.1.5 / §5.2):
  * Base CNN     = "Model 1 CNN" = 5 conv+maxpool+ReLU + 2 FC + softmax == CNN1D_extended.
  * Split        = 60/40 (§5.1.3), seeds 42-46 (matched sweep).
  * Report PER NUMBER OF EARLY EXITS n in {1,2,3,4} (Table 3 columns): a separate net is
    trained for each n, exits placed at the first n conv layers.
  * Loss weights = DECREASING first->last exit, in the paper's 1-4 range (§5.2, following
    [36]); linspace(4, 1, n+1).
  * Entropy threshold sweep = 0.1 .. 1.5 (Table 2). Operating point picked iso-accuracy on
    a 25% validation carve of train (most compute-savings within --iso-eps pts of the
    full-net val accuracy); frozen and scored on test.
  * Confidence = entropy of softmax at the exit (Eq. 4); exit if entropy < threshold (Alg. 1).

Reported per (dataset, seed, num_exits): overall accuracy (the BN column), exit-1 accuracy,
compute-savings, weighted P/R/F1. Full-net (no-exit) accuracy is also emitted as the
"Baseline" reference (Table 3 "Baseline" col) via num_exits=0.

Outputs (results_cnn/):  BRANCHYNET_perseed__<dataset>.csv  (adds `num_exits` column)
"""
import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import mseed_common as mc
import baseline_data as bd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(mc.SEEN_DIR))
from EENN_functions import CNN1D_extended_EENN     # noqa: E402

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EXITS = [1, 2, 3, 4]          # Table 3 columns
THRESH_LO, THRESH_HI = 0.1, 1.5   # Table 2 confidence-threshold range


def placement_for(n):
    """n early exits -> first n conv layers (1..n). Paper places early exits in the
    early/mid layers (§4.1.3 example: layers 1 and 2)."""
    return list(range(1, n + 1))


def decreasing_weights(n_ee):
    """DECREASING loss weights first->last exit, in the paper's 1-4 range (§5.2)."""
    return list(np.linspace(4.0, 1.0, n_ee + 1))       # n_ee early + 1 final


def compute_fractions(F, T, K, exit_layers):
    """{layer -> fraction of full-net MACs executed when exiting there}; layer 5 = full."""
    convs = [(F, 8), (8, 16), (16, 32), (32, 64), (64, 128)]
    L = T
    conv_macs, post_len = [], []
    for cin, cout in convs:
        conv_macs.append(cin * cout * 3 * L)
        L = math.floor(L / 2)
        post_len.append(L)
    cum = np.cumsum(conv_macs)
    final_head = 128 * post_len[-1] * 64 + 64 * K
    full = cum[-1] + final_head
    frac = {}
    for layer in exit_layers:
        ee_dim = convs[layer - 1][1] * post_len[layer - 1]
        frac[layer] = (cum[layer - 1] + ee_dim * 64 + 64 * K) / full
    frac[5] = 1.0
    return frac


def collect(model, V, y, batch=64):
    """forward_inference over (V=(n,F,T), y) -> E (n_ee,n) entropy, P (n_ee,n) preds,
    Pf (n,) final preds."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(V), torch.LongTensor(y)),
                        batch_size=batch, shuffle=False)
    ent, prd, fin = [], [], []
    with torch.no_grad():
        for bx, _ in loader:
            res = model.forward_inference(bx.to(DEVICE))
            early = [d for d in res if int(np.ravel(d["early_exit"])[0]) == 1]
            ent.append(np.stack([d["confidence"] for d in early], 0))
            prd.append(np.stack([d["output"] for d in early], 0))
            fin.append(np.asarray(res[-1]["output"]))
    return np.concatenate(ent, 1), np.concatenate(prd, 1), np.concatenate(fin)


def gate(E, P, Pf, tau, exit_layers):
    """First early exit (in order) with entropy < tau wins; else final."""
    n = E.shape[1]
    pred, layer, done = Pf.copy().astype(int), np.full(n, 5, int), np.zeros(n, bool)
    for i, Lp in enumerate(exit_layers):
        take = (E[i] < tau) & (~done)
        pred[take] = P[i][take].astype(int)
        layer[take] = Lp
        done |= take
    return pred, layer


def _sav(layer, frac):
    return float(np.mean([1.0 - frac[int(l)] for l in layer])) * 100.0


def train_one(V, y_idx, F, T, K, n_ee, epochs, seed):
    exit_layers = placement_for(n_ee)
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNN1D_extended_EENN(
        in_channels=F, out_classes=K, input_sequence_length=T,
        thresholds=[math.log(K)] * n_ee, num_exits=n_ee, is_training=1,
        n_data=T, exit_placement=list(exit_layers)).to(DEVICE)
    loader = DataLoader(TensorDataset(torch.FloatTensor(V), torch.LongTensor(y_idx)),
                        batch_size=32, shuffle=True)
    model.train_model(loader, loader, loss_weights=decreasing_weights(n_ee), num_epochs=epochs)
    return model, exit_layers


def run_dataset(dataset, seeds, epochs, num_exits_list, val_frac, iso_eps, outdir):
    ds = bd.canonical_name(dataset)
    print(f"\n=== BranchyNet(BN)  {ds}  num_exits={num_exits_list}  epochs={epochs}  "
          f"(paper Table 3 BN column; COMPUTE savings) ===", flush=True)
    rows = []
    for seed in seeds:
        X, y, tr, te = mc.load_split(ds, seed)
        V = np.swapaxes(np.asarray(bd.temporal_view(X, ds), dtype=np.float32), 1, 2)  # (N,F,T)
        classes = np.unique(y)
        yi = np.searchsorted(classes, y)
        F, T, K = V.shape[1], V.shape[2], len(classes)
        tr_fit, tr_val = train_test_split(tr, test_size=val_frac, random_state=seed)

        for n_ee in num_exits_list:
            model, exit_layers = train_one(V[tr_fit], yi[tr_fit], F, T, K, n_ee, epochs, seed)
            frac = compute_fractions(F, T, K, exit_layers)
            # iso-accuracy tau on validation
            Ev, Pv, Pfv = collect(model, V[tr_val], yi[tr_val])
            full_val_acc = accuracy_score(yi[tr_val], Pfv)              # tau->0: all final
            best = None
            for tau in np.linspace(THRESH_LO, THRESH_HI, 15):
                pv, lv = gate(Ev, Pv, Pfv, tau, exit_layers)
                a = accuracy_score(yi[tr_val], pv)
                if a >= full_val_acc - iso_eps / 100.0 and (best is None or _sav(lv, frac) > best[1]):
                    best = (tau, _sav(lv, frac))
            tau_star = best[0] if best else THRESH_LO
            # score on test
            Et, Pt, Pft = collect(model, V[te], yi[te])
            pt, lt = gate(Et, Pt, Pft, tau_star, exit_layers)
            acc = accuracy_score(yi[te], pt)
            exit1_acc = accuracy_score(yi[te], Pt[0])                   # exit-1 alone (Fig 7 style)
            m = bd.prf(classes[yi[te]], classes[pt])
            r = mc.make_row(ds, seed, acc, m["precision_weighted"], m["recall_weighted"],
                            m["f1_weighted"], _sav(lt, frac))
            r["num_exits"] = n_ee
            r["exit1_accuracy"] = round(exit1_acc * 100, 4)
            r["tau"] = round(float(tau_star), 4)
            rows.append(r)
            print(f"  seed={seed} exits={n_ee} tau*={tau_star:.2f}  BN_acc={acc*100:5.1f}%  "
                  f"exit1={exit1_acc*100:5.1f}%  compute_sav={r['savings']:5.1f}%  "
                  f"F1w={m['f1_weighted']:.3f}", flush=True)

        # num_exits=0 -> full-net (no early exit) = Table 3 "Baseline" reference
        model0, _ = train_one(V[tr_fit], yi[tr_fit], F, T, K, 1, epochs, seed)   # 1-exit net, use final only
        _, _, Pf0 = collect(model0, V[te], yi[te])
        acc0 = accuracy_score(yi[te], Pf0)
        m0 = bd.prf(classes[yi[te]], classes[Pf0])
        r0 = mc.make_row(ds, seed, acc0, m0["precision_weighted"], m0["recall_weighted"],
                         m0["f1_weighted"], 0.0)
        r0["num_exits"] = 0
        r0["exit1_accuracy"] = round(acc0 * 100, 4)
        r0["tau"] = 0.0
        rows.append(r0)
        print(f"  seed={seed} exits=0 (Baseline/full-net)  acc={acc0*100:5.1f}%", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"BRANCHYNET_perseed__{ds}.csv"
    fields = mc.PERSEED_FIELDS + ["num_exits", "exit1_accuracy", "tau"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"  -> wrote {out}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="BranchyNet-CNN reproducing the paper Table 3 BN column")
    ap.add_argument("--datasets", nargs="*", default=mc.DATASET_ORDER)
    ap.add_argument("--dataset", default=None, help="single dataset (SLURM-array); overrides --datasets")
    ap.add_argument("--seeds", nargs="*", type=int, default=mc.SEEDS)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--num-exits", nargs="*", type=int, default=NUM_EXITS,
                    help="report BN accuracy for each of these #early-exits (Table 3 cols)")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--iso-eps", type=float, default=1.0)
    ap.add_argument("--outdir", default=str(mc.HERE / "results_cnn"))
    args = ap.parse_args()
    datasets = [args.dataset] if args.dataset else args.datasets
    outdir = Path(args.outdir)
    print(f"BranchyNet-CNN (paper protocol) | seeds={args.seeds} | epochs={args.epochs} | "
          f"num_exits={args.num_exits} | datasets={datasets}", flush=True)
    for ds in datasets:
        run_dataset(ds, args.seeds, args.epochs, args.num_exits, args.val_frac, args.iso_eps, outdir)
    print(f"\nDONE. BRANCHYNET_perseed__*.csv in {outdir}")


if __name__ == "__main__":
    main()
