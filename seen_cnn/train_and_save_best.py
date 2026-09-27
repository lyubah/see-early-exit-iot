"""
train_and_save_best.py  --  train the selected CNN early-exit config and save its weights.

Trains one config (normally the one chosen from the sweep analysis) on CPU and saves ONLY
the state_dict .ckpt, so the file can be copied to the board for measurement with
Main_board_cnn.py. The checkpoint filename encodes the exit config, which
Main_board_cnn.py parses back out.

Covers backbone in {CNN1D, AlexNet} x variant in {Baseline, EE, SEEN}:
    SEEN      -> CNN1D_extended_EENN_partialsampling / AlexNetPartial   (9-arg, partial-window)
    EE / Baseline -> CNN1D_extended_EENN / AlexNetEENN                  (8-arg, full-window)
(The Baseline is the same backbone class as EE; only how it is *run* on the
board differs -- see Main_board_cnn.py. The trained weights are identical, so
we train the EE-style model for both EE and Baseline.)

Training classes come from the TRAINING modules (EENN_functions.py,
Alex_Net_functions.py) -- NOT the *_timers.py board/inference copies.
"""
import os
import sys
import argparse
import pickle
import numpy as np


# ---- locate REPO root (has NN_functions.py + Sensor_aware_early_exit.py) and add to sys.path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start):
    d = start
    for _ in range(8):
        if (os.path.exists(os.path.join(d, "NN_functions.py")) and
                os.path.exists(os.path.join(d, "Sensor_aware_early_exit.py"))):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return start


_REPO_ROOT = _find_repo_root(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from sklearn.model_selection import train_test_split

from utility_functions import prepare_data_mlp  # noqa: E402


# SEEN input sequence lengths per dataset (same values as Sensor_aware_early_exit.py)
_SEQ_LEN = {
    "PAMAP2": 192,
    "SelfRegulationSCP1": 112,
    "Shoaib": 72,
    "ERing": 8,
    "Epilepsy": 24,
    "WESADchest": 24,
    "EMGPhysical": 24,
}


def load_dataset(dataset_name):
    """Load Datasets/{ds}_dataLabels.pkl -> (data, labels). Search cwd + repo root."""
    for root in ("", _REPO_ROOT):
        for rel in (f"Datasets/{dataset_name}_dataLabels.pkl", f"{dataset_name}_dataLabels.pkl"):
            p = os.path.join(root, rel) if root else rel
            if os.path.exists(p):
                with open(p, "rb") as f:
                    d = pickle.load(f)
                return np.asarray(d["data"]), np.asarray(d["labels"]), p
    raise FileNotFoundError(
        f"No dataset pkl for {dataset_name} (looked for Datasets/{dataset_name}_dataLabels.pkl "
        f"in cwd and {_REPO_ROOT})")


def build_model(backbone, variant, n_channel, num_activities, input_seq_len,
                thresholds, num_exits, n_data, exit_placement, new_data_perc):
    """Build the TRAINING model matching backbone x variant (see module docstring)."""
    if backbone == "CNN1D":
        import EENN_functions as mod
        M = mod.CNN1D_extended_EENN_partialsampling if variant == "SEEN" else mod.CNN1D_extended_EENN
    elif backbone == "AlexNet":
        import Alex_Net_functions as mod
        M = mod.AlexNetPartial if variant == "SEEN" else mod.AlexNetEENN
    else:
        raise ValueError(f"--backbone must be CNN1D or AlexNet, got {backbone}")

    if variant == "SEEN":
        model = M(n_channel, num_activities, input_seq_len, thresholds, num_exits, 0,
                  n_data, exit_placement, new_data_perc)          # 9-arg
    else:
        model = M(n_channel, num_activities, input_seq_len, thresholds, num_exits, 0,
                  n_data, exit_placement)                          # 8-arg (EE / Baseline)
    return model


def main():
    ap = argparse.ArgumentParser(description="Train a CNN early-exit config and save its state_dict .ckpt")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--backbone", default="CNN1D", choices=["CNN1D", "AlexNet"])
    ap.add_argument("--variant", default="SEEN", choices=["Baseline", "EE", "SEEN"])
    ap.add_argument("--thresholds", nargs="+", type=float, required=True)
    ap.add_argument("--num_exits", type=int, required=True)
    ap.add_argument("--exit_placement", nargs="+", type=int, required=True)
    ap.add_argument("--new_data_perc", nargs="+", type=int, required=True)
    ap.add_argument("--loss_weights", nargs="+", type=float, required=True)
    ap.add_argument("--input_seq_len", type=int, default=24,
                    help="fallback seq len for datasets not listed in _SEQ_LEN")
    ap.add_argument("--num_epochs", type=int, default=20)
    ap.add_argument("--out_dir", default="./ckpts")
    ap.add_argument("--max_windows", type=int, default=0,
                    help="0 = all windows; >0 caps train/test indices for a fast smoke test")
    args = ap.parse_args()

    device = torch.device("cpu")

    data, labels_array, pkl_path = load_dataset(args.dataset_name)
    print(f"[data] {args.dataset_name}: loaded {pkl_path}")


    n_window, n_channel, n_data = data.shape
    num_activities = len(set(labels_array))
    input_sequence_length = _SEQ_LEN.get(args.dataset_name, args.input_seq_len)
    print(f"[data] windows={n_window} ch={n_channel} n_data={n_data} classes={num_activities} "
          f"input_seq_len={input_sequence_length}")

    # identical 60/40 split, random_state=0
    lst = list(range(0, n_window))
    X_train_ind, X_test_ind, _, _ = train_test_split(lst, labels_array, test_size=0.40, random_state=0)

    # PAMAP2 class-3 augmentation, applied AFTER the split and to the TRAIN side only.
    # (Previously the duplication happened before the split, so duplicated class-3 windows
    # landed in both sides and ~37% of the board's test windows had been trained on.)
    if args.dataset_name == "PAMAP2":
        tr = np.asarray(X_train_ind)
        aug_src = tr[labels_array[tr] == 3]
        new_idx = np.arange(len(labels_array), len(labels_array) + len(aug_src))
        data = np.append(data, data[aug_src], 0)
        labels_array = np.append(labels_array, labels_array[aug_src])
        X_train_ind = list(tr) + list(new_idx)
        print(f"[data] PAMAP2: +{len(aug_src)} class-3 train-only duplicates "
              f"(train {len(X_train_ind)}, test {len(X_test_ind)} raw windows)")

    if args.max_windows and args.max_windows > 0:
        X_train_ind = X_train_ind[:args.max_windows]
        X_test_ind = X_test_ind[:args.max_windows]
        print(f"[smoke] capped to {len(X_train_ind)} train / {len(X_test_ind)} test windows")

    train_features, test_features = prepare_data_mlp(data, labels_array, 0.4, 32, X_train_ind, X_test_ind)

    print(f"[cfg] backbone={args.backbone} variant={args.variant} num_exits={args.num_exits} "
          f"exit_placement={args.exit_placement} thresholds={args.thresholds} "
          f"new_data_perc={args.new_data_perc} loss_weights={args.loss_weights} epochs={args.num_epochs}")

    model = build_model(args.backbone, args.variant, n_channel, num_activities, input_sequence_length,
                        args.thresholds, args.num_exits, n_data, args.exit_placement, args.new_data_perc)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] built {type(model).__name__} params={n_params}")

    print(f"[train] starting train_model for {args.num_epochs} epoch(s) on {device} ...")
    model.is_training = 1  # AlexNet forward routes on self.is_training; must be 1 for forward_train
    model.train_model(train_features, test_features, args.loss_weights, args.num_epochs)

    # save ONLY the state_dict (no inference / no timing) -> ready to copy to the Pi
    os.makedirs(args.out_dir, exist_ok=True)
    fname = (f"{args.dataset_name}_{args.backbone}_{args.variant}"
             f"_thresholds{args.thresholds}_exit_placement{args.exit_placement}"
             f"_new_data_perc{args.new_data_perc}.ckpt")
    save_path = os.path.abspath(os.path.join(args.out_dir, fname))
    model.is_training = 0
    torch.save(model.state_dict(), save_path)

    print(f"[saved] {save_path}")
    print(f"[params] {n_params}")


if __name__ == "__main__":
    main()
