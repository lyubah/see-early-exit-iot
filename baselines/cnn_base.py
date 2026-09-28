#!/usr/bin/env python3
"""
cnn_base.py -- the SHARED CNN base classifier for the CNN-base comparative baselines.

This is the single source of truth for "the CNN base" used by every CNN baseline in
this sweep (Baseline-CNN, Truncation-CNN, NonMyopic-CNN, TEASER-CNN), so all cells are
apples-to-apples with each other AND with the proposed SEEN model.

BACKBONE = CNN1D_extended (NN_functions.py:369), the EXACT conv feature extractor the
SEEN early-exit model (CNN1D_extended_EENN_partialsampling) is built on:

    conv1: F  -> 8    k3 p1
    conv2: 8  -> 16   k3 p1
    conv3: 16 -> 32   k3 p1
    conv4: 32 -> 64   k3 p1
    conv5: 64 -> 128  k3 p1
    ReLU + MaxPool1d(2) after each conv

HEAD substitution (documented, deliberate): CNN1D_extended's original head is
`flatten -> Linear(32 * T, 64) -> Linear(64, K)`, which (a) hard-codes the sequence
length T and (b) carries a known width bug (32*T, not 128*T'). Both make it unusable
for VARIABLE-length inputs -- and this sweep needs variable length: static truncation
feeds P% prefixes, and tslearn NonMyopic evaluates prefixes down to t=1. We therefore
replace the flatten head with GLOBAL AVERAGE POOLING:

    AdaptiveAvgPool1d(1) -> Linear(128, 64) -> ReLU -> Dropout(0.5) -> Linear(64, K)

The conv backbone -- the part that defines "the CNN1D_extended architecture" -- is
byte-for-byte the same as SEEN's. MaxPool uses ceil_mode=True so short prefixes never
collapse to length 0 (len 1 -> 1 through all five pools).

The wrapper is a drop-in sklearn estimator (fit / predict / predict_proba over the
tslearn/TEASER (n, T, F) 3D layout), so it plugs into NonMyopicEarlyClassifier and the
TEASER analyzer exactly where a RandomForest went. Labels are remapped to 0..K-1 for
CrossEntropyLoss and mapped back on predict, so non-contiguous label subsets (per
cluster / per prefix) neither crash nor mislabel. Adapted from origin/baselines
NM_run_cnn.py::TslearnCNN (which used a shallower 32/64/128 stack); the only change is
the backbone -> CNN1D_extended and ceil_mode pooling.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin

DEVICE = torch.device("cpu")           # baselines run CPU-parity with the RF sweep
# threads: 1 locally (deterministic smoke) but honor CNN_THREADS on a cluster, where
# NonMyopic-CNN (~T CNN fits/seed) benefits from the allocated cores. Set e.g.
# `export CNN_THREADS=$SLURM_CPUS_PER_TASK` in the SLURM script.
torch.set_num_threads(int(os.environ.get("CNN_THREADS", "1")))

# CNN1D_extended conv widths (NN_functions.py:369) -- the SEEN backbone.
CONV_WIDTHS = (8, 16, 32, 64, 128)
DEFAULT_EPOCHS = 20
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 32


class _CNN1DExtendedGAP(nn.Module):
    """CNN1D_extended conv backbone (8/16/32/64/128, k3 p1, ReLU, MaxPool2 ceil) with a
    global-average-pool head so any sequence length >= 1 is valid."""

    def __init__(self, in_channels, n_classes):
        super().__init__()
        w = CONV_WIDTHS
        self.conv1 = nn.Conv1d(in_channels, w[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(w[0], w[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(w[1], w[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(w[2], w[3], kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(w[3], w[4], kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, ceil_mode=True)   # ceil: len 1 stays 1
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(w[4], 64)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # accept (n, T, F) from tslearn/TEASER and flip to (n, F, T) for Conv1d
        if x.dim() == 3 and x.shape[1] != self.conv1.in_channels:
            x = x.transpose(1, 2)
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        x = self.maxpool(self.relu(self.conv4(x)))
        x = self.maxpool(self.relu(self.conv5(x)))
        x = self.gap(x).view(x.size(0), -1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)


class SklearnCNN(BaseEstimator, ClassifierMixin):
    """CNN1D_extended base as a scikit-learn-style classifier over (n, T, F) inputs.

    Params
    ------
    n_channels : int or None   number of features per timestep (F); inferred from X if None
    n_classes  : int or None   inferred from y if None
    random_state : int         seeds torch + numpy for the fit
    n_epochs, lr, batch_size   training config (defaults match origin/baselines NM_run_cnn)
    """

    def __init__(self, n_channels=None, n_classes=None, random_state=42,
                 n_epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch_size=DEFAULT_BATCH):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size

    def fit(self, X, y):
        X = np.nan_to_num(np.asarray(X, dtype=np.float32))
        if X.ndim != 3:
            raise ValueError(f"SklearnCNN expects 3D (n, T, F); got {X.ndim}D {X.shape}")
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)              # -> 0..K-1
        nch = self.n_channels if self.n_channels is not None else X.shape[2]
        ncl = self.n_classes if self.n_classes is not None else len(self.classes_)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model = _CNN1DExtendedGAP(nch, ncl).to(DEVICE)
        self.model.train()
        n = len(X)
        # avoid a trailing singleton batch (BatchNorm-free here, but keeps steps stable)
        drop_last = (n % self.batch_size == 1) and n > self.batch_size
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X), torch.LongTensor(y_idx)),
            batch_size=self.batch_size, shuffle=True, drop_last=drop_last)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        for _ in range(self.n_epochs):
            for bx, by in loader:
                opt.zero_grad()
                loss = crit(self.model(bx.to(DEVICE)), by.to(DEVICE))
                loss.backward()
                opt.step()
        return self

    def _logits(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(X).to(DEVICE))

    def predict(self, X):
        idx = torch.max(self._logits(X), 1)[1].cpu().numpy()
        return self.classes_[idx]                              # map back to real labels

    def predict_proba(self, X):
        return F.softmax(self._logits(X), dim=1).cpu().numpy()


def count_params():
    """Parameter count of the backbone for a representative (F=9, K=5) config -- handy
    for a 'CNN size' footnote next to the RF's 50x30 trees."""
    m = _CNN1DExtendedGAP(9, 5)
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    # self-test: variable-length safety (t=1 prefix must not crash) + label remap
    import sys
    rng = np.random.RandomState(0)
    for T in (1, 3, 20, 200, 896):
        X = rng.randn(40, T, 9).astype(np.float32)
        y = rng.randint(0, 5, size=40)
        clf = SklearnCNN(random_state=42, n_epochs=1).fit(X, y)
        p = clf.predict(X)
        pr = clf.predict_proba(X)
        assert p.shape == (40,) and pr.shape == (40, 5), (T, p.shape, pr.shape)
        assert set(np.unique(p)).issubset(set(np.unique(y)))
        print(f"T={T:4d}  ok  proba_sum~{pr.sum(1).mean():.3f}")
    # non-contiguous label subset
    X = rng.randn(30, 50, 3).astype(np.float32); y = rng.choice([2, 7], size=30)
    clf = SklearnCNN(random_state=1, n_epochs=1).fit(X, y)
    assert set(np.unique(clf.predict(X))).issubset({2, 7})
    print(f"non-contiguous labels ok | backbone params(F=9,K=5) = {count_params():,}")
    print("cnn_base self-test PASSED")
