"""
Per-dataset settings shared by the Non-Myopic baselines (tslearn
NonMyopicEarlyClassifier), plus FlatWrap, which lets a scikit-learn classifier act as
tslearn's base classifier.

The multi-seed runners read only ``nc`` (number of clusters) from CONFIG; the base
classifier and the time cost are fixed rules set in the runners themselves.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

CONFIG = {
    "Epilepsy":           dict(nc=3),
    "PAMAP2":             dict(nc=2),
    "Shoaib":             dict(nc=3),
    "WESADchest":         dict(nc=3),
    "EMGPhysical":        dict(nc=1),
    "SelfRegulationSCP1": dict(nc=2),
}


# ── base classifiers (flatten the (n, t, C) prefix -> (n, t*C)) ──────────────

class FlatWrap(BaseEstimator, ClassifierMixin):
    def __init__(self, maker=None):
        self.maker = maker

    def fit(self, X, y):
        X = np.asarray(X)
        Xf = X.reshape(X.shape[0], -1) if X.ndim == 3 else X
        self.clf_ = self.maker()
        self.clf_.fit(np.nan_to_num(Xf), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        X = np.asarray(X)
        Xf = X.reshape(X.shape[0], -1) if X.ndim == 3 else X
        return self.clf_.predict(np.nan_to_num(Xf))

    def predict_proba(self, X):
        X = np.asarray(X)
        Xf = X.reshape(X.shape[0], -1) if X.ndim == 3 else X
        return self.clf_.predict_proba(np.nan_to_num(Xf))
