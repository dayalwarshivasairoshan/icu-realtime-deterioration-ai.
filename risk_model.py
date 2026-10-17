"""Deterioration risk model: trained offline, scored per reading.

Trains on a synthetic cohort (different seed from the demo cohort). On IBM LinuxONE the
trained model can be exported to ONNX and served via ONNX Runtime / Snap ML so inference
can use the Telum on-chip AI accelerator. See docs/ARCHITECTURE.md.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from .features import FeatureWindow
from .simulator import make_cohort, stream

HORIZON = 240


def build_dataset(n_patients: int, seed: int):
    patients = make_cohort(n_patients, seed, frac_sick=0.5, horizon=HORIZON)
    windows = {p.pid: FeatureWindow() for p in patients}
    X, y = [], []
    for t, p, r in stream(patients, HORIZON, seed + 1):
        f = windows[p.pid].push(r)
        if t >= 5:
            X.append(f)
            y.append(int(p.progress(t) >= 0.25))  # clearly deteriorating
    return np.array(X), np.array(y)


def train(seed: int = 7):
    X, y = build_dataset(150, seed)
    Xv, yv = build_dataset(50, seed + 100)
    model = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=seed)
    model.fit(X, y)
    auc = roc_auc_score(yv, model.predict_proba(Xv)[:, 1])
    return model, auc


def risk(model, features: np.ndarray) -> float:
    return float(model.predict_proba(features.reshape(1, -1))[0, 1])
