"""Probability calibration fitted on out-of-fold scores with group cross-fitting.

For OOF fold f we fit the calibrator on the OOF scores of the other folds and apply it to fold f,
so every calibrated OOF score is out of sample. The final calibrator (for the test set) is fitted
on all OOF scores; it is applied to the final model's raw scores (which are in-sample-free by
construction only approximately - documented limitation, evaluated on the untouched holdout).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.model = None

    def fit(self, raw: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "Calibrator":
        if self.method == "none":
            return self
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y, sample_weight=w)
        else:
            x = _logit(raw).reshape(-1, 1)
            self.model = LogisticRegression(C=1e6, max_iter=1000).fit(x, y, sample_weight=w)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.method == "none" or self.model is None:
            return raw.astype(np.float32)
        if self.method == "isotonic":
            return self.model.predict(raw).astype(np.float32)
        return self.model.predict_proba(_logit(raw).reshape(-1, 1))[:, 1].astype(np.float32)

    def to_json(self) -> dict:
        if self.method == "sigmoid" and self.model is not None:
            return {"method": "sigmoid", "coef": float(self.model.coef_[0][0]), "intercept": float(self.model.intercept_[0])}
        if self.method == "isotonic" and self.model is not None:
            return {"method": "isotonic", "x": self.model.X_thresholds_.tolist(), "y": self.model.y_thresholds_.tolist()}
        return {"method": "none"}

    @staticmethod
    def from_json(d: dict) -> "Calibrator":
        c = Calibrator(d["method"])
        if d["method"] == "sigmoid":
            c.model = LogisticRegression(); c.model.coef_ = np.array([[d["coef"]]]); c.model.intercept_ = np.array([d["intercept"]]); c.model.classes_ = np.array([0, 1])
        elif d["method"] == "isotonic":
            c.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.array(d["x"]), np.array(d["y"]))
        return c


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cross_fit_oof(method: str, raw: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.empty(len(raw), np.float32)
    for f in np.unique(folds):
        m = folds == f
        out[m] = Calibrator(method).fit(raw[~m], y[~m]).transform(raw[m])
    return out


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_p": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    eps = 1e-7
    ll = -np.mean(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
    return {"log_loss": float(ll), "brier": float(np.mean((p - y) ** 2)), "bins": rows,
            "precision_at_0.9": float(y[p >= 0.9].mean()) if (p >= 0.9).any() else None}
