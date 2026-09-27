"""Pair classifier training with grouped out-of-fold predictions.

* candidates are generated only for DEV anchors (training truth never touches the holdout)
* label = pair in truth; negatives are the natural retrieved non-matches (hard by construction)
* anchor weighting: equal total weight per S1 (macro-aware) or per pair (ablation)
* model: CatBoost (Apache-2.0) by default, LightGBM fallback
* OOF: for each dev fold f, fit on the other dev folds, predict fold f -> oof.parquet
* final model: fit on all dev folds (used for holdout confirmation and test inference)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import io as bio
from .config import Config
from .features import FEATURE_SCHEMA_VERSION, feature_columns

log = logging.getLogger(__name__)


def fit_one(cfg: Config, X: pd.DataFrame, y: np.ndarray, w: np.ndarray, Xv: pd.DataFrame | None = None, yv: np.ndarray | None = None):
    """Fit one classifier; returns an object with .predict_proba-like ``predict(X)``."""
    model = cfg.get("training.model", "catboost")
    if model == "catboost":
        try:
            from catboost import CatBoostClassifier, Pool
        except ImportError:
            log.warning("catboost not installed -> lightgbm fallback")
            model = "lightgbm"
    if model == "catboost":
        p = cfg.get("training.catboost", {})
        clf = CatBoostClassifier(depth=int(p.get("depth", 8)), learning_rate=float(p.get("learning_rate", 0.05)),
                                 iterations=int(p.get("iterations", 2000)), l2_leaf_reg=float(p.get("l2_leaf_reg", 3.0)),
                                 loss_function="Logloss", eval_metric="Logloss", od_type="Iter", od_wait=int(p.get("od_wait", 100)),
                                 random_seed=cfg.seed, thread_count=int(p.get("thread_count", -1)), verbose=200, allow_writing_files=False)
        eval_set = Pool(Xv, yv) if Xv is not None else None
        clf.fit(Pool(X, y, weight=w), eval_set=eval_set, use_best_model=eval_set is not None)
        return _Wrapped("catboost", clf)
    import lightgbm as lgb
    p = cfg.get("training.lightgbm", {})
    clf = lgb.LGBMClassifier(num_leaves=int(p.get("num_leaves", 127)), learning_rate=float(p.get("learning_rate", 0.05)),
                             n_estimators=int(p.get("n_estimators", 3000)), min_child_samples=int(p.get("min_child_samples", 100)),
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=cfg.seed, verbose=-1)
    if Xv is not None:
        clf.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        clf.fit(X, y, sample_weight=w)
    return _Wrapped("lightgbm", clf)


class _Wrapped:
    def __init__(self, kind: str, clf):
        self.kind, self.clf = kind, clf

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: Path) -> Path:
        if self.kind == "catboost":
            p = path.with_suffix(".cbm"); self.clf.save_model(str(p))
        else:
            p = path.with_suffix(".txt"); self.clf.booster_.save_model(str(p))
        return p

    @staticmethod
    def load(path: Path) -> "_Wrapped":
        if path.suffix == ".cbm":
            from catboost import CatBoostClassifier
            c = CatBoostClassifier(); c.load_model(str(path)); return _Wrapped("catboost", c)
        import lightgbm as lgb
        b = lgb.Booster(model_file=str(path))
        class _B:  # minimal predict_proba shim for a raw booster
            def predict_proba(self, X):
                p = b.predict(X); return np.c_[1 - p, p]
        return _Wrapped("lightgbm", _B())

    def feature_importance(self, cols: list[str]) -> dict[str, float]:
        try:
            imp = self.clf.get_feature_importance() if self.kind == "catboost" else self.clf.feature_importances_
            return dict(sorted(zip(cols, map(float, imp)), key=lambda x: -x[1]))
        except Exception:
            return {}


def anchor_weights(F: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "per_pair":
        return np.ones(len(F), np.float32)
    n = F.groupby("s1_id")["cand_id"].transform("size").values
    return (1.0 / n).astype(np.float32) * float(np.mean(n))  # equal total per anchor, mean weight 1


def train_oof(cfg: Config, F: pd.DataFrame, folds: np.ndarray, out_dir: Path) -> tuple[pd.DataFrame, "_Wrapped", list[str]]:
    """Grouped OOF over dev folds + final model on all dev data. Writes artefacts to out_dir."""
    cols = feature_columns(F)
    y = F["y"].values.astype(np.int8)
    w = anchor_weights(F, cfg.get("training.anchor_weighting", "equal_per_anchor"))
    n_folds = int(folds.max()) + 1
    oof = np.full(len(F), np.nan, np.float32)
    for f in range(n_folds):
        tr, va = folds != f, folds == f
        if va.sum() == 0:
            continue
        log.info("OOF fold %d: train %d pairs (%d pos) | val %d pairs (%d pos)", f, tr.sum(), y[tr].sum(), va.sum(), y[va].sum())
        m = fit_one(cfg, F.loc[tr, cols], y[tr], w[tr], F.loc[va, cols], y[va])
        oof[va] = m.predict(F.loc[va, cols])
    assert not np.isnan(oof).any(), "OOF has NaN (a fold got no predictions)"
    oof_df = F[["s1_id", "cand_id", "y"]].copy()
    oof_df["fold"] = folds
    oof_df["raw"] = oof
    oof_df.to_parquet(out_dir / "oof.parquet", index=False)
    log.info("fitting final model on all dev folds (%d pairs)", len(F))
    final = fit_one(cfg, F[cols], y, w)
    mp = final.save(out_dir / "model")
    bio.write_json({"feature_columns": cols, "feature_schema_version": FEATURE_SCHEMA_VERSION, "model_file": mp.name,
                    "model_kind": final.kind, "n_train_pairs": int(len(F)), "n_pos": int(y.sum()),
                    "anchor_weighting": cfg.get("training.anchor_weighting"),
                    "feature_importance": final.feature_importance(cols)}, out_dir / "model.json")
    return oof_df, final, cols


def load_model(run_dir: Path) -> tuple["_Wrapped", dict]:
    meta = bio.read_json(run_dir / "model.json")
    return _Wrapped.load(run_dir / meta["model_file"]), meta


def check_schema(F: pd.DataFrame, meta: dict) -> list[str]:
    cols = meta["feature_columns"]
    missing = [c for c in cols if c not in F.columns]
    if missing or meta.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"feature schema mismatch: missing={missing[:5]} version={meta.get('feature_schema_version')} vs {FEATURE_SCHEMA_VERSION}")
    return cols
