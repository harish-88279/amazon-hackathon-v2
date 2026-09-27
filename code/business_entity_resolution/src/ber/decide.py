"""Decision layer: turn calibrated pair probabilities into a complete prediction set per S1.

Policies:
  threshold             one global cutoff tuned by exact macro F0.5 over complete S1 sets
  threshold_per_source  S2/S3 cutoffs (adopted only when the gain is stable under bootstrap)
Optional one_to_one (each target kept only for its best S1) is an ABLATION, off by default,
because target exclusivity is not a rule of the task.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .metric import entity_f05

log = logging.getLogger(__name__)


def apply_policy(scored: pd.DataFrame, policy: dict) -> dict[str, set[str]]:
    """scored: s1_id, cand_id, p -> {s1_id: set(kept)} (anchors with nothing kept are absent)."""
    df = scored
    if policy.get("one_to_one"):
        best = df.groupby("cand_id")["p"].transform("max")
        df = df[df["p"] >= best - 1e-12]
    t2, t3 = float(policy["t_s2"]), float(policy["t_s3"])
    thr = np.where(df["cand_id"].str.startswith("S3").values, t3, t2)
    keep = df[df["p"].values >= thr]
    out: dict[str, set[str]] = {}
    for a, b in zip(keep["s1_id"].values, keep["cand_id"].values):
        out.setdefault(a, set()).add(b)
    return out


def _score(pred: dict, truth: dict, ids: list[str]) -> float:
    return float(np.mean([entity_f05(truth[i], pred.get(i, ())) for i in ids]))


def sweep(scored: pd.DataFrame, truth: dict[str, set], ids: list[str], grid: tuple[float, float, float],
          per_source: bool = False, one_to_one: bool = False, bootstrap: int = 0, seed: int = 42) -> dict:
    """Threshold sweep by exact macro F0.5 on complete S1 sets. Returns best policy + curve."""
    start, stop, step = grid
    ts = np.round(np.arange(start, stop + 1e-9, step), 4)
    curve = []
    for t in ts:
        pol = {"t_s2": t, "t_s3": t, "one_to_one": one_to_one}
        curve.append({"t": float(t), "macro_f05": _score(apply_policy(scored, pol), truth, ids)})
    best = max(curve, key=lambda r: r["macro_f05"])
    policy = {"policy": "threshold", "t_s2": best["t"], "t_s3": best["t"], "one_to_one": one_to_one, "macro_f05": best["macro_f05"]}
    # plateau check: width of the region within 0.002 of the peak
    near = [r["t"] for r in curve if r["macro_f05"] >= best["macro_f05"] - 0.002]
    policy["plateau_width"] = float(max(near) - min(near)) if near else 0.0
    if per_source:
        b0 = best["t"]
        cands = []
        for d2 in (-0.1, -0.05, 0.0, 0.05, 0.1):
            for d3 in (-0.1, -0.05, 0.0, 0.05, 0.1):
                t2, t3 = float(np.clip(b0 + d2, 0.01, 0.99)), float(np.clip(b0 + d3, 0.01, 0.99))
                cands.append({"t_s2": t2, "t_s3": t3, "macro_f05": _score(apply_policy(scored, {"t_s2": t2, "t_s3": t3, "one_to_one": one_to_one}), truth, ids)})
        bps = max(cands, key=lambda r: r["macro_f05"])
        gain = bps["macro_f05"] - best["macro_f05"]
        stable = True
        if bootstrap and gain > 0:
            rng = np.random.default_rng(seed)
            ids_arr = np.array(ids)
            p_glob = apply_policy(scored, {"t_s2": b0, "t_s3": b0, "one_to_one": one_to_one})
            p_ps = apply_policy(scored, {"t_s2": bps["t_s2"], "t_s3": bps["t_s3"], "one_to_one": one_to_one})
            wins = 0
            for _ in range(bootstrap):
                smp = rng.choice(ids_arr, size=len(ids_arr), replace=True)
                wins += _score(p_ps, truth, list(smp)) > _score(p_glob, truth, list(smp))
            stable = wins / bootstrap >= 0.9
        policy["per_source_candidate"] = {**bps, "gain": gain, "stable": stable}
        if gain > 0.001 and stable:
            policy.update({"policy": "threshold_per_source", "t_s2": bps["t_s2"], "t_s3": bps["t_s3"], "macro_f05": bps["macro_f05"]})
    policy["curve"] = curve
    return policy


def bootstrap_ci(pred: dict, truth: dict, ids: list[str], groups: dict[str, str] | None, n: int = 200, seed: int = 42) -> dict:
    """Bootstrap by identity group (all S1 of a component resampled together)."""
    rng = np.random.default_rng(seed)
    if groups:
        by_g: dict[str, list[str]] = {}
        for i in ids:
            by_g.setdefault(groups.get(i, i), []).append(i)
        units = list(by_g.values())
    else:
        units = [[i] for i in ids]
    per = {i: entity_f05(truth[i], pred.get(i, ())) for i in ids}
    unit_scores = np.array([sum(per[i] for i in u) for u in units]); unit_n = np.array([len(u) for u in units])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(units), len(units))
        vals.append(unit_scores[idx].sum() / unit_n[idx].sum())
    return {"mean": float(np.mean(vals)), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], "n_boot": n}
