"""Identity-group splits.

Groups = connected components of the label graph (S1 <-> labelled S2/S3). A singleton S1 is its
own group; if a target is shared by several S1 rows the whole component is one group and is
flagged. We then create an untouched final HOLDOUT (by group, stratified by country x match
bucket) and grouped K folds on the remaining DEV groups. Split ids and seed are persisted.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from . import io as bio
from .config import Config
from .data import load_norm, load_truth

log = logging.getLogger(__name__)


class UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def identity_groups(truth: pd.DataFrame) -> tuple[dict[str, str], int]:
    """Map every S1 id -> group id; returns (mapping, number of multi-S1 components)."""
    uf = UnionFind()
    for s1, m in zip(truth["source1_entity_id"], truth["matched"]):
        uf.find(s1)
        for t in m:
            uf.union(s1, t)
    groups = {s1: uf.find(s1) for s1 in truth["source1_entity_id"]}
    sizes = defaultdict(int)
    for g in groups.values():
        sizes[g] += 1
    return groups, sum(1 for v in sizes.values() if v > 1)


def _bucket(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4+"


def make_split(cfg: Config) -> dict:
    """Return the split manifest {s1_id: {"group", "part": holdout|dev, "fold": int|-1}, ...}."""
    truth = load_truth(cfg)
    s1 = load_norm(cfg, "train", "s1", ["entity_id", "country_norm"])
    country = dict(zip(s1["entity_id"].astype(str), s1["country_norm"].astype(str)))
    groups, n_multi = identity_groups(truth)
    df = pd.DataFrame({
        "s1_id": truth["source1_entity_id"],
        "group": [groups[s] for s in truth["source1_entity_id"]],
        "country": [country.get(s, "") for s in truth["source1_entity_id"]],
        "bucket": [_bucket(len(m)) for m in truth["matched"]],
    })
    # group-level stratum = (country, bucket) of the group's first row (groups are almost always size 1)
    gdf = df.groupby("group").agg(country=("country", "first"), bucket=("bucket", "first"), n=("s1_id", "size")).reset_index()
    rng = np.random.default_rng(cfg.seed)
    gdf["u"] = rng.random(len(gdf))
    gdf["stratum"] = gdf["country"] + "|" + gdf["bucket"]
    frac = float(cfg.get("split.holdout_fraction", 0.15))
    n_folds = int(cfg.get("split.n_folds", 5))
    gdf["part"] = "dev"
    gdf["fold"] = -1
    for _, idx in gdf.groupby("stratum").groups.items():
        sub = gdf.loc[idx].sort_values("u")
        k = int(round(frac * len(sub)))
        gdf.loc[sub.index[:k], "part"] = "holdout"
        dev_idx = sub.index[k:]
        gdf.loc[dev_idx, "fold"] = np.arange(len(dev_idx)) % n_folds
    part = dict(zip(gdf["group"], gdf["part"]))
    fold = dict(zip(gdf["group"], gdf["fold"]))
    manifest = {
        "seed": cfg.seed, "holdout_fraction": frac, "n_folds": n_folds,
        "n_groups": int(len(gdf)), "multi_s1_groups": int(n_multi),
        "entities": {s: {"group": g, "part": part[g], "fold": int(fold[g])} for s, g in groups.items()},
    }
    manifest["hash"] = hashlib.sha256(str(sorted((s, v["part"], v["fold"]) for s, v in manifest["entities"].items())).encode()).hexdigest()[:12]
    manifest["summary"] = {
        "holdout_entities": sum(1 for v in manifest["entities"].values() if v["part"] == "holdout"),
        "dev_entities": sum(1 for v in manifest["entities"].values() if v["part"] == "dev"),
        "by_fold": {int(k): int(v) for k, v in pd.Series([v["fold"] for v in manifest["entities"].values()]).value_counts().sort_index().items()},
    }
    assert_no_leak(manifest, truth)
    return manifest


def assert_no_leak(manifest: dict, truth: pd.DataFrame) -> None:
    """Every identity group lives in exactly one part/fold; every labelled target belongs to one group."""
    ent = manifest["entities"]
    seen: dict[str, tuple] = {}
    for s1, m in zip(truth["source1_entity_id"], truth["matched"]):
        key = (ent[s1]["part"], ent[s1]["fold"])
        for t in m:
            if t in seen and seen[t] != key:
                raise AssertionError(f"leak: target {t} is labelled from two different splits")
            seen[t] = key


def target_owner_split(manifest: dict, truth: pd.DataFrame) -> dict[str, tuple[str, int]]:
    """target id -> (part, fold) of the S1 group that owns it (for negative-sampling exclusion)."""
    ent = manifest["entities"]
    out = {}
    for s1, m in zip(truth["source1_entity_id"], truth["matched"]):
        for t in m:
            out[t] = (ent[s1]["part"], ent[s1]["fold"])
    return out


def anchors(manifest: dict, part: str, fold: int | None = None, exclude_fold: int | None = None) -> list[str]:
    ent = manifest["entities"]
    out = []
    for s, v in ent.items():
        if v["part"] != part:
            continue
        if fold is not None and v["fold"] != fold:
            continue
        if exclude_fold is not None and v["fold"] == exclude_fold:
            continue
        out.append(s)
    return sorted(out)
