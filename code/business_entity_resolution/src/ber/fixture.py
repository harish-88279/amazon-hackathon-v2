"""Build a small self-consistent dev dataset (from the real training data) or a tiny synthetic
multi-country fixture, in the official folder layout, so the whole CLI can be exercised quickly.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import io as bio

log = logging.getLogger(__name__)


def from_train(data_root: Path, out_root: Path, n_anchors: int, seed: int = 42, validator: Path | None = None) -> Path:
    """Sample n anchors for a pseudo-train and n for a pseudo-test split with held-out GT."""
    rng = np.random.default_rng(seed)
    paths = bio.dataset_paths(data_root)
    truth = bio.read_truth(paths["train_truth"])
    s1 = bio.read_tsv(paths["train_s1"], bio.C.SOURCE_COLUMNS)
    pick = rng.choice(len(truth), size=min(2 * n_anchors, len(truth)), replace=False)
    parts = {"train": truth.iloc[pick[:n_anchors]], "test": truth.iloc[pick[n_anchors:]]}
    keep = {k: set(x for m in g["matched"] for x in m) for k, g in parts.items()}
    n_distract = {k: max(1, int(0.35 * len(v)) // 2) for k, v in keep.items()}
    subsets = {k: {"s2": [], "s3": []} for k in parts}
    pools = {k: {"s2": [], "s3": []} for k in parts}
    all_keep = keep["train"] | keep["test"]
    for src in ("s2", "s3"):
        for chunk in bio.read_tsv(paths[f"train_{src}"], bio.C.SOURCE_COLUMNS, chunksize=500_000):
            free = chunk[~chunk["entity_id"].isin(all_keep)]
            for k in parts:
                subsets[k][src].append(chunk[chunk["entity_id"].isin(keep[k])])
                pools[k][src].append(free.sample(min(len(free), max(n_distract[k] // 10, 1)), random_state=int(rng.integers(1e9))))
    out = out_root / "dataset"
    kw = dict(sep="\t", index=False, quoting=3, encoding="utf-8", lineterminator="\n")
    for k, g in parts.items():
        d = out / k
        d.mkdir(parents=True, exist_ok=True)
        s1_sub = s1[s1["entity_id"].isin(set(g["source1_entity_id"]))]
        s1_sub.to_csv(d / f"{k}_source1.tsv", **kw)
        for src in ("s2", "s3"):
            pl = pd.concat(pools[k][src])
            df = pd.concat([pd.concat(subsets[k][src]), pl.sample(min(n_distract[k], len(pl)), random_state=seed)])
            df.to_csv(d / f"{k}_source{src[-1]}.tsv", **kw)
        g[["source1_entity_id", "matched_entity_ids"]].to_csv(d / (f"{k}_ground_truth.tsv" if k == "train" else "hidden_ground_truth.tsv"), **kw)
        log.info("%s: S1 %d, links %d", k, len(s1_sub), len(keep[k]))
    if validator and validator.exists():
        (out_root / "utils").mkdir(exist_ok=True)
        shutil.copy(validator, out_root / "utils" / "validate_submission.py")
    return out


def synthetic(out_root: Path) -> Path:
    """Tiny multi-country fixture with positives, singletons, an unknown country, accents and blanks."""
    out = out_root / "dataset"
    rows1 = [("S1-1", "Kaveri Precision Private Limited", "12 MG Road, Chennai 600001", "India"),
             ("S1-2", "Blue Finch Bakery", "Unit 4, 12 MG Road, Chennai", "India"),
             ("S1-3", "Harbor Pharmacy", "18 Oak Street, Boston, MA", "US"),
             ("S1-4", "Harbor Pharmacy", "81 Oak Street, Boston, MA", "US"),
             ("S1-5", "Atelier Lumière SARL", "12 rue des Écoles 75005 Paris", "France"),
             ("S1-6", "Lone Singleton Traders", "9 Nowhere Lane, Austin, TX", "US"),
             ("S1-7", "", "", "Testland"),
             ("S1-8", "Payne Enterprises", "3315 Fremont Street, Peoria, IL", "US")]
    rows2 = [("S2-1", "Kaveri Precision Pvt Ltd", "12 M G Rd Chennai 600001", "India"),
             ("S2-2", "Blue Finch Bakery", "Unit 4 12 MG Road Chennai", "India"),
             ("S2-3", "Harbor Pharmacy", "18 Oak St, Boston, Massachusetts", "US"),
             ("S2-4", "ATELIER LUMIERE", "12 RUE DES ECOLES 75005 PARIS", "France"),
             ("S2-5", "Lone Star Traders", "9 Nowhere Ln, Austin, TX", "US"),
             ("S2-6", "Payne Enterpires", "3315 FREMONT ST, PEORIA, IL", "US"),
             ("S2-7", "Payne Énterprises", "3315 Fremont St, Peoria, Illinois", "US")]
    rows3 = [("S3-1", "Kaveri Precision", "12 MG Road Chennai", "India"),
             ("S3-2", "Harbor Pharmacy Inc", "81 Oak Street Boston MA", "US"),
             ("S3-3", "Atelier Lumiere", "12 r des Ecoles, Paris", "France"),
             ("S3-4", "Payne Enterprises LLC", "Fremont St, Peoria, Illinois", "US"),
             ("S3-5", "Dréxkor", "9 Nowhere Lane Austin", "US")]
    truth = [("S1-1", "S2-1,S3-1"), ("S1-2", "S2-2"), ("S1-3", "S2-3"), ("S1-4", "S3-2"), ("S1-5", "S2-4,S3-3"),
             ("S1-6", ""), ("S1-7", ""), ("S1-8", "S2-6,S2-7,S3-4")]
    kw = dict(sep="\t", index=False, quoting=3, encoding="utf-8", lineterminator="\n")
    for split in ("train", "test"):
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows1, columns=bio.C.SOURCE_COLUMNS).to_csv(d / f"{split}_source1.tsv", **kw)
        pd.DataFrame(rows2, columns=bio.C.SOURCE_COLUMNS).to_csv(d / f"{split}_source2.tsv", **kw)
        pd.DataFrame(rows3, columns=bio.C.SOURCE_COLUMNS).to_csv(d / f"{split}_source3.tsv", **kw)
        pd.DataFrame(truth, columns=bio.C.TRUTH_COLUMNS).to_csv(d / (f"{split}_ground_truth.tsv" if split == "train" else "hidden_ground_truth.tsv"), **kw)
    return out
