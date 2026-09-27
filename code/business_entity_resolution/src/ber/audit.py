"""Data audit (run before learning): manifest with hashes, integrity checks, label-graph audit, EDA."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import io as bio
from .config import Config
from .data import load_norm, load_truth

log = logging.getLogger(__name__)


def manifest(cfg: Config) -> dict:
    out = {}
    for k, p in bio.dataset_paths(cfg.data_root).items():
        if not p.exists():
            out[k] = {"path": str(p), "missing": True}
            continue
        out[k] = {"path": str(p), "bytes": p.stat().st_size, "sha256": bio.sha256_file(p)}
    return out


def _bucket(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4+"


def run_audit(cfg: Config, out_dir: Path) -> dict:
    """Full audit. Raises ContractError on integrity failures; writes audit.json + eda tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = {"environment": bio.environment_info(), "manifest": manifest(cfg)}
    missing = [k for k, v in rep["manifest"].items() if v.get("missing")]
    if missing:
        raise bio.C.ContractError(f"dataset incomplete, missing: {missing}")

    # --- per source counts / missingness / lengths (from normalised cache)
    eda = {}
    ids = {}
    for split in ("train", "test"):
        for src in ("s1", "s2", "s3"):
            df = load_norm(cfg, split, src, ["entity_id", "country_norm", "name_missing", "addr_missing",
                                             "name_fold", "addr_fold", "name_key", "postal"])
            ids[(split, src)] = set(df["entity_id"].astype(str))
            g = df.groupby("country_norm")
            eda[f"{split}_{src}"] = {
                "rows": int(len(df)),
                "by_country": g.size().to_dict(),
                "name_missing": int(df["name_missing"].sum()),
                "addr_missing": int(df["addr_missing"].sum()),
                "addr_missing_by_country": g["addr_missing"].mean().round(4).to_dict(),
                "postal_present_by_country": g["postal"].apply(lambda s: float((s.astype(str) != "").mean())).round(4).to_dict(),
                "name_len_mean": float(df["name_fold"].str.len().mean()),
                "addr_len_mean": float(df["addr_fold"].str.len().mean()),
                "exact_dup_name_key_rows": int(df["name_key"].astype(str).duplicated(keep=False).sum()),
            }
            del df
    rep["eda"] = eda

    # --- truth integrity + label graph
    truth = load_truth(cfg)
    s1_train = ids[("train", "s1")]
    tgt_train = ids[("train", "s2")] | ids[("train", "s3")]
    truth_ids = set(truth["source1_entity_id"])
    unknown_s1 = truth_ids - s1_train
    missing_truth = s1_train - truth_ids
    if unknown_s1:
        raise bio.C.ContractError(f"{len(unknown_s1)} ground-truth S1 ids not in train_source1, e.g. {sorted(unknown_s1)[:3]}")
    if missing_truth:
        raise bio.C.ContractError(f"{len(missing_truth)} train S1 ids have no ground-truth row (data integrity issue), e.g. {sorted(missing_truth)[:3]}")
    owners: dict[str, list[str]] = {}
    unknown_targets, s1_in_labels = 0, 0
    counts = []
    for s1, m in zip(truth["source1_entity_id"], truth["matched"]):
        counts.append(len(m))
        for t in m:
            if t.startswith("S1-"):
                s1_in_labels += 1
            elif t not in tgt_train:
                unknown_targets += 1
            owners.setdefault(t, []).append(s1)
    multi = {t: o for t, o in owners.items() if len(o) > 1}
    if s1_in_labels:
        raise bio.C.ContractError(f"{s1_in_labels} S1 ids appear as match targets in the truth file")
    if unknown_targets:
        raise bio.C.ContractError(f"{unknown_targets} truth target ids are not in train_source2/3")
    bucket = Counter(_bucket(c) for c in counts)
    rep["truth"] = {
        "rows": int(len(truth)),
        "labelled_links": int(sum(counts)),
        "links_to_s2": int(sum(1 for m in truth["matched"] for t in m if t.startswith("S2-"))),
        "links_to_s3": int(sum(1 for m in truth["matched"] for t in m if t.startswith("S3-"))),
        "match_count_distribution": dict(sorted(Counter(counts).items())),
        "match_bucket_distribution": dict(bucket),
        "singleton_fraction": float(counts.count(0) / len(counts)),
        "all_empty_baseline_macro_f05": float(counts.count(0) / len(counts)),
        "targets_with_multiple_s1_owners": len(multi),
        "multi_owner_examples": dict(list(multi.items())[:5]),
        "duplicates_inside_label_lists": int(truth.attrs.get("duplicates_inside_lists", 0)),
        "s2_rows_unlabelled": int(len(ids[("train", "s2")]) - len({t for t in owners if t.startswith("S2-")})),
        "s3_rows_unlabelled": int(len(ids[("train", "s3")]) - len({t for t in owners if t.startswith("S3-")})),
    }
    # train/test exact text overlap (structural check only; never exploited)
    tr = load_norm(cfg, "train", "s1", ["name_fold", "addr_fold"])
    te = load_norm(cfg, "test", "s1", ["name_fold", "addr_fold"])
    key_tr = set((tr["name_fold"].astype(str) + "|" + tr["addr_fold"].astype(str)))
    key_te = te["name_fold"].astype(str) + "|" + te["addr_fold"].astype(str)
    rep["train_test_exact_text_overlap_s1"] = int(key_te.isin(key_tr).sum())
    bio.write_json(rep, out_dir / "audit.json")
    log.info("audit written to %s", out_dir / "audit.json")
    return rep
