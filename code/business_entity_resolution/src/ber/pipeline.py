"""Stage orchestration: split -> retrieve -> train (features, OOF, calibration, decision, evaluation,
error analysis, stress tests) -> freeze (+ one untouched holdout confirmation) -> infer -> validate.
Every stage writes machine-readable artefacts under runs/<run_id>/ and advances the run status.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import io as bio
from . import runs
from .calibrate import Calibrator, cross_fit_oof, reliability
from .config import Config
from .data import ensure_normalized, load_norm, load_truth, s1_ids_in_order
from .decide import apply_policy, bootstrap_ci, sweep
from .error_analysis import analyse
from .evaluate import slices
from .features import build_features, feature_columns, fit_idf, iter_features, load_features
from .retrieve import read_pairs, retrieval_ceiling, retrieve
from .splits import anchors, assert_no_leak, make_split, target_owner_split
from .train import check_schema, fit_one, load_model, train_oof

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ split
def stage_split(cfg: Config, run_id: str) -> dict:
    d = runs.run_dir(cfg, run_id)
    man = make_split(cfg)
    bio.write_json(man, d / "splits" / "dev_split.json")
    bio.write_json({k: v for k, v in man.items() if k != "entities"}, d / "splits" / "summary.json")
    runs.set_status(cfg, run_id, "SPLIT", split_hash=man["hash"])
    log.info("split %s: %s", man["hash"], man["summary"])
    return man


def _split_manifest(cfg: Config, run_id: str) -> dict:
    p = runs.run_dir(cfg, run_id, create=False) / "splits" / "dev_split.json"
    if not p.exists():
        raise FileNotFoundError(f"no split manifest in run {run_id}; run `split` first")
    return bio.read_json(p)


# ------------------------------------------------------------------ retrieve
def stage_retrieve(cfg: Config, run_id: str, split: str, force: bool = False) -> dict:
    """split: dev | holdout | test. Persists pairs parquet + ceiling stats (labelled splits)."""
    d = runs.run_dir(cfg, run_id)
    out = d / f"pairs_{split}.parquet"
    if out.exists() and not force:
        log.info("pairs exist for %s (use --force to rebuild)", split)
        return bio.read_json(d / f"retrieval_{split}.json")
    t0 = time.time()
    chunk = int(cfg.get("inference.pair_chunk", 2_000_000))
    if split == "test":
        stats = retrieve(cfg, "test", out, pair_chunk=chunk)
        ids = s1_ids_in_order(cfg, "test")
        stats["ceiling"] = None
    else:
        man = _split_manifest(cfg, run_id)
        ids = anchors(man, split)
        if not ids:
            stats = {"anchors": 0, "pairs": 0, "ceiling": None, "note": f"no {split} anchors (holdout_fraction=0?)"}
            retrieve(cfg, "train", out, anchor_ids=[], pair_chunk=chunk)
            bio.write_json(stats, d / f"retrieval_{split}.json")
            return stats
        if split == "dev":
            n = int(cfg.get("training.train_s1_sample", 0) or 0)
            if n and n < len(ids):
                rng = np.random.default_rng(cfg.seed)
                ids = sorted(rng.choice(np.array(ids), size=n, replace=False).tolist())
        stats = retrieve(cfg, "train", out, anchor_ids=ids, pair_chunk=chunk)
        pairs = read_pairs(out)
        for c in ("s1_id", "cand_id"):
            pairs[c] = pairs[c].astype(object).astype(str)
        truth = bio.truth_sets(load_truth(cfg))
        s1 = load_norm(cfg, "train", "s1", ["entity_id", "country_norm"])
        country = dict(zip(s1["entity_id"].astype(str), s1["country_norm"].astype(str)))
        stats["ceiling"] = retrieval_ceiling(pairs, truth, ids, country)
        # per-channel marginal contribution (recall if that channel were removed)
        pos = {(a, t) for a in ids for t in truth.get(a, ())}
        key = pd.MultiIndex.from_arrays([pairs["s1_id"], pairs["cand_id"]])
        hit = pd.Series([k in pos for k in key], index=pairs.index)
        n_pos = len(pos)
        contrib = {}
        for ch, mask in (("name_char", pairs["name_cos"].notna()), ("addr_char", pairs["addr_cos"].notna()),
                         ("comb_word", pairs["comb_cos"].notna()), ("exact", pairs["exact_mask"] > 0)):
            only = (hit & mask & (pairs["n_channels"] == 1)).sum()
            contrib[ch] = {"recall_alone": float((hit & mask).sum() / n_pos) if n_pos else None,
                           "unique_positives": int(only), "pairs": int(mask.sum())}
        stats["channel_contribution"] = contrib
    n1 = len(ids)
    n_t = sum(len(pd.read_parquet(ensure_normalized(cfg, "train" if split != "test" else "test", s), columns=["entity_id"])) for s in ("s2", "s3"))
    stats["reduction_ratio"] = 1 - stats["pairs"] / (n1 * n_t) if n1 * n_t else None
    stats["seconds"] = round(time.time() - t0, 1)
    bio.write_json(stats, d / f"retrieval_{split}.json")
    if split == "dev":
        runs.set_status(cfg, run_id, "RETRIEVED")
    log.info("retrieval %s: %s", split, {k: v for k, v in stats.items() if k not in ("ceiling", "channel_contribution")})
    if stats["ceiling"]:
        log.info("ceiling %s: %s", split, stats["ceiling"])
    return stats


# ------------------------------------------------------------------ train
def _labelled_table(cfg: Config, run_id: str, split: str) -> tuple[pd.DataFrame, dict]:
    """Features + labels + fold + owner-fold for a labelled split."""
    d = runs.run_dir(cfg, run_id)
    fpath = d / f"features_{split}.parquet"
    idf_path = d / "idf.json"
    if idf_path.exists():
        idf = bio.read_json(idf_path)
    else:
        idf = fit_idf(cfg, "train"); bio.write_json(idf, idf_path)
    if not fpath.exists():
        build_features(cfg, "train", d / f"pairs_{split}.parquet", fpath, idf, int(cfg.get("inference.pair_chunk", 2_000_000)))
    F = load_features(fpath)
    truth_df = load_truth(cfg)
    truth = bio.truth_sets(truth_df)
    man = _split_manifest(cfg, run_id)
    ent = man["entities"]
    owner = target_owner_split(man, truth_df)
    F["y"] = np.fromiter(((b in truth.get(a, ())) for a, b in zip(F["s1_id"], F["cand_id"])), np.int8, len(F))
    F["fold"] = np.fromiter((ent[a]["fold"] for a in F["s1_id"]), np.int16, len(F))
    # owner fold of the candidate target: -2 = unlabelled distractor, -1 = holdout-owned, else dev fold
    def _own(b):
        o = owner.get(b)
        return -2 if o is None else (-1 if o[0] == "holdout" else o[1])
    F["owner_fold"] = np.fromiter((_own(b) for b in F["cand_id"]), np.int16, len(F))
    return F, truth


def stage_train(cfg: Config, run_id: str, stress: bool = True) -> dict:
    d = runs.run_dir(cfg, run_id)
    t0 = time.time()
    F, truth = _labelled_table(cfg, run_id, "dev")
    # never fit on pairs whose target belongs to a held-out identity (holdout or the validation fold)
    fit_mask_holdout = F["owner_fold"].values != -1
    log.info("dev table: %d pairs, %d positives, %d anchors; %d pairs dropped from fitting (holdout-owned targets)",
             len(F), int(F["y"].sum()), F["s1_id"].nunique(), int((~fit_mask_holdout).sum()))
    Ff = F[fit_mask_holdout].reset_index(drop=True)
    folds = Ff["fold"].values.astype(int)
    # OOF with per-fold exclusion of the validation fold's owned targets from the training side
    cols = feature_columns(Ff)
    y = Ff["y"].values.astype(np.int8)
    from .train import anchor_weights
    w = anchor_weights(Ff, cfg.get("training.anchor_weighting", "equal_per_anchor"))
    oof = np.full(len(Ff), np.nan, np.float32)
    n_folds = int(folds.max()) + 1
    for f in range(n_folds):
        va = folds == f
        tr = (folds != f) & (Ff["owner_fold"].values != f)
        if va.sum() == 0:
            continue
        log.info("OOF fold %d: fit %d pairs / eval %d pairs", f, tr.sum(), va.sum())
        m = fit_one(cfg, Ff.loc[tr, cols], y[tr], w[tr], Ff.loc[va, cols], y[va])
        oof[va] = m.predict(Ff.loc[va, cols])
    assert not np.isnan(oof).any()
    log.info("final model on all dev pairs (%d)", len(Ff))
    final = fit_one(cfg, Ff[cols], y, w)
    mp = final.save(d / "model")
    from .features import FEATURE_SCHEMA_VERSION
    bio.write_json({"feature_columns": cols, "feature_schema_version": FEATURE_SCHEMA_VERSION, "model_file": mp.name,
                    "model_kind": final.kind, "n_train_pairs": int(len(Ff)), "n_pos": int(y.sum()),
                    "feature_importance": final.feature_importance(cols)}, d / "model.json")
    # calibration (cross-fit on OOF)
    method = cfg.get("calibration.method", "sigmoid")
    cal_oof = cross_fit_oof(method, oof, y, folds)
    calib = Calibrator(method).fit(oof, y)
    bio.write_json(calib.to_json(), d / "calibration.json")
    rel = {"raw": reliability(oof, y), "calibrated": reliability(cal_oof, y)}
    scored = pd.DataFrame({"s1_id": Ff["s1_id"].values, "cand_id": Ff["cand_id"].values, "raw": oof, "p": cal_oof, "y": y, "fold": folds})
    scored.to_parquet(d / "oof_scored.parquet", index=False)
    # decision policy on complete S1 sets (all dev anchors of the sample, incl. those without candidates)
    ids = sorted(set(F["s1_id"]))
    man = _split_manifest(cfg, run_id)
    dev_ids = anchors(man, "dev")
    n = int(cfg.get("training.train_s1_sample", 0) or 0)
    if n and n < len(dev_ids):
        rng = np.random.default_rng(cfg.seed)
        dev_ids = sorted(rng.choice(np.array(dev_ids), size=n, replace=False).tolist())
    ids = dev_ids
    g = cfg.get("decision.grid", [0.05, 0.96, 0.01])
    policy = sweep(scored, truth, ids, tuple(g), per_source=cfg.get("decision.policy") == "threshold_per_source",
                   one_to_one=bool(cfg.get("decision.one_to_one", False)), bootstrap=int(cfg.get("decision.bootstrap_samples", 0)), seed=cfg.seed)
    pred = apply_policy(scored, policy)
    s1 = load_norm(cfg, "train", "s1", ["entity_id", "country_norm", "addr_missing"])
    country = dict(zip(s1["entity_id"].astype(str), s1["country_norm"].astype(str)))
    addrm = dict(zip(s1["entity_id"].astype(str), s1["addr_missing"].astype(int)))
    groups = {i: man["entities"][i]["group"] for i in ids}
    metrics = {
        "run_id": run_id, "n_dev_anchors": len(ids), "policy": {k: v for k, v in policy.items() if k != "curve"},
        "slices": slices(pred, truth, ids, country, addrm),
        "bootstrap": bootstrap_ci(pred, truth, ids, groups, n=int(cfg.get("decision.bootstrap_samples", 200)), seed=cfg.seed),
        "calibration": rel,
        "baselines": _baselines(scored, truth, ids, F),
        "seconds": round(time.time() - t0, 1),
    }
    bio.write_json(policy["curve"], d / "threshold_curve.json")
    # error analysis
    cand = F.groupby("s1_id")["cand_id"].apply(set).to_dict()
    s1_raw = load_norm(cfg, "train", "s1", ["entity_id", "business_name", "business_address"])
    need = set(scored["cand_id"])
    tgt = pd.concat([load_norm(cfg, "train", s, ["entity_id", "business_name", "business_address"]) for s in ("s2", "s3")])
    tgt = tgt[tgt["entity_id"].isin(need)]
    err, err_sum = analyse(pred, truth, ids, cand, scored, s1_raw.astype(str), tgt.astype(str), policy)
    err.to_csv(d / "error_report_dev.csv", index=False)
    metrics["error_summary"] = err_sum
    if stress:
        metrics["stress"] = _stress_tests(cfg, Ff, cols, y, w, truth, country, policy, ids, scored)
    bio.write_json(metrics, d / "metrics_dev.json")
    runs.set_status(cfg, run_id, "EVALUATED", dev_macro_f05=policy["macro_f05"])
    log.info("DEV macro F0.5 = %.4f  policy=%s", policy["macro_f05"], {k: policy[k] for k in ("t_s2", "t_s3", "one_to_one")})
    return metrics


def _baselines(scored: pd.DataFrame, truth: dict, ids: list[str], F: pd.DataFrame) -> dict:
    from .metric import entity_f05
    empty = float(np.mean([entity_f05(truth[i], ()) for i in ids]))
    # conservative exact rule: exact folded name AND (exact address OR postal+house agree)
    rule = F[(F["n_exact_fold"] == 1) & ((F["a_exact"] == 1) | ((F["a_postal"] == 2) & (F["a_house"] == 2)))]
    pr: dict[str, set] = {}
    for a, b in zip(rule["s1_id"], rule["cand_id"]):
        pr.setdefault(a, set()).add(b)
    exact = float(np.mean([entity_f05(truth[i], pr.get(i, ())) for i in ids]))
    # raw-score threshold 0.5 without calibration/sweep
    p05 = apply_policy(scored.assign(p=scored["raw"]), {"t_s2": 0.5, "t_s3": 0.5})
    raw05 = float(np.mean([entity_f05(truth[i], p05.get(i, ())) for i in ids]))
    return {"all_empty": empty, "conservative_exact_rule": exact, "raw_threshold_0.5": raw05}


def _stress_tests(cfg, Ff, cols, y, w, truth, country, policy, ids, scored) -> dict:
    """Country-held-out proxies for France + same-country grouped (already the main eval)."""
    out = {}
    cs = sorted(set(country[i] for i in ids))
    if len(cs) < 2:
        return {"note": "single country in dev sample"}
    anchor_country = np.array([country.get(a, "") for a in Ff["s1_id"]])
    for tr_c in cs:
        tr = anchor_country == tr_c
        te = ~tr
        if tr.sum() == 0 or te.sum() == 0:
            continue
        m = fit_one(cfg, Ff.loc[tr, cols], y[tr], w[tr])
        p = Calibrator(cfg.get("calibration.method", "sigmoid")).fit(m.predict(Ff.loc[tr, cols]), y[tr]).transform(m.predict(Ff.loc[te, cols]))
        sc = pd.DataFrame({"s1_id": Ff.loc[te, "s1_id"].values, "cand_id": Ff.loc[te, "cand_id"].values, "p": p})
        te_ids = [i for i in ids if country.get(i) != tr_c]
        pred = apply_policy(sc, policy)
        from .metric import summary
        s = summary(truth, pred, te_ids)
        out[f"train_{tr_c}__eval_others"] = {"macro_f05": s["macro_f05"], "singleton_false_merge_rate": s["singleton_false_merge_rate"],
                                             "pair_precision": s["pair_precision"], "pair_recall": s["pair_recall"], "n": len(te_ids)}
    return out


# ------------------------------------------------------------------ freeze + holdout
def stage_freeze(cfg: Config, run_id: str, out_config: Path) -> dict:
    d = runs.run_dir(cfg, run_id, create=False)
    metrics = bio.read_json(d / "metrics_dev.json")
    pol = metrics["policy"]
    final = cfg.with_overrides(**{"frozen.run_id": run_id, "frozen.policy": {k: pol[k] for k in ("policy", "t_s2", "t_s3", "one_to_one")},
                                  "frozen.dev_macro_f05": pol["macro_f05"], "frozen.split_hash": _split_manifest(cfg, run_id)["hash"]})
    final.dump(out_config)
    # one untouched holdout confirmation
    st = stage_retrieve(cfg, run_id, "holdout")
    if st.get("anchors", 0) == 0:
        hm = {"run_id": run_id, "note": "no holdout anchors - holdout confirmation skipped", "slices": {"overall": {"macro_f05": float("nan")}}, "bootstrap": None}
        bio.write_json(hm, d / "metrics_holdout.json")
        runs.set_status(cfg, run_id, "FROZEN", final_config=str(out_config))
        return hm
    F, truth = _labelled_table(cfg, run_id, "holdout")
    model, meta = load_model(d)
    cols = check_schema(F, meta)
    calib = Calibrator.from_json(bio.read_json(d / "calibration.json"))
    p = calib.transform(model.predict(F[cols]))
    sc = pd.DataFrame({"s1_id": F["s1_id"].values, "cand_id": F["cand_id"].values, "p": p})
    ids = anchors(_split_manifest(cfg, run_id), "holdout")
    pred = apply_policy(sc, pol)
    s1 = load_norm(cfg, "train", "s1", ["entity_id", "country_norm"])
    country = dict(zip(s1["entity_id"].astype(str), s1["country_norm"].astype(str)))
    hm = {"run_id": run_id, "n_holdout_anchors": len(ids), "policy": pol, "slices": slices(pred, truth, ids, country),
          "bootstrap": bootstrap_ci(pred, truth, ids, None, n=int(cfg.get("decision.bootstrap_samples", 200)), seed=cfg.seed),
          "ceiling": bio.read_json(d / "retrieval_holdout.json")["ceiling"]}
    bio.write_json(hm, d / "metrics_holdout.json")
    runs.set_status(cfg, run_id, "FROZEN", holdout_macro_f05=hm["slices"]["overall"]["macro_f05"], final_config=str(out_config))
    log.info("HOLDOUT macro F0.5 = %.4f (dev was %.4f)", hm["slices"]["overall"]["macro_f05"], pol["macro_f05"])
    return hm


# ------------------------------------------------------------------ infer
def _pair_fingerprint(it) -> tuple[int, int]:
    """(count, xor of 64-bit hashes) over an iterable of (s1, cand) - used to prove the candidate file
    equals the scored pair set without holding either in memory."""
    import hashlib
    n, x = 0, 0
    for a, b in it:
        n += 1
        x ^= int.from_bytes(hashlib.blake2b(f"{a}\t{b}".encode(), digest_size=8).digest(), "little")
    return n, x


def stage_infer(cfg: Config, run_id: str, split: str = "test", force: bool = False) -> dict:
    from .validate import hard_checks, official
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = runs.run_dir(cfg, run_id, create=False)
    pol = cfg.get("frozen.policy") or bio.read_json(d / "metrics_dev.json")["policy"]
    t0 = time.time()
    stats = stage_retrieve(cfg, run_id, split, force=force)
    ppath = d / f"pairs_{split}.parquet"
    fpath = d / f"features_{split}.parquet"
    idf = bio.read_json(d / "idf.json")
    if force or not fpath.exists():
        build_features(cfg, split, ppath, fpath, idf, int(cfg.get("inference.pair_chunk", 2_000_000)))
    model, meta = load_model(d)
    calib = Calibrator.from_json(bio.read_json(d / "calibration.json"))
    spath = d / f"scored_{split}.parquet"
    sw = None
    n_scored = 0
    fp_n, fp_x = 0, 0
    for F in iter_features(fpath):
        cols = check_schema(F, meta)
        raw = model.predict(F[cols])
        if not np.isfinite(raw).all():
            raise RuntimeError("non-finite scores produced - aborting inference")
        p = calib.transform(raw)
        t = pa.table({"s1_id": pa.array(F["s1_id"].values, type=pa.string()), "cand_id": pa.array(F["cand_id"].values, type=pa.string()),
                      "raw": pa.array(raw, type=pa.float32()), "p": pa.array(p, type=pa.float32())})
        if sw is None:
            sw = pq.ParquetWriter(spath.with_suffix(".tmp.parquet"), t.schema, compression="zstd")
        sw.write_table(t)
        n, x = _pair_fingerprint(zip(F["s1_id"].values, F["cand_id"].values))
        fp_n += n; fp_x ^= x; n_scored += len(F)
        del F, t
    sw.close()
    spath.with_suffix(".tmp.parquet").replace(spath)
    n_pairs = pq.ParquetFile(ppath).metadata.num_rows
    if n_scored != n_pairs:
        raise RuntimeError(f"scored {n_scored} pairs but retrieval persisted {n_pairs}")
    # decisions + outputs, streamed by row group (rows are grouped by anchor)
    ids = s1_ids_in_order(cfg, split)
    pred: dict[str, list[str]] = {}
    cand_index: dict[str, tuple[int, int]] = {}
    t2, t3 = float(pol["t_s2"]), float(pol["t_s3"])
    if pol.get("one_to_one"):
        raise NotImplementedError("one_to_one is an ablation for labelled splits only; not supported in streaming inference")
    off = 0
    sf = pq.ParquetFile(spath)
    for g in range(sf.num_row_groups):
        rg = sf.read_row_group(g)
        s1 = rg.column("s1_id").to_numpy(zero_copy_only=False)
        cd = rg.column("cand_id").to_numpy(zero_copy_only=False)
        p = rg.column("p").to_numpy()
        thr = np.where(np.char.startswith(cd.astype(str), "S3"), t3, t2)
        keep = p >= thr
        # boundaries of anchor runs
        change = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1], True])
        for a, b in zip(change[:-1], change[1:]):
            aid = s1[a]
            cand_index[aid] = (off + a, off + b)
            if keep[a:b].any():
                pred[aid] = cd[a:b][keep[a:b]].tolist()
        off += len(s1)
    # candidate lists are read back lazily by slicing the scored file (avoids 80M python strings at once)
    cand_col = pq.read_table(spath, columns=["cand_id"]).column("cand_id")
    class _CandMap(dict):
        def get(self, k, default=()):
            if k in cand_index:
                a, b = cand_index[k]
                return cand_col.slice(a, b - a).to_pylist()
            return default
    out_dir = cfg.output_root
    bio.write_id_list_tsv(out_dir / "matching_results.tsv", "matched_entity_ids", ids, pred, atomic=True)
    bio.write_id_list_tsv(out_dir / "candidate_pairs.tsv", "candidate_entity_ids", ids, _CandMap(), atomic=True)
    n_no_cand = sum(1 for i in ids if i not in cand_index)
    rep = {"run_id": run_id, "policy": pol, "n_s1": len(ids), "n_pairs_scored": int(n_scored), "anchors_without_candidates": n_no_cand,
           "n_matches": int(sum(len(v) for v in pred.values())), "n_empty_predictions": int(sum(1 for i in ids if not pred.get(i))),
           "retrieval": {k: v for k, v in stats.items() if k != "ceiling"}, "seconds": round(time.time() - t0, 1)}
    runs.set_status(cfg, run_id, "INFERRED")
    test_dir = cfg.data_root / "test"
    rep["hard_checks"] = hard_checks(out_dir, test_dir, scored_fingerprint=(fp_n, fp_x))
    rep["official_validator"] = official(cfg.validator, out_dir, test_dir)
    bio.write_json(rep, d / f"inference_{split}.json")
    if rep["hard_checks"]["ok"] and rep["official_validator"].get("pass"):
        runs.set_status(cfg, run_id, "VALIDATED", output_sha256=rep["hard_checks"]["sha256_matching"])
    else:
        log.error("VALIDATION FAILED: %s / %s", rep["hard_checks"]["problems"], rep["official_validator"].get("stdout", "")[-500:])
    return rep
