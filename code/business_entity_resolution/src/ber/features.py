"""Pair features with an explicit, versioned schema.

Families: name similarity, address similarity (with present/missing agreement codes), context
(retrieval provenance, ranks, gaps, block sizes, commonness) and interactions. Missingness is
explicit: two blank fields never yield a perfect similarity. Feature order is fixed by
FEATURE_SCHEMA_VERSION and stored with every model; inference fails on a mismatch.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .config import Config
from .data import load_norm

log = logging.getLogger(__name__)
FEATURE_SCHEMA_VERSION = "v2"
ID_COLS = ["s1_id", "cand_id"]
NORM_COLS = ["entity_id", "name_fold", "name_core", "name_sorted", "name_key", "name_legal", "name_acronym",
             "name_meta", "name_missing", "name_nonlatin", "addr_fold", "addr_missing", "addr_nonlatin", "postal", "house_no", "unit", "addr_nums",
             "street_tokens", "locality_tokens", "addr_conf", "country_norm"]
CONTEXT_SRC = ["n_tset", "n_jw", "a_tset", "name_cos", "comb_cos", "nm_x_ad"]


def _pw(fn, a, b) -> np.ndarray:
    try:
        from rapidfuzz.process import cpdist
        return cpdist(a, b, scorer=fn, dtype=np.float32)
    except Exception:
        return np.fromiter((fn(x, y) for x, y in zip(a, b)), np.float32, len(a))


def _jac(a, b) -> np.ndarray:
    out = np.zeros(len(a), np.float32)
    for k, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        u = len(sx | sy)
        out[k] = len(sx & sy) / u if u else 0.0
    return out


def _contain(a, b) -> np.ndarray:
    """token containment: |A∩B| / min(|A|,|B|)"""
    out = np.zeros(len(a), np.float32)
    for k, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        m = min(len(sx), len(sy))
        out[k] = len(sx & sy) / m if m else 0.0
    return out


def _agree(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """present/present equal=2, conflict=-1 ; one missing=0 ; both missing=1 (explicit 4 states)."""
    am, bm = a == "", b == ""
    return np.where(am & bm, 1, np.where(am | bm, 0, np.where(a == b, 2, -1))).astype(np.int8)


def _rare_overlap(a, b, idf: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """idf-weighted shared / disagreeing token mass (rarity from TRAIN text only)."""
    shared = np.zeros(len(a), np.float32); diff = np.zeros(len(a), np.float32)
    for k, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        shared[k] = sum(idf.get(t, 1.0) for t in sx & sy)
        diff[k] = sum(idf.get(t, 1.0) for t in sx ^ sy)
    return shared, diff


def fit_idf(cfg: Config, source_split: str = "train") -> dict[str, float]:
    """Token rarity weights from training S1+S2+S3 name/address text (never from test)."""
    from collections import Counter
    cnt: Counter = Counter()
    n = 0
    for src in ("s1", "s2", "s3"):
        df = load_norm(cfg, source_split, src, ["name_fold", "addr_fold"])
        for col in ("name_fold", "addr_fold"):
            for s in df[col].astype(str).tolist():
                cnt.update(set(s.split()))
            n += len(df)
    return {t: float(np.log((n + 1) / (c + 1)) + 1) for t, c in cnt.items() if c >= 2}


def _pair_block(a: pd.DataFrame, idf: dict[str, float]) -> pd.DataFrame:
    F = pd.DataFrame(index=a.index)
    F["s1_id"], F["cand_id"] = a["s1_id"].values, a["cand_id"].values
    an, bn = a["a_name_core"].tolist(), a["b_name_core"].tolist()
    af, bf = a["a_name_fold"].tolist(), a["b_name_fold"].tolist()
    nm = (a["a_name_missing"].values | a["b_name_missing"].values).astype(bool)
    # ---- name
    F["n_exact_fold"] = (np.array(af) == np.array(bf)).astype(np.int8) & ~nm
    F["n_exact_core"] = (np.array(an) == np.array(bn)).astype(np.int8) & ~nm
    F["n_exact_sorted"] = (a["a_name_sorted"].values == a["b_name_sorted"].values).astype(np.int8) & ~nm
    F["n_key_eq"] = (a["a_name_key"].values == a["b_name_key"].values).astype(np.int8) & ~nm
    F["n_jw"] = _pw(JaroWinkler.normalized_similarity, an, bn)
    F["n_lev"] = _pw(Levenshtein.normalized_similarity, an, bn)
    F["n_key_lev"] = _pw(Levenshtein.normalized_similarity, a["a_name_key"].tolist(), a["b_name_key"].tolist())
    F["n_tset"] = _pw(fuzz.token_set_ratio, af, bf) / 100.0
    F["n_tsort"] = _pw(fuzz.token_sort_ratio, af, bf) / 100.0
    F["n_partial"] = _pw(fuzz.partial_ratio, an, bn) / 100.0
    F["n_jac"] = _jac(an, bn)
    F["n_contain"] = _contain(an, bn)
    sh, df_ = _rare_overlap(an, bn, idf)
    F["n_rare_shared"], F["n_rare_diff"] = sh, df_
    F["n_acronym"] = _agree(a["a_name_acronym"].values.astype(str), a["b_name_acronym"].values.astype(str))
    F["n_meta"] = _agree(a["a_name_meta"].values.astype(str), a["b_name_meta"].values.astype(str))
    F["n_legal"] = _agree(a["a_name_legal"].values.astype(str), a["b_name_legal"].values.astype(str))
    F["n_suffix_only_diff"] = ((F["n_exact_core"] == 1) & (F["n_exact_fold"] == 0)).astype(np.int8)
    la, lb = np.array([len(x) for x in an]), np.array([len(x) for x in bn])
    F["n_len_ratio"] = (np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)).astype(np.float32)
    F["n_ntok_a"] = np.array([x.count(" ") + 1 if x else 0 for x in an], np.int8)
    F["n_ntok_b"] = np.array([x.count(" ") + 1 if x else 0 for x in bn], np.int8)
    F["n_order_changed"] = ((F["n_tsort"] > 0.95) & (F["n_lev"] < 0.9)).astype(np.int8)
    F["n_missing"] = nm.astype(np.int8)
    for c in ("n_jw", "n_lev", "n_tset", "n_tsort", "n_partial", "n_key_lev"):
        F.loc[nm, c] = 0.0
    # ---- address
    aa, ba = a["a_addr_fold"].tolist(), a["b_addr_fold"].tolist()
    am = (a["a_addr_missing"].values | a["b_addr_missing"].values).astype(bool)
    F["a_exact"] = (np.array(aa) == np.array(ba)).astype(np.int8) & ~am
    F["a_jw"] = _pw(JaroWinkler.normalized_similarity, aa, ba)
    F["a_lev"] = _pw(Levenshtein.normalized_similarity, aa, ba)
    F["a_tset"] = _pw(fuzz.token_set_ratio, aa, ba) / 100.0
    F["a_tsort"] = _pw(fuzz.token_sort_ratio, aa, ba) / 100.0
    F["a_jac"] = _jac(aa, ba)
    F["a_postal"] = _agree(a["a_postal"].values.astype(str), a["b_postal"].values.astype(str))
    F["a_house"] = _agree(a["a_house_no"].values.astype(str), a["b_house_no"].values.astype(str))
    F["a_unit"] = _agree(a["a_unit"].values.astype(str), a["b_unit"].values.astype(str))
    F["a_nums_ordered_eq"] = ((a["a_addr_nums"].values == a["b_addr_nums"].values) & (a["a_addr_nums"].values != "")).astype(np.int8)
    F["a_nums_jac"] = _jac(a["a_addr_nums"].tolist(), a["b_addr_nums"].tolist())
    F["a_street_jac"] = _jac(a["a_street_tokens"].tolist(), a["b_street_tokens"].tolist())
    F["a_locality_jac"] = _jac(a["a_locality_tokens"].tolist(), a["b_locality_tokens"].tolist())
    sh, df_ = _rare_overlap(aa, ba, idf)
    F["a_rare_shared"], F["a_rare_diff"] = sh, df_
    F["a_conf_a"] = a["a_addr_conf"].values.astype(np.int8)
    F["a_conf_b"] = a["b_addr_conf"].values.astype(np.int8)
    F["a_missing_a"] = a["a_addr_missing"].values.astype(np.int8)
    F["a_missing_b"] = a["b_addr_missing"].values.astype(np.int8)
    for c in ("a_jw", "a_lev", "a_tset", "a_tsort"):
        F.loc[am, c] = 0.0
    # ---- context / provenance
    for c in ("name_cos", "addr_cos", "comb_cos", "dense_cos"):
        F[c] = a[c].fillna(0).astype(np.float32).values
        F[f"{c}_present"] = a[c].notna().astype(np.int8).values
    for c in ("rank_name", "rank_addr", "rank_comb"):
        F[c] = a[c].fillna(99).astype(np.float32).values
    F["exact_mask"] = a["exact_mask"].astype(np.int8).values
    F["n_channels"] = a["n_channels"].astype(np.int8).values
    F["src_is_s3"] = (a["src"].values == "S3").astype(np.int8)
    F["country_same"] = (a["a_country_norm"].values == a["b_country_norm"].values).astype(np.int8)
    F["country_unknown"] = ((a["a_country_norm"].values == "") | (a["b_country_norm"].values == "")).astype(np.int8)
    F["fields_available"] = (4 - F["n_missing"] - F["a_missing_a"] - F["a_missing_b"]).astype(np.int8)
    F["script_nonlatin_b"] = (a["b_name_nonlatin"].values.astype(int) | a["b_addr_nonlatin"].values.astype(int)).astype(np.int8)
    F["script_nonlatin_a"] = (a["a_name_nonlatin"].values.astype(int) | a["a_addr_nonlatin"].values.astype(int)).astype(np.int8)
    # ---- interactions
    F["nm_x_ad"] = (F["n_tset"] * F["a_tset"]).astype(np.float32)
    F["name_hi_addr_conflict"] = ((F["n_tset"] > 0.85) & ((F["a_house"] == -1) | (F["a_postal"] == -1))).astype(np.int8)
    F["addr_hi_name_low"] = ((F["a_tset"] > 0.85) & (F["n_tset"] < 0.5)).astype(np.int8)
    F["nums_conflict_hidden"] = ((F["a_tset"] > 0.8) & (F["a_nums_jac"] < 0.5) & (F["a_missing_a"] == 0) & (F["a_missing_b"] == 0)).astype(np.int8)
    return F


def _context(L: pd.DataFrame) -> pd.DataFrame:
    C = pd.DataFrame(index=L.index)
    g = L.groupby("s1_id", sort=False)
    C["blk_size"] = g["cand_id"].transform("size").astype(np.int16)
    for col in CONTEXT_SRC:
        C[f"{col}_rk"] = g[col].rank(ascending=False, method="min").astype(np.int16)
        C[f"{col}_gap"] = (g[col].transform("max") - L[col]).astype(np.float32)
    gc = L.groupby("cand_id", sort=False)
    C["cand_n_blocks"] = gc["s1_id"].transform("size").astype(np.int16)
    C["cand_best_nm"] = gc["n_tset"].transform("max").astype(np.float32)
    C["cand_nm_gap"] = (C["cand_best_nm"] - L["n_tset"]).astype(np.float32)
    C["cand_best_comb"] = gc["nm_x_ad"].transform("max").astype(np.float32)
    C["cand_comb_gap"] = (C["cand_best_comb"] - L["nm_x_ad"]).astype(np.float32)
    C["name_commonness"] = L.groupby("b_name_key", sort=False)["cand_id"].transform("size").clip(upper=30000).astype(np.int16)
    return C


def build_features(cfg: Config, split: str, pairs_path, out_path, idf: dict[str, float], pair_chunk: int = 2_000_000):
    """Compute pair + context features for the pair table at ``pairs_path`` (anchor-grouped row groups,
    as written by retrieve) and write Parquet with the SAME row-group boundaries + a context file."""
    out_path = __import__("pathlib").Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(pairs_path)
    ids_s1 = pd.read_parquet(pairs_path, columns=["s1_id"], dtype_backend="pyarrow")["s1_id"]
    s1 = load_norm(cfg, split, "s1", NORM_COLS)
    s1 = s1[s1["entity_id"].isin(ids_s1.unique())].set_index("entity_id")
    del ids_s1
    ids_c = pd.read_parquet(pairs_path, columns=["cand_id"], dtype_backend="pyarrow")["cand_id"].unique()
    idx = pd.concat([load_norm(cfg, split, "s2", NORM_COLS), load_norm(cfg, split, "s3", NORM_COLS)], ignore_index=True)
    idx = idx[idx["entity_id"].isin(ids_c)].set_index("entity_id")
    del ids_c
    s1.columns = [f"a_{c}" for c in s1.columns]
    idx.columns = [f"b_{c}" for c in idx.columns]
    writer, keys = None, []
    tmp = out_path.with_suffix(".tmp.parquet")
    for g in range(pf.num_row_groups):
        part = pf.read_row_group(g).to_pandas(types_mapper=pd.ArrowDtype)
        if part.empty:
            continue
        a = part.join(s1, on="s1_id").join(idx, on="cand_id")
        for c in a.columns:
            if c.startswith(("a_", "b_")) and not c.endswith(("_missing", "_conf", "_nonlatin")):
                a[c] = a[c].astype(object).fillna("").astype(str)
            elif c.endswith(("_missing", "_conf", "_nonlatin")):
                a[c] = a[c].fillna(0).astype(np.int8)
        for c in ("s1_id", "cand_id", "src"):
            a[c] = a[c].astype(object).astype(str)
        F = _pair_block(a, idf)
        k = F[ID_COLS + CONTEXT_SRC].copy()
        k["b_name_key"] = a["b_name_key"].values
        for c in ("s1_id", "cand_id", "b_name_key"):
            k[c] = k[c].astype("string[pyarrow]")
        keys.append(k)
        t = pa.Table.from_pandas(F, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t)
        log.info("features row group %d/%d (%d pairs)", g + 1, pf.num_row_groups, len(part))
        del a, F, part
    if writer is None:
        raise ValueError("no pairs to featurise")
    writer.close()
    L = pd.concat(keys, ignore_index=True)
    del keys
    C = _context(L)
    del L
    ctx_path = out_path.with_name(out_path.stem + "_ctx.parquet")
    pf2 = pq.ParquetFile(tmp)
    cw, off = None, 0
    ctmp = ctx_path.with_suffix(".tmp.parquet")
    for g in range(pf2.num_row_groups):
        n = pf2.metadata.row_group(g).num_rows
        t = pa.Table.from_pandas(C.iloc[off:off + n], preserve_index=False)
        if cw is None:
            cw = pq.ParquetWriter(ctmp, t.schema, compression="zstd")
        cw.write_table(t)
        off += n
    cw.close()
    ctmp.replace(ctx_path)
    tmp.replace(out_path)
    return out_path


def iter_features(path):
    """Yield feature chunks (pair + context joined) row-group by row-group."""
    path = __import__("pathlib").Path(path)
    pf = pq.ParquetFile(path)
    pc = pq.ParquetFile(path.with_name(path.stem + "_ctx.parquet"))
    assert pf.num_row_groups == pc.num_row_groups
    for i in range(pf.num_row_groups):
        yield pd.concat([pf.read_row_group(i).to_pandas(), pc.read_row_group(i).to_pandas()], axis=1)


def load_features(path) -> pd.DataFrame:
    return pd.concat(list(iter_features(path)), ignore_index=True)


def feature_columns(F: pd.DataFrame) -> list[str]:
    # NEVER include label-derived columns (y, fold, owner_fold) - they would leak truth into the model
    return [c for c in F.columns if c not in ID_COLS + ["y", "fold", "weight", "owner_fold", "raw", "p"]]
