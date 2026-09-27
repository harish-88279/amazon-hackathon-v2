"""Candidate retrieval: a union of complementary channels from each S1 anchor to S2 and S3
(indexed separately so one source cannot crowd out the other), with provenance per pair.

Channels (all sparse / exact; no dense Cartesian products):
  name_char   char TF-IDF (2-5 grams, char_wb) on name_core          -> cosine top-K
  addr_char   char TF-IDF (3-5 grams) on addr_fold                    -> cosine top-K   (address-only matches)
  comb_word   word TF-IDF (1-2 grams, rarity weighted) on "name_fold | addr_fold" with field prefixes -> top-K
  exact       exact keys: name_key, name_sorted, (postal, first name token), (house_no, street token),
              (metaphone, locality token). Blocks larger than exact_block_cap are ranked by
              name/address char similarity and truncated (counted).
  dense       optional multilingual-e5-small + FAISS (disabled by default)

Country routing: anchors with a known country query the same-country index; anchors with an
unknown / missing country (or when country_filter=false) query the global index. Never drops
an anchor.

Output columns: s1_id, cand_id, src (S2/S3), country_rel (same|unknown|cross), name_cos, addr_cos,
comb_cos, dense_cos, rank_name, rank_addr, rank_comb, exact_mask, n_channels.
The union is capped at ``union_budget`` per anchor by best normalised score (truncations logged).
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config
from .data import ensure_normalized, load_norm

log = logging.getLogger(__name__)

RETR_COLS = ["entity_id", "name_fold", "name_core", "name_sorted", "name_key", "name_meta", "addr_fold",
             "postal", "house_no", "street_tokens", "locality_tokens", "country_norm", "name_missing", "addr_missing"]
EX_NAMEKEY, EX_SORTED, EX_POSTAL, EX_HOUSE, EX_META = 1, 2, 4, 8, 16
CHANNELS = ("name_char", "addr_char", "comb_word", "exact", "dense")


def _topn(A: sp.csr_matrix, B_T: sp.csr_matrix, k: int, lower: float) -> sp.csr_matrix:
    try:
        from sparse_dot_topn import sp_matmul_topn
        return sp_matmul_topn(A, B_T, top_n=k, threshold=lower, sort=True)
    except ImportError:  # pragma: no cover - slow fallback
        rows, cols, vals = [], [], []
        for s in range(0, A.shape[0], 1000):
            Cm = (A[s:s + 1000] @ B_T).tocsr()
            for i in range(Cm.shape[0]):
                lo, hi = Cm.indptr[i], Cm.indptr[i + 1]
                d, ix = Cm.data[lo:hi], Cm.indices[lo:hi]
                keep = d >= lower
                d, ix = d[keep], ix[keep]
                if len(d) > k:
                    part = np.argpartition(-d, k)[:k]
                    d, ix = d[part], ix[part]
                rows += [s + i] * len(d); cols += ix.tolist(); vals += d.tolist()
        return sp.csr_matrix((vals, (rows, cols)), shape=(A.shape[0], B_T.shape[1]))


def _tfidf_channel(q_text: pd.Series, i_text: pd.Series, k: int, lower: float, analyzer: str,
                   ngram: tuple[int, int], chunk: int, min_df: int = 2):
    """Fit TF-IDF on the INDEX text only (frozen transform for queries), return (q_pos, i_pos, cos, rank)."""
    vq, vi = q_text.astype(str).str.len() > 0, i_text.astype(str).str.len() > 0
    if vq.sum() == 0 or vi.sum() == 0:
        e = np.array([], int)
        return e, e, np.array([], np.float32), e
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min(min_df, int(vi.sum())), sublinear_tf=True,
                          dtype=np.float32, lowercase=False, token_pattern=r"(?u)\b\w+\b" if analyzer == "word" else None)
    try:
        I_T = vec.fit_transform(i_text[vi].astype(str).tolist()).T.tocsr()
    except ValueError as e:  # empty vocabulary on degenerate inputs
        log.warning("tfidf channel skipped: %s", e)
        e_ = np.array([], int)
        return e_, e_, np.array([], np.float32), e_
    i_pos, q_pos = np.flatnonzero(vi.values), np.flatnonzero(vq.values)
    qs, iss, vs, rk = [], [], [], []
    for s in range(0, len(q_pos), chunk):
        part = q_pos[s:s + chunk]
        Q = vec.transform(q_text.iloc[part].astype(str).tolist())
        Cm = _topn(Q, I_T, k, lower).tocsr()
        Cm.sort_indices()
        for r in range(Cm.shape[0]):
            lo, hi = Cm.indptr[r], Cm.indptr[r + 1]
            if hi == lo:
                continue
            d, ix = Cm.data[lo:hi], Cm.indices[lo:hi]
            order = np.argsort(-d, kind="stable")
            qs.append(np.full(len(d), part[r])); iss.append(i_pos[ix[order]]); vs.append(d[order]); rk.append(np.arange(1, len(d) + 1))
    if not qs:
        e = np.array([], int)
        return e, e, np.array([], np.float32), e
    return np.concatenate(qs), np.concatenate(iss), np.concatenate(vs).astype(np.float32), np.concatenate(rk).astype(np.int16)


def _first(s: pd.Series) -> list[str]:
    return [x.split(" ", 1)[0] if isinstance(x, str) and x else "" for x in s.astype(object).fillna("").tolist()]


def _exact_channel(q: pd.DataFrame, i: pd.DataFrame, cap: int, stats: dict):
    """Exact-key blocks with provenance mask. Returns (q_pos, i_pos, mask)."""
    def col(df, c):
        return df[c].astype(object).fillna("").tolist()
    keys = {
        EX_NAMEKEY: (col(q, "name_key"), col(i, "name_key"), lambda x: len(x) >= 4),
        EX_SORTED: (col(q, "name_sorted"), col(i, "name_sorted"), lambda x: len(x) >= 4),
        EX_POSTAL: ([f"{p}|{t}" for p, t in zip(col(q, "postal"), _first(q["name_core"]))],
                    [f"{p}|{t}" for p, t in zip(col(i, "postal"), _first(i["name_core"]))],
                    lambda x: not x.startswith("|") and not x.endswith("|")),
        EX_HOUSE: ([f"{h}|{t}" for h, t in zip(col(q, "house_no"), _first(q["street_tokens"]))],
                   [f"{h}|{t}" for h, t in zip(col(i, "house_no"), _first(i["street_tokens"]))],
                   lambda x: not x.startswith("|") and not x.endswith("|")),
        EX_META: ([f"{m}|{t}" for m, t in zip(col(q, "name_meta"), _first(q["locality_tokens"]))],
                  [f"{m}|{t}" for m, t in zip(col(i, "name_meta"), _first(i["locality_tokens"]))],
                  lambda x: not x.startswith("|") and not x.endswith("|")),
    }
    pairs: dict[tuple[int, int], int] = {}
    for bit, (kq, ki, ok) in keys.items():
        inv: dict[str, list[int]] = defaultdict(list)
        for pos, key in enumerate(ki):
            if ok(key):
                inv[key].append(pos)
        big = 0
        for pos, key in enumerate(kq):
            if not ok(key):
                continue
            block = inv.get(key)
            if not block:
                continue
            if len(block) > cap:
                big += 1
                # rank inside the block by name char similarity to the anchor (deterministic), keep top cap
                from rapidfuzz import fuzz
                qa, qn = q["addr_fold"].iat[pos], q["name_core"].iat[pos]
                scored = sorted(((fuzz.ratio(qn, i["name_core"].iat[b]) + fuzz.ratio(qa, i["addr_fold"].iat[b]), -b) for b in block), reverse=True)
                block = [-b for _, b in scored[:cap]]
            for b in block:
                pairs[(pos, b)] = pairs.get((pos, b), 0) | bit
        stats[f"exact_big_blocks_bit{bit}"] = big
    if not pairs:
        e = np.array([], int)
        return e, e, np.array([], np.int8)
    arr = np.array(list(pairs.keys()), dtype=int)
    return arr[:, 0], arr[:, 1], np.array(list(pairs.values()), dtype=np.int8)


def _dense_channel(q: pd.DataFrame, i: pd.DataFrame, k: int, cfg: Config):  # pragma: no cover (optional)
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning("dense channel requested but sentence-transformers/faiss missing -> skipped")
        return None
    model = SentenceTransformer(cfg.get("retrieval.dense.model"), revision=cfg.get("retrieval.dense.revision"))
    def txt(df):
        return ("query: name: " + df["name_fold"].astype(str) + " address: " + df["addr_fold"].astype(str) + " country: " + df["country_norm"].astype(str)).tolist()
    Ei = model.encode(txt(i), batch_size=256, normalize_embeddings=True)
    Eq = model.encode(txt(q), batch_size=256, normalize_embeddings=True)
    index = faiss.IndexFlatIP(Ei.shape[1])
    index.add(Ei.astype(np.float32))
    D, I = index.search(Eq.astype(np.float32), k)
    return np.repeat(np.arange(len(q)), k), I.ravel(), D.ravel().astype(np.float32)


def _retrieve_block(q: pd.DataFrame, i: pd.DataFrame, cfg: Config, stats: dict, use_dense: bool) -> pd.DataFrame:
    """All channels for one (anchor set, one target source) block. Returns pair frame with positions."""
    r = cfg["retrieval"]
    chunk = int(r["query_chunk"])
    frames = []
    qp, ip, v, rk = _tfidf_channel(q["name_core"], i["name_core"], r["k_name_char"], r["min_name_cos"], "char_wb", (2, 5), chunk)
    frames.append(pd.DataFrame({"qi": qp, "ii": ip, "name_cos": v, "rank_name": rk}))
    qp, ip, v, rk = _tfidf_channel(q["addr_fold"], i["addr_fold"], r["k_addr_char"], r["min_addr_cos"], "char_wb", (3, 5), chunk)
    frames.append(pd.DataFrame({"qi": qp, "ii": ip, "addr_cos": v, "rank_addr": rk}))
    comb_q = "n_" + q["name_fold"].astype(str).str.replace(" ", " n_", regex=False) + " a_" + q["addr_fold"].astype(str).str.replace(" ", " a_", regex=False)
    comb_i = "n_" + i["name_fold"].astype(str).str.replace(" ", " n_", regex=False) + " a_" + i["addr_fold"].astype(str).str.replace(" ", " a_", regex=False)
    qp, ip, v, rk = _tfidf_channel(comb_q, comb_i, r["k_combined_word"], r["min_comb_cos"], "word", (1, 2), chunk, min_df=1)
    frames.append(pd.DataFrame({"qi": qp, "ii": ip, "comb_cos": v, "rank_comb": rk}))
    qp, ip, mask = _exact_channel(q, i, int(r["exact_block_cap"]), stats)
    frames.append(pd.DataFrame({"qi": qp, "ii": ip, "exact_mask": mask}))
    if use_dense:
        res = _dense_channel(q, i, r["k_dense"], cfg)
        if res is not None:
            qp, ip, v = res
            frames.append(pd.DataFrame({"qi": qp, "ii": ip, "dense_cos": v}))
    cand = pd.concat(frames, ignore_index=True)
    for c in ("name_cos", "addr_cos", "comb_cos", "dense_cos"):
        if c not in cand:
            cand[c] = np.nan
    for c in ("rank_name", "rank_addr", "rank_comb"):
        if c not in cand:
            cand[c] = np.nan
    if "exact_mask" not in cand:
        cand["exact_mask"] = 0
    agg = cand.groupby(["qi", "ii"], sort=False).agg(
        name_cos=("name_cos", "max"), addr_cos=("addr_cos", "max"), comb_cos=("comb_cos", "max"), dense_cos=("dense_cos", "max"),
        rank_name=("rank_name", "min"), rank_addr=("rank_addr", "min"), rank_comb=("rank_comb", "min"),
        exact_mask=("exact_mask", lambda s: int(np.bitwise_or.reduce(s.fillna(0).astype(int).values))),
    ).reset_index()
    agg["n_channels"] = (agg[["name_cos", "addr_cos", "comb_cos", "dense_cos"]].notna().sum(axis=1) + (agg["exact_mask"] > 0)).astype("int8")
    return agg  # qi/ii are integer positions into q / i (ids are attached later with Arrow take)


def retrieve(cfg: Config, split: str, out_path, anchor_ids: list[str] | None = None, use_dense: bool | None = None,
             pair_chunk: int = 2_000_000) -> dict:
    """Run retrieval for ``split`` (all anchors or a subset) and stream the pair table to ``out_path``
    (Parquet, Arrow strings). Row groups never split an anchor and rows are grouped by anchor, so
    downstream stages can process the file row-group by row-group. Returns stats."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    use_dense = bool(cfg.get("retrieval.dense.enabled", False)) if use_dense is None else use_dense
    budget = int(cfg.get("retrieval.union_budget"))
    country_filter = bool(cfg.get("retrieval.country_filter", True))
    stats: dict = {"anchors": 0, "pairs": 0, "pairs_before_budget": 0, "truncated_anchors": 0, "unknown_country_anchors": 0, "anchors_with_candidates": 0}
    s1_all = load_norm(cfg, split, "s1", RETR_COLS)
    if anchor_ids is not None:
        s1_all = s1_all[s1_all["entity_id"].isin(list(anchor_ids))].reset_index(drop=True)
    stats["anchors"] = int(len(s1_all))
    countries = sorted(s1_all["country_norm"].astype(str).unique().tolist())
    tgt_countries = set()
    for src in ("s2", "s3"):
        tgt_countries |= set(pd.read_parquet(ensure_normalized(cfg, split, src), columns=["country_norm"])["country_norm"].unique().tolist())
    out_path = __import__("pathlib").Path(out_path)
    tmp = out_path.with_suffix(".tmp.parquet")
    writer = None
    schema = None
    for country in countries:
        q = s1_all[s1_all["country_norm"].astype(str) == country].reset_index(drop=True)
        known = country_filter and country != "" and country in tgt_countries
        blocks = []
        for src in ("s2", "s3"):
            i = load_norm(cfg, split, src, RETR_COLS, country=country if known else None)
            log.info("retrieve %s | country=%r (%s) | %d anchors x %d %s targets", split, country, "same" if known else "global", len(q), len(i), src.upper())
            if len(i) == 0:
                continue
            blk = _retrieve_block(q, i, cfg, stats, use_dense)
            blk["cand_id"] = pd.array(pa.array(i["entity_id"], type=pa.string()).take(pa.array(blk["ii"].values)), dtype="string[pyarrow]")
            blk["src"] = src.upper()
            blocks.append(blk.drop(columns=["ii"]))
            del i, blk
        if not known:
            stats["unknown_country_anchors"] += len(q)
        if not blocks:
            continue
        pairs = pd.concat(blocks, ignore_index=True)
        del blocks
        score = pairs[["name_cos", "addr_cos", "comb_cos", "dense_cos"]].fillna(0).max(axis=1) + 0.05 * pairs["n_channels"] + 0.1 * (pairs["exact_mask"] > 0)
        pairs["_s"] = score.astype("float32")
        pairs = pairs.sort_values(["qi", "_s", "cand_id"], ascending=[True, False, True], kind="stable")
        stats["pairs_before_budget"] += int(len(pairs))
        sizes = pairs.groupby("qi").size()
        stats["truncated_anchors"] += int((sizes > budget).sum())
        pairs = pairs.groupby("qi", sort=False).head(budget).drop(columns="_s").reset_index(drop=True)
        stats["anchors_with_candidates"] += int(pairs["qi"].nunique())
        pairs["s1_id"] = pd.array(pa.array(q["entity_id"], type=pa.string()).take(pa.array(pairs["qi"].values)), dtype="string[pyarrow]")
        pairs["country_rel"] = "same" if known else "unknown"
        for c in ("name_cos", "addr_cos", "comb_cos", "dense_cos", "rank_name", "rank_addr", "rank_comb"):
            pairs[c] = pairs[c].astype("float32")
        pairs["exact_mask"] = pairs["exact_mask"].astype("int8")
        cols = ["s1_id", "cand_id", "src", "country_rel", "name_cos", "addr_cos", "comb_cos", "dense_cos", "rank_name", "rank_addr", "rank_comb", "exact_mask", "n_channels"]
        qi = pairs["qi"].values
        # row groups of <= pair_chunk rows that never split an anchor
        bounds = list(range(0, len(pairs), pair_chunk)) + [len(pairs)]
        for k in range(1, len(bounds) - 1):
            b = bounds[k]
            while b < len(pairs) and qi[b] == qi[b - 1]:
                b += 1
            bounds[k] = b
        for k in range(len(bounds) - 1):
            part = pairs.iloc[bounds[k]:bounds[k + 1]][cols]
            if part.empty:
                continue
            t = pa.Table.from_pandas(part, preserve_index=False)
            if writer is None:
                schema = t.schema
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            writer.write_table(t.cast(schema))
        stats["pairs"] += int(len(pairs))
        log.info("   country %r -> %d pairs (%.1f per anchor)", country, len(pairs), len(pairs) / max(len(q), 1))
        del pairs, q
    if writer is None:  # no pairs at all: write an empty typed table
        cols = {"s1_id": pa.string(), "cand_id": pa.string(), "src": pa.string(), "country_rel": pa.string(), "name_cos": pa.float32(), "addr_cos": pa.float32(),
                "comb_cos": pa.float32(), "dense_cos": pa.float32(), "rank_name": pa.float32(), "rank_addr": pa.float32(), "rank_comb": pa.float32(), "exact_mask": pa.int8(), "n_channels": pa.int8()}
        pq.write_table(pa.table({k: pa.array([], type=v) for k, v in cols.items()}), tmp)
    else:
        writer.close()
    tmp.replace(out_path)
    stats["pairs_per_anchor"] = float(stats["pairs"] / max(stats["anchors"], 1))
    return stats


def read_pairs(path, columns=None) -> pd.DataFrame:
    """Load a pair table with Arrow-backed strings (memory-lean)."""
    return pd.read_parquet(path, columns=columns, dtype_backend="pyarrow")


def retrieval_ceiling(pairs: pd.DataFrame, truth: dict[str, set], anchor_ids: list[str], country: dict[str, str] | None = None) -> dict:
    """Recall statistics + macro oracle (perfect classifier inside the candidates) + reduction ratio."""
    from .metric import entity_f05
    cand = pairs.groupby("s1_id")["cand_id"].apply(set).to_dict() if len(pairs) else {}
    links = hit = 0
    per_ent_rec, all_found, any_found = [], 0, 0
    oracle = []
    by_country: dict[str, list] = defaultdict(list)
    by_src = {"S2": [0, 0], "S3": [0, 0]}
    by_bucket: dict[str, list] = defaultdict(list)
    for a in anchor_ids:
        g = truth.get(a, set())
        c = cand.get(a, set())
        o = 1.0 if not g else entity_f05(g, g & c)
        oracle.append(o)
        if country:
            by_country[country.get(a, "")].append(o)
        b = "0" if not g else "1" if len(g) == 1 else "2-3" if len(g) <= 3 else "4+"
        by_bucket[b].append(o)
        if not g:
            continue
        tp = len(g & c)
        links += len(g); hit += tp
        per_ent_rec.append(tp / len(g)); all_found += int(tp == len(g)); any_found += int(tp > 0)
        for t in g:
            k = t[:2]
            by_src[k][0] += 1; by_src[k][1] += int(t in c)
    n_ns = len(per_ent_rec)
    return {
        "anchors": len(anchor_ids), "positive_links": links,
        "micro_link_recall": hit / links if links else None,
        "mean_entity_recall_nonsingleton": float(np.mean(per_ent_rec)) if n_ns else None,
        "frac_nonsingleton_all_found": all_found / n_ns if n_ns else None,
        "frac_nonsingleton_any_found": any_found / n_ns if n_ns else None,
        "macro_oracle_f05": float(np.mean(oracle)) if oracle else None,
        "oracle_by_country": {k: float(np.mean(v)) for k, v in by_country.items()},
        "oracle_by_bucket": {k: float(np.mean(v)) for k, v in sorted(by_bucket.items())},
        "recall_by_source": {k: (v[1] / v[0] if v[0] else None) for k, v in by_src.items()},
    }
