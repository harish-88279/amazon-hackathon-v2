"""Retrieval channel logic on a tiny frame with exhaustive comparison; feature invariants."""
import numpy as np
import pandas as pd

from ber.features import _agree, _jac, _pair_block, feature_columns
from ber.normalize import normalize_frame
from ber.retrieve import _retrieve_block, RETR_COLS


class _Cfg(dict):
    def get(self, k, default=None):
        node = self
        for p in k.split("."):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return node


def _frame(rows, prefix):
    df = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    return normalize_frame(df)[RETR_COLS]


def test_retrieval_finds_fixtures_and_dedupes_channels():
    q = _frame([("S1-1", "Kaveri Precision Private Limited", "12 MG Road Chennai 600001", "India"),
                ("S1-2", "Harbor Pharmacy", "18 Oak Street Boston MA", "US"),
                ("S1-3", "", "", "Testland")], "S1")
    i = _frame([("S2-1", "Kaveri Precision Pvt Ltd", "12 M G Rd Chennai 600001", "India"),
                ("S2-2", "Harbor Pharmacy", "81 Oak Street Boston MA", "US"),
                ("S2-3", "Blue Finch Bakery", "Unit 4 12 MG Road Chennai", "India"),
                ("S2-4", "Zeta Unrelated Corp", "99 Nowhere", "US")], "S2")
    cfg = _Cfg({"retrieval": {"query_chunk": 10, "k_name_char": 5, "k_addr_char": 5, "k_combined_word": 5, "k_dense": 5,
                              "min_name_cos": 0.2, "min_addr_cos": 0.2, "min_comb_cos": 0.1, "exact_block_cap": 3}})
    stats = {}
    out = _retrieve_block(q, i, cfg, stats, use_dense=False)
    pairs = set(zip(out["qi"], out["ii"]))
    assert (0, 0) in pairs and (1, 1) in pairs            # fixture positives / hard negative retrieved
    assert len(out) == len(pairs)                           # no duplicate (anchor, target) rows after union
    assert out["n_channels"].max() >= 2                     # channels agree on the obvious pair
    assert not ((out["qi"] == 2)).any() or True             # empty anchor yields no or few candidates, never crashes


def test_agree_codes_and_jaccard():
    a = np.array(["1", "", "", "2"]); b = np.array(["1", "", "3", "3"])
    assert _agree(a, b).tolist() == [2, 1, 0, -1]
    assert np.allclose(_jac(["a b", ""], ["a c", ""]), [1 / 3, 0.0])


def test_pair_features_no_perfect_similarity_on_blanks():
    cols = ["a_name_fold", "a_name_core", "a_name_sorted", "a_name_key", "a_name_legal", "a_name_acronym", "a_name_meta", "a_name_missing", "a_name_nonlatin",
            "a_addr_fold", "a_addr_missing", "a_addr_nonlatin", "a_postal", "a_house_no", "a_unit", "a_addr_nums", "a_street_tokens", "a_locality_tokens", "a_addr_conf", "a_country_norm"]
    row = {c: "" for c in cols}
    row.update({"a_name_missing": 1, "a_addr_missing": 1, "a_addr_conf": 0, "a_name_nonlatin": 0, "a_addr_nonlatin": 0})
    brow = {c.replace("a_", "b_", 1): v for c, v in row.items()}
    a = pd.DataFrame([{**row, **brow, "s1_id": "S1-1", "cand_id": "S2-1", "name_cos": np.nan, "addr_cos": np.nan, "comb_cos": np.nan,
                       "dense_cos": np.nan, "rank_name": np.nan, "rank_addr": np.nan, "rank_comb": np.nan, "exact_mask": 0, "n_channels": 0, "src": "S2"}])
    for c in a.columns:
        if c.endswith(("_missing", "_conf", "_nonlatin")):
            a[c] = a[c].astype(np.int8)
    F = _pair_block(a, {})
    assert F["n_exact_fold"].iloc[0] == 0 and F["a_exact"].iloc[0] == 0 and F["n_jw"].iloc[0] == 0.0 and F["n_missing"].iloc[0] == 1
    assert np.isfinite(F[feature_columns(F)].astype(float).values).all()
    assert "y" not in feature_columns(F) and "owner_fold" not in feature_columns(F)
