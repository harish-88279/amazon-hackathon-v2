"""Parsing / serialisation contract tests."""
from pathlib import Path

import pytest

from ber import contracts as C
from ber import io as bio


def _w(p: Path, text: str, enc="utf-8"):
    p.write_text(text, encoding=enc, newline="")
    return p


def test_read_source_ok_with_bom_accents_commas_and_empty_final_cell(tmp_path):
    p = _w(tmp_path / "s.tsv", "﻿entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tAtelier Lumière\t12 rue, Paris\tFrance\nS1-2\t\t\t\n")
    df = bio.read_source(p, "s1")
    assert list(df.columns) == C.SOURCE_COLUMNS
    assert df.loc[0, "business_address"] == "12 rue, Paris" and df.loc[1, "business_name"] == "" and df.loc[1, "country"] == ""


def test_one_column_file_rejected(tmp_path):
    p = _w(tmp_path / "s.tsv", "entity_id,business_name,business_address,country\nS1-1,x,y,US\n")
    with pytest.raises(C.ContractError, match="not tab separated"):
        bio.read_source(p, "s1")


def test_missing_column_duplicate_and_wrong_prefix(tmp_path):
    p = _w(tmp_path / "a.tsv", "entity_id\tbusiness_name\tcountry\nS1-1\tx\tUS\n")
    with pytest.raises(C.ContractError, match="expected columns"):
        bio.read_source(p, "s1")
    p = _w(tmp_path / "b.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tx\ty\tUS\nS1-1\tx\ty\tUS\n")
    with pytest.raises(C.ContractError, match="duplicate"):
        bio.read_source(p, "s1")
    p = _w(tmp_path / "c.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-1\tx\ty\tUS\n")
    with pytest.raises(C.ContractError, match="prefix"):
        bio.read_source(p, "s1")


def test_truth_parsing_and_errors(tmp_path):
    p = _w(tmp_path / "t.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\nS1-3\tS2-9,S2-9\n")
    df = bio.read_truth(p)
    assert df.loc[1, "matched"] == [] and df.loc[0, "matched"] == ["S2-1", "S3-1"] and df.attrs["duplicates_inside_lists"] == 1
    p = _w(tmp_path / "d.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-1\t\n")
    with pytest.raises(C.ContractError, match="duplicate"):
        bio.read_truth(p)
    p = _w(tmp_path / "n.tsv", "source1_entity_id\tmatched_entity_ids\nS1-1\tnan\n")
    with pytest.raises(C.ContractError, match="null-like"):
        bio.read_truth(p)


def test_leading_zero_postal_preserved_as_string(tmp_path):
    p = _w(tmp_path / "s.tsv", "entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-1\tx\t01234 Main St\tUS\n")
    assert bio.read_source(p, "s2").loc[0, "business_address"].startswith("01234")


def test_output_bytes_exact(tmp_path):
    out = tmp_path / "m.tsv"
    bio.write_id_list_tsv(out, "matched_entity_ids", ["S1-a", "S1-b"], {"S1-a": ["S3-c", "S2-a", "S2-b"]})
    assert out.read_bytes() == b"source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a,S2-b,S3-c\nS1-b\t\n"
    back = bio.read_id_list_tsv(out, "matched_entity_ids")
    assert back == {"S1-a": ["S2-a", "S2-b", "S3-c"], "S1-b": []}


def test_output_rejects_duplicates_and_s1_targets(tmp_path):
    with pytest.raises(C.ContractError):
        bio.write_id_list_tsv(tmp_path / "x.tsv", "matched_entity_ids", ["S1-a"], {"S1-a": ["S2-a", "S2-a"]})
    with pytest.raises(C.ContractError):
        bio.write_id_list_tsv(tmp_path / "y.tsv", "matched_entity_ids", ["S1-a"], {"S1-a": ["S1-b"]})


def test_split_id_list_rules():
    assert C.split_id_list("") == []
    assert C.split_id_list("S2-1,S3-2") == ["S2-1", "S3-2"]
    for bad in ("S2-1,,S3-2", "S2-1, S3-2", "S2-1,S2-1"):
        with pytest.raises(C.ContractError):
            C.split_id_list(bad)
