"""Strict TSV I/O, data manifests and output serialisation.

Rules enforced here (see contracts.py):
  * tab delimiter only; one-column result => malformed schema error, never a guessed delimiter
  * everything is read as str; empty cells stay ""; no NaN coercion; BOM stripped from the header only
  * required columns validated before access; malformed lines reported with file + line number
  * ids validated (prefix, no whitespace/tab/comma); duplicate record ids rejected
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
from pathlib import Path
from typing import Iterator

import pandas as pd

from . import contracts as C

log = logging.getLogger(__name__)

TSV_KW = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=csv.QUOTE_NONE,
              encoding="utf-8", engine="c")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _strip_bom(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.columns) and df.columns[0].startswith("﻿"):
        df = df.rename(columns={df.columns[0]: df.columns[0].lstrip("﻿")})
    return df


def _check_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    if list(df.columns[: len(required)]) != required or len(df.columns) != len(required):
        if len(df.columns) == 1:
            raise C.ContractError(f"{path}: only one column parsed -> file is not tab separated or header is wrong: {list(df.columns)}")
        raise C.ContractError(f"{path}: expected columns {required}, found {list(df.columns)}")


def read_tsv(path: Path, required: list[str], chunksize: int | None = None):
    """Read a challenge TSV strictly. Returns a DataFrame, or an iterator of chunks."""
    path = Path(path)
    if not path.exists():
        raise C.ContractError(f"missing file: {path}")
    if chunksize:
        def gen() -> Iterator[pd.DataFrame]:
            first = True
            for i, ch in enumerate(pd.read_csv(path, chunksize=chunksize, on_bad_lines="error", **TSV_KW)):
                ch = _strip_bom(ch) if first else ch
                if first:
                    _check_columns(ch, required, path)
                    first = False
                yield ch
        return gen()
    try:
        df = pd.read_csv(path, on_bad_lines="error", **TSV_KW)
    except pd.errors.ParserError as e:  # pandas reports the line number in the message
        raise C.ContractError(f"{path}: malformed line -> {e}") from e
    df = _strip_bom(df)
    _check_columns(df, required, path)
    return df


def read_source(path: Path, source: str, validate_ids: bool = True) -> pd.DataFrame:
    df = read_tsv(path, C.SOURCE_COLUMNS)
    if validate_ids:
        ids = df["entity_id"]
        bad = ids[(ids == "") | (ids != ids.str.strip())]
        if len(bad):
            raise C.ContractError(f"{path}: {len(bad)} blank/whitespace entity ids, first at line {bad.index[0] + 2}")
        wrong = ids[~ids.str.startswith(C.PREFIX[source])]
        if len(wrong):
            raise C.ContractError(f"{path}: {len(wrong)} ids without the {C.PREFIX[source]} prefix, e.g. {wrong.iloc[0]!r} at line {wrong.index[0] + 2}")
        dup = ids[ids.duplicated()]
        if len(dup):
            raise C.ContractError(f"{path}: {len(dup)} duplicate entity ids, e.g. {dup.iloc[0]!r} at line {dup.index[0] + 2}")
    return df


def read_truth(path: Path) -> pd.DataFrame:
    """Ground truth as a DataFrame with a parsed ``matched`` column (list[str]).

    Duplicates inside a raw label list are flagged (logged + counted) but the set is kept,
    per the challenge definition. Duplicate S1 rows are an error.
    """
    df = read_tsv(path, C.TRUTH_COLUMNS)
    s1 = df["source1_entity_id"]
    if (s1 == "").any():
        raise C.ContractError(f"{path}: blank source1_entity_id at line {(s1 == '').idxmax() + 2}")
    dup = s1[s1.duplicated()]
    if len(dup):
        raise C.ContractError(f"{path}: duplicate ground-truth rows for {dup.iloc[0]!r} (line {dup.index[0] + 2}); {len(dup)} total")
    matched, n_dup_inside, n_null_like = [], 0, 0
    for cell in df["matched_entity_ids"]:
        if C.is_null_like(cell):
            n_null_like += 1
        parts = [p.strip() for p in cell.split(",")] if cell else []
        parts = [p for p in parts if p]
        if len(set(parts)) != len(parts):
            n_dup_inside += 1
        matched.append(sorted(set(parts)))
    if n_null_like:
        raise C.ContractError(f"{path}: {n_null_like} null-like label cells (e.g. 'nan') - labels must be empty strings")
    df = df.assign(matched=matched)
    df.attrs["duplicates_inside_lists"] = n_dup_inside
    if n_dup_inside:
        log.warning("%s: %d label lists contain repeated ids (kept as sets)", path, n_dup_inside)
    return df


def truth_sets(df: pd.DataFrame) -> dict[str, set[str]]:
    return {a: set(b) for a, b in zip(df["source1_entity_id"], df["matched"])}


# ------------------------------------------------------------------ manifest
def dataset_paths(data_root: Path) -> dict[str, Path]:
    return {
        "train_s1": data_root / "train" / "train_source1.tsv",
        "train_s2": data_root / "train" / "train_source2.tsv",
        "train_s3": data_root / "train" / "train_source3.tsv",
        "train_truth": data_root / "train" / "train_ground_truth.tsv",
        "test_s1": data_root / "test" / "test_source1.tsv",
        "test_s2": data_root / "test" / "test_source2.tsv",
        "test_s3": data_root / "test" / "test_source3.tsv",
    }


def environment_info() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "machine": platform.machine()}
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        info["ram_gb"] = round(vm.total / 2**30, 1)
    except Exception:
        try:
            info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        except Exception:
            info["ram_gb"] = None
    try:
        import shutil
        info["free_disk_gb"] = round(shutil.disk_usage(os.getcwd()).free / 2**30, 1)
    except Exception:
        info["free_disk_gb"] = None
    info["gpu"] = "none detected (CPU pipeline)"
    return info


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ outputs
def write_id_list_tsv(path: Path, list_col: str, s1_ids: list[str], mapping: dict[str, list[str] | set[str]],
                      atomic: bool = True) -> Path:
    """Write one of the two output files with exact bytes: header, one literal tab, comma-joined sorted ids.

    Row universe = ``s1_ids`` in the given (input) order. Ids are validated; duplicates inside a
    list are a bug upstream and raise (the caller must dedupe and log before calling).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp") if atomic else path
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{list_col}\n")
        for s1 in s1_ids:
            C.check_id(s1, "s1")
            ids = list(mapping.get(s1, ()))
            if len(set(ids)) != len(ids):
                raise C.ContractError(f"duplicate ids in list for {s1}: upstream join bug")
            for x in ids:
                C.check_id(x)
                if x.startswith("S1-"):
                    raise C.ContractError(f"S1 id {x} inside the list for {s1}")
            f.write(f"{s1}\t{','.join(sorted(ids))}\n")
    if atomic:
        os.replace(tmp, path)
    return path


def read_id_list_tsv(path: Path, list_col: str) -> dict[str, list[str]]:
    """Strict reader for our own output files (used by validate)."""
    df = read_tsv(path, ["source1_entity_id", list_col])
    out: dict[str, list[str]] = {}
    for i, (a, b) in enumerate(zip(df["source1_entity_id"], df[list_col])):
        if a in out:
            raise C.ContractError(f"{path}: duplicate row for {a} at line {i + 2}")
        if C.is_null_like(b):
            raise C.ContractError(f"{path}: null-like cell {b!r} at line {i + 2}")
        out[a] = C.split_id_list(b)
    return out
