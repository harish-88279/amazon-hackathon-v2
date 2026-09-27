"""Dataset access with a content-addressed cache of normalised tables.

Cache key = sha256(source file) + NORMALIZER_VERSION. Normalisation streams in row chunks and
writes Parquet with zstd; loads use Arrow-backed strings and optional country push-down filters
so a 5M-row source costs a few hundred MB instead of several GB.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import io as bio
from .config import Config
from .normalize import ADDR_COLS, NAME_COLS, NORMALIZER_VERSION, normalize_frame

log = logging.getLogger(__name__)
NORM_CHUNK = 400_000
ALL_COLS = ["entity_id", "business_name", "business_address", "country", "country_norm"] + NAME_COLS + ADDR_COLS


def source_path(cfg: Config, split: str, source: str) -> Path:
    return bio.dataset_paths(cfg.data_root)[f"{split}_{source}"]


def _cache_path(cfg: Config, split: str, source: str) -> Path:
    p = source_path(cfg, split, source)
    key = bio.sha256_file(p)[:16]
    return cfg.cache_root / "norm" / f"{split}_{source}_{key}_{NORMALIZER_VERSION}.parquet"


def ensure_normalized(cfg: Config, split: str, source: str, force: bool = False) -> Path:
    out = _cache_path(cfg, split, source)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    src = source_path(cfg, split, source)
    writer, n = None, 0
    tmp = out.with_suffix(".tmp.parquet")
    for chunk in bio.read_tsv(src, bio.C.SOURCE_COLUMNS, chunksize=NORM_CHUNK):
        table = pa.Table.from_pandas(normalize_frame(chunk), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
        writer.write_table(table)
        n += len(chunk)
        log.info("normalised %s/%s: %d rows", split, source, n)
    if writer is None:
        raise bio.C.ContractError(f"{src}: no rows")
    writer.close()
    tmp.replace(out)
    return out


def load_norm(cfg: Config, split: str, source: str, columns: list[str] | None = None,
              country: str | None = None) -> pd.DataFrame:
    """Load the normalised table (Arrow strings). ``country`` filters on country_norm (push-down)."""
    p = ensure_normalized(cfg, split, source)
    filters = [("country_norm", "==", country)] if country is not None else None
    df = pd.read_parquet(p, columns=columns, filters=filters, dtype_backend="pyarrow")
    for c in ("name_missing", "addr_missing", "addr_conf", "name_nonlatin", "addr_nonlatin"):
        if c in df:
            df[c] = df[c].astype("int8")
    return df.reset_index(drop=True)


def load_truth(cfg: Config) -> pd.DataFrame:
    return bio.read_truth(bio.dataset_paths(cfg.data_root)["train_truth"])


def s1_ids_in_order(cfg: Config, split: str) -> list[str]:
    """Authoritative row universe for outputs: the S1 file in input order."""
    return bio.read_tsv(source_path(cfg, split, "s1"), bio.C.SOURCE_COLUMNS)["entity_id"].tolist()


def countries(cfg: Config, split: str) -> list[str]:
    s = pd.read_parquet(ensure_normalized(cfg, split, "s1"), columns=["country_norm"])["country_norm"]
    return sorted(s.unique().tolist())
