"""Data contracts: column names, ID rules, and strict validation helpers shared by parsers,
the output writer and the tests. Nothing here touches the filesystem.
"""
from __future__ import annotations

import re

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CAND_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]
PREFIX = {"s1": "S1-", "s2": "S2-", "s3": "S3-"}
_ID_RE = re.compile(r"^S[123]-[A-Za-z0-9_]+$")


class ContractError(ValueError):
    """Raised for any violation of the data/output contract. Message is actionable."""


def check_id(value: str, source: str | None = None) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"invalid entity id {value!r} (empty or surrounding whitespace)")
    if "\t" in value or "," in value or "\n" in value:
        raise ContractError(f"entity id {value!r} contains a tab, comma or newline")
    if not _ID_RE.match(value):
        raise ContractError(f"entity id {value!r} does not look like S1-/S2-/S3-<id>")
    if source and not value.startswith(PREFIX[source]):
        raise ContractError(f"entity id {value!r} does not carry the {PREFIX[source]} prefix expected for {source}")


def split_id_list(cell: str) -> list[str]:
    """Parse a comma-separated id cell. Empty cell -> []. Raises on blanks/duplicates."""
    if cell is None:
        raise ContractError("id cell is None (parser NA handling is wrong)")
    if cell == "":
        return []
    parts = cell.split(",")
    for p in parts:
        if p == "" or p != p.strip():
            raise ContractError(f"malformed id list {cell!r}: empty element or surrounding whitespace")
    if len(set(parts)) != len(parts):
        raise ContractError(f"duplicate ids inside list {cell!r}")
    return parts


def is_null_like(cell: str) -> bool:
    return cell.strip().lower() in {"nan", "none", "null", "[]", "n/a", "na"} and cell != ""
