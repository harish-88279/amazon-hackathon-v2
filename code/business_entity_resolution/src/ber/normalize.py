"""Text views for names and addresses. Raw fields are never overwritten.

Views produced per record (all deterministic, local, country-agnostic):
  name_raw            original string
  name_cons           conservative: NFKC, casefold, whitespace/punct split, digits kept, '&'->'and'
  name_fold           accent-folded name_cons (retrieval view; original evidence stays in name_cons)
  name_sorted         tokens of name_fold sorted (word-order invariant)
  name_core           name_fold minus legal-form tokens (supplementary view, never the only one)
  name_legal          legal-form tokens found (e.g. "pvt ltd", "sarl") joined by space
  name_key            name_core without spaces (catches concatenated / domain style names)
  name_acronym        first letters of name_core tokens (>=2 tokens) else ""
  name_meta           metaphone of name_core
  addr_cons / addr_fold  analogous conservative + folded address views (abbreviations are NOT
                      expanded in the name; address abbreviations are mapped to short forms in
                      addr_fold only, so 'St' inside a business name is untouched)
  plus parsed components from parse_address.py

NORMALIZER_VERSION is part of every cache key.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .parse_address import ADDR_ABBREV, parse_address

NORMALIZER_VERSION = "v3"  # v3: anyascii transliteration of non-Latin scripts, Indic vowel signs preserved

try:
    import jellyfish

    def _metaphone(s: str) -> str:
        return jellyfish.metaphone(s) if s else ""
except Exception:  # pragma: no cover
    def _metaphone(s: str) -> str:
        return re.sub(r"[aeiou]", "", s)[:10]

# legal-form tokens after folding (multi-word first). These are removed only in name_core.
LEGAL_FORMS = [
    "private limited", "pvt limited", "pvt ltd", "public limited company", "limited liability partnership",
    "limited liability company", "societe anonyme", "sarl", "sasu", "eurl", "snc", "sas", "gmbh", "llc",
    "llp", "plc", "ltda", "incorporated", "corporation", "corp", "limited", "ltd", "inc", "pvt", "co",
    "company", "lp", "opc", "pty", "bv", "nv", "ag", "sa", "srl", "spa",
]
_LEGAL_RE = re.compile(r"\b(" + "|".join(re.escape(x) for x in sorted(LEGAL_FORMS, key=len, reverse=True)) + r")\b")

_WS = re.compile(r"\s+")
# punctuation -> space, but keep letters, digits and whitespace; '/' '-' become spaces so 12/3 -> "12 3" (not 123)
# punctuation/symbols -> space; keep word chars, whitespace, combining marks and Indic vowel signs (U+0900-U+0DFF)
_PUNCT = re.compile(r"[^\w\s\u0300-\u036F\u0900-\u0DFF]", re.UNICODE)
_DOMAIN = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|in|fr|net|org|co|io|biz|info)\b")


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


try:
    from anyascii import anyascii as _anyascii  # ISC licence; deterministic local transliteration
except Exception:  # pragma: no cover
    _anyascii = None


def fold_accents(s: str) -> str:
    """Accent folding; non-Latin scripts (Devanagari, Telugu, ...) are transliterated to ASCII
    deterministically and locally (anyascii). The conservative view keeps the original script."""
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    if _anyascii is not None and not s.isascii():
        s = _anyascii(s).casefold()
        s = _PUNCT.sub(" ", s)
        s = _WS.sub(" ", s).strip()
    return s


def script_flag(s: str) -> int:
    """0 = ascii/latin only, 1 = contains non-Latin script characters."""
    return int(any(ord(c) > 0x2FF and c.isalpha() for c in s))


def conservative(s: str) -> str:
    """NFKC + casefold + '&'->'and' + punctuation to spaces + collapse. Digits preserved."""
    s = nfkc(s).casefold()
    s = s.replace("&", " and ").replace("+", " plus ")
    s = _DOMAIN.sub(r"\1", s)                 # acmefoods.com -> acmefoods
    s = s.replace("'", "").replace("’", "")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def name_views(raw: str) -> dict[str, str]:
    cons = conservative(raw)
    fold = fold_accents(cons)
    legal = " ".join(sorted(set(_LEGAL_RE.findall(fold))))
    core = _WS.sub(" ", _LEGAL_RE.sub(" ", fold)).strip() or fold
    toks = core.split()
    return {
        "name_cons": cons,
        "name_fold": fold,
        "name_sorted": " ".join(sorted(fold.split())),
        "name_core": core,
        "name_legal": legal,
        "name_key": core.replace(" ", ""),
        "name_acronym": "".join(t[0] for t in toks) if len(toks) >= 2 else "",
        "name_meta": _metaphone(core),
        "name_missing": int(cons == ""),
        "name_nonlatin": script_flag(cons),
    }


def address_views(raw: str) -> dict:
    cons = conservative(raw)
    fold = fold_accents(cons)
    fold = " ".join(ADDR_ABBREV.get(t, t) for t in fold.split())
    parsed = parse_address(raw, fold)
    return {"addr_cons": cons, "addr_fold": fold, "addr_missing": int(cons == ""), "addr_nonlatin": script_flag(cons), **parsed}


NAME_COLS = ["name_cons", "name_fold", "name_sorted", "name_core", "name_legal", "name_key", "name_acronym",
             "name_meta", "name_missing", "name_nonlatin"]
ADDR_COLS = ["addr_cons", "addr_fold", "addr_missing", "addr_nonlatin", "postal", "house_no", "unit", "addr_nums", "street_tokens",
             "locality_tokens", "addr_conf"]


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add all views to a source frame (raw columns kept)."""
    out = df.copy()
    nv = pd.DataFrame([name_views(x) for x in out["business_name"].tolist()], index=out.index)
    av = pd.DataFrame([address_views(x) for x in out["business_address"].tolist()], index=out.index)
    out = pd.concat([out, nv[NAME_COLS], av[ADDR_COLS]], axis=1)
    out["country_norm"] = out["country"].map(lambda c: fold_accents(conservative(c)))
    for c in ("name_missing", "addr_missing", "addr_conf", "name_nonlatin", "addr_nonlatin"):
        out[c] = out[c].astype("int8")
    return out
