"""Generic (country-agnostic) address component extraction with confidence flags.

We never infer a location from an external database. Components:
  postal          5-6 digit token (US ZIP 5, India PIN 6, France 5) -> string, "" if absent
  house_no        first short numeric token (<=5 digits, optional letter) that is not the postal code
  unit            token following unit/apt/ste/fl/no markers, or "" ; kept separate from house_no
  addr_nums       all numeric sequences in order (space-joined) - raw evidence, '12/3' -> '12 3'
  street_tokens   tokens of the first comma segment minus numbers (street-ish evidence)
  locality_tokens tokens of the last two comma segments (city / region-ish evidence)
  addr_conf       0 = empty, 1 = something parsed, 2 = house number and postal both found
"""
from __future__ import annotations

import re

# Address token abbreviations -> canonical SHORT form. Applied to addr_fold only.
ADDR_ABBREV = {
    "street": "st", "str": "st", "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "place": "pl", "court": "ct", "circle": "cir", "parkway": "pkwy",
    "square": "sq", "terrace": "ter", "trail": "trl", "route": "rte", "expressway": "expy", "north": "n",
    "south": "s", "east": "e", "west": "w", "apartment": "apt", "suite": "ste", "building": "bldg", "floor": "fl",
    "number": "no", "near": "nr", "opposite": "opp", "junction": "jn", "sector": "sec", "block": "blk",
    "extension": "extn", "ext": "extn", "township": "twp", "district": "dist", "taluk": "tq", "taluka": "tq",
    "rue": "r", "chemin": "ch", "allee": "all", "impasse": "imp", "quai": "q", "cours": "crs", "faubourg": "fbg",
    "saint": "st", "sainte": "ste",
}

_DIGITS = re.compile(r"\d+")
_POSTAL = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
_HOUSE = re.compile(r"^\d{1,5}[a-z]?$")
_UNIT_MARKERS = {"unit", "apt", "ste", "fl", "no", "flat", "shop", "office", "room", "door", "plot"}


def parse_address(raw: str, fold: str) -> dict:
    if not fold:
        return {"postal": "", "house_no": "", "unit": "", "addr_nums": "", "street_tokens": "",
                "locality_tokens": "", "addr_conf": 0}
    toks = fold.split()
    nums = _DIGITS.findall(fold)
    postal_m = _POSTAL.search(fold)
    postal = postal_m.group(1) if postal_m else ""
    house, unit = "", ""
    for i, t in enumerate(toks):
        if t in _UNIT_MARKERS and i + 1 < len(toks) and _HOUSE.match(toks[i + 1]):
            unit = toks[i + 1]
    for i, t in enumerate(toks):
        if _HOUSE.match(t) and t != postal and t != unit and not (i > 0 and toks[i - 1] in _UNIT_MARKERS):
            house = t
            break
    # comma segments come from the RAW string (fold has punctuation removed)
    segs = [s.strip() for s in raw.split(",") if s.strip()]
    from .normalize import conservative, fold_accents  # local import to avoid cycle at module load
    first = fold_accents(conservative(segs[0])) if segs else ""
    last2 = fold_accents(conservative(" ".join(segs[-2:]))) if segs else ""
    first = " ".join(ADDR_ABBREV.get(t, t) for t in first.split())
    last2 = " ".join(ADDR_ABBREV.get(t, t) for t in last2.split())
    street = " ".join(t for t in first.split() if not t.isdigit() and t != unit)
    locality = " ".join(sorted(set(t for t in last2.split() if not t.isdigit())))
    conf = 2 if (house and postal) else 1
    return {"postal": postal, "house_no": house, "unit": unit, "addr_nums": " ".join(nums),
            "street_tokens": street, "locality_tokens": locality, "addr_conf": conf}
