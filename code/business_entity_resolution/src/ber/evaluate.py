"""Slice evaluation of complete prediction sets against the exact metric."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .metric import entity_f05, summary


def _bucket(n: int) -> str:
    return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4+"


def slices(pred: dict, truth: dict, ids: list[str], country: dict[str, str], addr_missing: dict[str, int] | None = None) -> dict:
    per = {i: entity_f05(truth[i], pred.get(i, ())) for i in ids}
    out = {"overall": summary(truth, pred, ids)}
    by = defaultdict(list)
    for i in ids:
        by[("country", country.get(i, ""))].append(per[i])
        by[("bucket", _bucket(len(truth[i])))].append(per[i])
        if addr_missing is not None:
            by[("s1_addr_missing", str(addr_missing.get(i, 0)))].append(per[i])
    for (kind, key), v in sorted(by.items()):
        out.setdefault(kind, {})[key] = {"n": len(v), "macro_f05": float(np.mean(v))}
    # source-pair behaviour
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for i in ids:
        g, p = truth[i], set(pred.get(i, ()))
        for t in g & p: tp[t[:2]] += 1
        for t in p - g: fp[t[:2]] += 1
        for t in g - p: fn[t[:2]] += 1
    out["by_source"] = {s: {"tp": tp[s], "fp": fp[s], "fn": fn[s],
                            "precision": tp[s] / (tp[s] + fp[s]) if tp[s] + fp[s] else None,
                            "recall": tp[s] / (tp[s] + fn[s]) if tp[s] + fn[s] else None} for s in ("S2", "S3")}
    worst = min(((k, v["macro_f05"]) for kind in ("country", "bucket") for k, v in out.get(kind, {}).items()), key=lambda x: x[1], default=None)
    out["worst_slice"] = worst
    return out
