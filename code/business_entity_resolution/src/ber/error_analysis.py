"""Assign every validation error its earliest actionable cause and rank by lost F0.5."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .metric import entity_f05


def analyse(pred: dict, truth: dict, ids: list[str], cand: dict[str, set], scored: pd.DataFrame,
            s1: pd.DataFrame, tgt: pd.DataFrame, policy: dict, top_n: int = 200) -> tuple[pd.DataFrame, dict]:
    """Returns (error table sorted by lost score, cause counts)."""
    t = min(float(policy.get("t_s2", 0.5)), float(policy.get("t_s3", 0.5)))
    p_map = {(a, b): float(p) for a, b, p in zip(scored["s1_id"], scored["cand_id"], scored["p"])}
    s1i = s1.set_index("entity_id"); tgi = tgt.set_index("entity_id")
    rows, causes = [], Counter()
    for i in ids:
        g, p, c = truth[i], set(pred.get(i, ())), cand.get(i, set())
        score = entity_f05(g, p)
        if score >= 1.0:
            continue
        lost = 1.0 - score
        missed, extra = g - p, p - g
        if not g and p:
            cause = "false_merge_on_singleton"
        elif missed and (missed - c):
            cause = "missed_by_retrieval"
        elif missed:
            cause = "retrieved_but_scored_low"
        elif extra:
            cause = "false_merge_nonsingleton"
        else:
            cause = "other"
        causes[cause] += lost
        ex = next(iter(extra), None) or next(iter(missed), None)
        rows.append({
            "s1_id": i, "lost": round(lost, 4), "cause": cause, "n_truth": len(g), "n_pred": len(p), "n_cand": len(c),
            "n_missed": len(missed), "n_extra": len(extra),
            "s1_name": s1i.at[i, "business_name"] if i in s1i.index else "", "s1_addr": s1i.at[i, "business_address"] if i in s1i.index else "",
            "example_id": ex, "example_name": tgi.at[ex, "business_name"] if ex in tgi.index else "",
            "example_addr": tgi.at[ex, "business_address"] if ex in tgi.index else "",
            "example_p": p_map.get((i, ex)), "threshold": t,
        })
    df = pd.DataFrame(rows).sort_values("lost", ascending=False) if rows else pd.DataFrame()
    total_lost = sum(causes.values())
    summary = {"total_lost_f05": total_lost, "lost_by_cause": dict(causes),
               "lost_share_by_cause": {k: v / total_lost for k, v in causes.items()} if total_lost else {},
               "n_error_entities": len(rows)}
    return df.head(top_n) if len(df) else df, summary
