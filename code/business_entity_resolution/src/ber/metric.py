"""Exact competition metric: macro F0.5 over Source-1 entities with the singleton rule.

Two independent implementations are provided and cross-checked by the tests:
  * :func:`entity_f05` - count form   1.25*tp / (0.25*|G| + |P|)
  * :func:`entity_f05_pr` - precision/recall form with explicit empty handling
Both operate on already-parsed sets; the validator must reject duplicates *before* scoring.
"""
from __future__ import annotations

from typing import Iterable, Mapping


def entity_f05(truth: Iterable[str], prediction: Iterable[str]) -> float:
    g, p = set(truth), set(prediction)
    if not g:
        return float(not p)
    tp = len(g & p)
    return 1.25 * tp / (0.25 * len(g) + len(p))


def entity_f05_pr(truth: Iterable[str], prediction: Iterable[str], beta: float = 0.5) -> float:
    g, p = set(truth), set(prediction)
    if not g:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    tp = len(g & p)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def macro_f05(truth_by_id: Mapping[str, Iterable[str]], pred_by_id: Mapping[str, Iterable[str]],
              expected_ids: Iterable[str]) -> float:
    ids = list(expected_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Expected IDs must be unique and nonempty")
    if set(truth_by_id) != set(ids):
        raise ValueError("Truth coverage mismatch")
    if set(pred_by_id) != set(ids):
        raise ValueError("Prediction coverage mismatch")
    return sum(entity_f05(truth_by_id[i], pred_by_id[i]) for i in ids) / len(ids)


def per_entity_scores(truth_by_id: Mapping[str, Iterable[str]], pred_by_id: Mapping[str, Iterable[str]],
                      expected_ids: Iterable[str]) -> dict[str, float]:
    """Per-entity scores; prediction defaults to empty for ids absent from ``pred_by_id``."""
    return {i: entity_f05(truth_by_id[i], pred_by_id.get(i, ())) for i in expected_ids}


def summary(truth_by_id: Mapping[str, set], pred_by_id: Mapping[str, Iterable[str]], expected_ids: Iterable[str]) -> dict:
    """Macro F0.5 plus diagnostics (pair P/R, singleton behaviour, non-singleton recall)."""
    ids = list(expected_ids)
    scores = per_entity_scores(truth_by_id, pred_by_id, ids)
    tp = fp = fn = 0
    single_n = single_ok = 0
    nonsingle_full = nonsingle_n = 0
    for i in ids:
        g, p = set(truth_by_id[i]), set(pred_by_id.get(i, ()))
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        if not g:
            single_n += 1; single_ok += int(not p)
        else:
            nonsingle_n += 1; nonsingle_full += int(g <= p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "macro_f05": sum(scores.values()) / len(ids) if ids else 0.0,
        "n_entities": len(ids),
        "n_singletons": single_n,
        "singleton_accuracy": single_ok / single_n if single_n else None,
        "singleton_false_merge_rate": 1 - single_ok / single_n if single_n else None,
        "nonsingleton_all_matches_found": nonsingle_full / nonsingle_n if nonsingle_n else None,
        "pair_precision": prec, "pair_recall": rec,
        "all_empty_baseline": single_n / len(ids) if ids else 0.0,
    }
