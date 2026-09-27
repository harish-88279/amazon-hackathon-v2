"""Metric tests: exact fixtures from the spec, exhaustive tiny universe, two implementations agree."""
import itertools
import random
from math import isclose

import pytest

from ber.metric import entity_f05, entity_f05_pr, macro_f05, summary


def test_spec_fixtures():
    assert entity_f05([], []) == 1.0
    assert entity_f05([], ["x"]) == 0.0
    assert entity_f05(["a"], []) == 0.0
    assert entity_f05(["a"], ["b"]) == 0.0
    assert entity_f05(["a"], ["a"]) == 1.0
    assert isclose(entity_f05(["a", "b"], ["a"]), 5 / 6)
    assert isclose(entity_f05(["a", "b"], ["a", "b", "c"]), 5 / 7)


def test_macro_fixture_and_singleton_denominator():
    truth = {"1": {"a", "b"}, "2": set(), "3": {"z"}}
    pred = {"1": {"a", "b", "c"}, "2": set(), "3": {"q"}}
    assert isclose(macro_f05(truth, pred, ["1", "2", "3"]), 4 / 7)
    # all-empty system scores the singleton proportion
    assert isclose(macro_f05(truth, {k: set() for k in truth}, ["1", "2", "3"]), 1 / 3)


def test_coverage_errors():
    with pytest.raises(ValueError):
        macro_f05({"1": set()}, {"1": set()}, ["1", "1"])
    with pytest.raises(ValueError):
        macro_f05({"1": set()}, {"2": set()}, ["1"])
    with pytest.raises(ValueError):
        macro_f05({"1": set(), "2": set()}, {"1": set()}, ["1"])


def test_two_implementations_agree_exhaustively():
    universe = ["a", "b", "c"]
    subsets = [set(c) for r in range(4) for c in itertools.combinations(universe, r)]
    for g in subsets:
        for p in subsets:
            assert isclose(entity_f05(g, p), entity_f05_pr(g, p))


def test_two_implementations_agree_random():
    rng = random.Random(0)
    ids = [f"x{i}" for i in range(12)]
    for _ in range(500):
        g = set(rng.sample(ids, rng.randint(0, 6)))
        p = set(rng.sample(ids, rng.randint(0, 6)))
        assert isclose(entity_f05(g, p), entity_f05_pr(g, p))


def test_summary_fields():
    truth = {"1": {"a"}, "2": set()}
    s = summary(truth, {"1": {"a"}, "2": {"b"}}, ["1", "2"])
    assert s["singleton_false_merge_rate"] == 1.0 and s["macro_f05"] == 0.5 and s["all_empty_baseline"] == 0.5
