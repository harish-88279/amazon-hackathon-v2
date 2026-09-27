import pandas as pd
import pytest

from ber.splits import assert_no_leak, identity_groups, target_owner_split


def _truth(rows):
    return pd.DataFrame({"source1_entity_id": [r[0] for r in rows], "matched": [r[1] for r in rows]})


def test_components_merge_shared_targets():
    t = _truth([("S1-1", ["S2-1"]), ("S1-2", ["S2-1", "S3-9"]), ("S1-3", []), ("S1-4", ["S3-4"])])
    g, n_multi = identity_groups(t)
    assert g["S1-1"] == g["S1-2"] != g["S1-3"] and g["S1-3"] != g["S1-4"] and n_multi == 1


def test_leak_assertion():
    t = _truth([("S1-1", ["S2-1"]), ("S1-2", ["S2-1"])])
    man = {"entities": {"S1-1": {"part": "dev", "fold": 0}, "S1-2": {"part": "dev", "fold": 1}}}
    with pytest.raises(AssertionError):
        assert_no_leak(man, t)
    man["entities"]["S1-2"]["fold"] = 0
    assert_no_leak(man, t)
    assert target_owner_split(man, t)["S2-1"] == ("dev", 0)
