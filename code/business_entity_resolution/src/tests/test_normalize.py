from ber.normalize import address_views, conservative, fold_accents, name_views


def test_idempotent_and_raw_preserved():
    for s in ("Kaveri Precision Pvt. Ltd.", "ATELIER LUMIÈRE SARL", "12/3 Lake Road"):
        c = conservative(s)
        assert conservative(c) == c
    v = name_views("Kaveri Precision Private Limited")
    assert v["name_core"] == "kaveri precision" and v["name_legal"] == "private limited" and v["name_fold"] == "kaveri precision private limited"


def test_digits_and_municipal_numbering_not_collapsed():
    a, b = address_views("12/3 Lake Road, Pune"), address_views("123 Lake Road, Pune")
    assert a["addr_fold"] != b["addr_fold"] and a["addr_nums"] == "12 3" and b["addr_nums"] == "123"
    assert a["house_no"] == "12" and b["house_no"] == "123"


def test_unit_vs_house_number():
    v = address_views("Unit 4, 12 MG Road, Chennai 600001")
    assert v["unit"] == "4" and v["house_no"] == "12" and v["postal"] == "600001"


def test_st_in_name_not_expanded_and_abbrev_in_address_mapped():
    assert name_views("St. Mary's Bakery")["name_core"] == "st marys bakery"
    assert address_views("12 Fremont Street")["addr_fold"] == "12 fremont st"
    assert address_views("12 FREMONT SAINT")["addr_fold"] == "12 fremont st"


def test_accents_and_scripts_folded_but_conservative_view_keeps_them():
    v = name_views("Atelier Lumière")
    assert v["name_cons"] == "atelier lumière" and v["name_fold"] == "atelier lumiere"
    v = name_views("इंडो फूड्स लिमिटेड")
    assert v["name_nonlatin"] == 1 and v["name_cons"].startswith("इंडो") and v["name_fold"].isascii() and v["name_fold"]


def test_empty_fields_are_explicitly_missing():
    n, a = name_views(""), address_views("")
    assert n["name_missing"] == 1 and a["addr_missing"] == 1 and a["addr_conf"] == 0 and a["postal"] == ""


def test_domain_style_name_maps_to_key():
    assert name_views("maurewilliamscolombier.com")["name_key"] == name_views("Maure Williams Colombier Inc")["name_key"]
