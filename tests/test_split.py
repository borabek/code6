"""Tests for the leakage-safe train/val/test split and part-family grouping.

The split is the load-bearing honesty guarantee: if sibling part-family variants
straddle train and val, the reported F1 is inflated by memorised near-duplicates.
These assert that grouping keeps a family on ONE side and that the three-way
split is a clean partition.
"""
import json_dataset as jd
from train_cp import split_part_ids, three_way_split


def test_family_key_strips_variant_suffix():
    assert jd.family_key("ABC-1234-02") == jd.family_key("ABC-1234-05")
    assert jd.family_key("ABC-1234-02") == "ABC-1234"


def test_group_split_keeps_a_family_on_one_side():
    ids = ["P-100-%02d" % i for i in range(20)] + ["Q-200-%02d" % i for i in range(20)]
    keys = jd.build_group_keys(
        [jd.Part(i, [], [], [], [], []) for i in ids], mode="prefix")
    train, val = split_part_ids(ids, val_frac=0.5, seed=0, group_keys=keys)
    fam_p = [x for x in ids if x.startswith("P-")]
    # every P-variant must be entirely in train OR entirely in val, never split
    assert all(x in train for x in fam_p) or all(x in val for x in fam_p)


def test_split_is_deterministic():
    ids = ["part-%03d" % i for i in range(50)]
    assert split_part_ids(ids, seed=0) == split_part_ids(ids, seed=0)


def test_three_way_split_is_a_clean_partition():
    ids = ["part-%03d" % i for i in range(100)]
    train, val, test = three_way_split(ids, val_frac=0.2, test_frac=0.2, seed=1)
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert len(train) + len(val) + len(test) == len(ids)
    assert len(val) > 0 and len(test) > 0          # both carved with these fracs


def test_test_frac_zero_reproduces_two_way():
    ids = ["part-%03d" % i for i in range(60)]
    train2, val2 = split_part_ids(ids, val_frac=0.2, seed=3)
    train3, val3, test3 = three_way_split(ids, val_frac=0.2, test_frac=0.0, seed=3)
    assert test3 == set() and train3 == train2 and val3 == val2
