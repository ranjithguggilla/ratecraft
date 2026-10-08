from __future__ import annotations

import numpy as np
import pytest

from ratecraft.split import grouped_split, profile_groups, random_split


def test_identical_rating_factors_share_a_group(table):
    doubled = table.iloc[[0, 1, 0, 2]].reset_index(drop=True)
    groups = profile_groups(doubled)
    assert groups[0] == groups[2]
    assert len(set(groups)) == 3


def test_no_group_straddles_two_folds():
    groups = np.random.default_rng(0).permutation(np.repeat(np.arange(500), 10))
    sp = grouped_split(groups, (0.6, 0.2, 0.2), seed=1)
    folds = [set(groups[sp.train]), set(groups[sp.valid]), set(groups[sp.test])]
    assert not (folds[0] & folds[1]) and not (folds[0] & folds[2]) and not (folds[1] & folds[2])
    assert sp.train.size + sp.valid.size + sp.test.size == groups.size
    assert len(folds[0]) == 300 and len(folds[1]) == 100


def test_grouped_split_is_reproducible_and_seed_dependent():
    groups = np.repeat(np.arange(200), 3)
    a = grouped_split(groups, (0.6, 0.2, 0.2), seed=4)
    b = grouped_split(groups, (0.6, 0.2, 0.2), seed=4)
    c = grouped_split(groups, (0.6, 0.2, 0.2), seed=5)
    assert np.array_equal(a.test, b.test)
    assert not np.array_equal(a.test, c.test)


def test_random_split_covers_every_row_once():
    sp = random_split(1_000, (0.6, 0.2, 0.2), seed=0)
    allrows = np.concatenate([sp.train, sp.valid, sp.test])
    assert np.array_equal(np.sort(allrows), np.arange(1_000))
    assert sp.sizes() == {"train": 600, "valid": 200, "test": 200}


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to one"):
        random_split(10, (0.5, 0.2, 0.2), seed=0)
