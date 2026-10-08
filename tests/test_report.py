from __future__ import annotations

import pytest

from ratecraft.report import BOOSTED, FLAT, GLM_TARIFF, HYBRID, TWEEDIE_GBM, TWEEDIE_GLM, select


def scores(tariff, tweedie_glm, hybrid, boosted, tweedie_gbm, flat=10.0):
    """Validation deviances given as improvements over a flat score of 10."""
    return {FLAT: flat, GLM_TARIFF: flat - tariff, TWEEDIE_GLM: flat - tweedie_glm, HYBRID: flat - hybrid,
            BOOSTED: flat - boosted, TWEEDIE_GBM: flat - tweedie_gbm}


def test_each_tier_has_to_earn_its_complexity():
    shipped, decisions = select(scores(1.0, 0.9, 1.1, 1.2, 1.15))
    assert shipped == BOOSTED
    assert [d["promoted"] for d in decisions] == [True, True]
    assert decisions[1]["needed"] == pytest.approx(1.1 * 1.05)


def test_a_challenger_that_is_better_but_not_by_enough_stays_out():
    shipped, decisions = select(scores(1.0, 0.95, 1.04, 1.045, 1.0))
    assert shipped == GLM_TARIFF
    assert [d["promoted"] for d in decisions] == [False, False]


def test_a_tier_can_be_skipped():
    shipped, decisions = select(scores(1.0, 0.9, 1.01, 1.2, 1.0))
    assert decisions[0]["promoted"] is False
    assert shipped == BOOSTED


def test_the_better_member_of_a_tier_represents_it():
    shipped, _ = select(scores(0.9, 1.0, 0.95, 0.9, 0.9))
    assert shipped == TWEEDIE_GLM
