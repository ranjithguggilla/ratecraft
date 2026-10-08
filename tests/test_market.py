from __future__ import annotations

import numpy as np
import pytest

from ratecraft.config import CostModel
from ratecraft.market import challenger_share, simulate, to_frame

COSTS = CostModel(target_loss_ratio=0.7, expense_ratio=0.25, price_elasticity=4.0, book_policy_years=1_000)


def test_cheapest_quote_wins_and_ties_split():
    share = challenger_share([100, 100, 100], [90, 110, 100], None)
    assert share.tolist() == [1.0, 0.0, 0.5]


def test_logit_demand_matches_the_documented_example():
    assert challenger_share([100], [100], 4.0)[0] == pytest.approx(0.5)
    # a quote 10% dearer than the rival's wins 41% instead of 50%
    assert challenger_share([100], [110], 4.0)[0] == pytest.approx(1 / (1 + 1.1**4))
    assert challenger_share([100], [110], 4.0)[0] == pytest.approx(0.4058, abs=1e-4)


def test_identical_tariffs_split_the_market_evenly():
    rng = np.random.default_rng(0)
    tech = rng.uniform(50, 500, 1_000)
    claims = rng.gamma(0.5, tech)
    exposure = np.ones(1_000)
    a, b = simulate(("a", "b"), (tech, tech), claims, exposure, COSTS, None)
    assert a.share_of_customers == pytest.approx(0.5)
    assert a.loss_ratio == pytest.approx(b.loss_ratio)
    assert a.premium + b.premium == pytest.approx(np.sum(tech / 0.7))


def test_the_cruder_tariff_is_adversely_selected():
    rng = np.random.default_rng(1)
    true_cost = np.exp(rng.normal(5, 0.8, 20_000))
    claims = rng.poisson(true_cost / 200) * 200.0
    exposure = np.ones_like(true_cost)
    flat = np.full_like(true_cost, true_cost.mean())
    for elasticity in (None, 4.0):
        crude, sharp = simulate(("flat", "sharp"), (flat, true_cost), claims, exposure, COSTS, elasticity)
        assert crude.loss_ratio > 0.8 > sharp.loss_ratio
        assert crude.result < 0 < sharp.result


def test_frame_scales_results_to_the_book_size():
    tech = np.array([100.0, 100.0])
    books = simulate(("a", "b"), (tech, tech * 2), np.array([10.0, 10.0]), np.array([0.5, 0.5]), COSTS, None)
    frame = to_frame(books, "s", COSTS)
    a = frame.iloc[0]
    assert a["earned_policy_years"] == pytest.approx(1.0)
    assert a["result_per_book"] == pytest.approx(a["underwriting_result"] * 1_000)
    assert a["combined_ratio"] == pytest.approx(a["loss_ratio"] + 0.25)
