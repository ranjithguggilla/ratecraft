from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_tweedie_deviance

from ratecraft.metrics import (
    actual_vs_expected,
    balance,
    double_lift,
    gini,
    group_bootstrap,
    improvement,
    lorenz_curve,
    murphy,
    poisson_deviance,
    tweedie_deviance,
    unit_deviance,
)


@pytest.fixture(scope="module")
def book():
    rng = np.random.default_rng(11)
    # Large enough that each of ten bands holds roughly a thousand claims.
    n = 200_000
    exposure = rng.uniform(0.05, 1.0, n)
    true_rate = np.exp(rng.normal(-2.6, 0.6, n))
    counts = rng.poisson(true_rate * exposure).astype(float)
    return counts, true_rate, exposure


@pytest.mark.parametrize("power", [1.0, 1.35, 1.5, 1.8, 2.0])
def test_unit_deviance_matches_sklearn(power):
    rng = np.random.default_rng(int(power * 100))
    y = rng.gamma(1.0, 3.0, 500) * (rng.random(500) < (0.4 if power < 2 else 1.0))
    if power == 2.0:
        y = y + 0.1  # gamma deviance needs y > 0
    mu = rng.uniform(0.5, 4.0, 500)
    w = rng.uniform(0.1, 1.0, 500)
    ours = np.average(unit_deviance(y, mu, power), weights=w)
    assert ours == pytest.approx(mean_tweedie_deviance(y, mu, sample_weight=w, power=power), rel=1e-10)
    assert tweedie_deviance(y, mu, w, power) == pytest.approx(ours, rel=1e-10)


def test_poisson_deviance_is_zero_for_a_perfect_fit():
    y = np.array([0.0, 1.0, 3.0])
    assert poisson_deviance(y, np.where(y > 0, y, 1e-12)) == pytest.approx(0.0, abs=1e-9)


def test_improvement_against_flat():
    assert improvement(90.0, 100.0) == pytest.approx(0.10)


def test_gini_is_zero_for_a_flat_rate_and_rewards_ranking(book):
    counts, true_rate, exposure = book
    assert gini(counts, np.full_like(true_rate, 0.07), exposure) == pytest.approx(0.0, abs=1e-12)
    informed = gini(counts, true_rate, exposure)
    noisy = gini(counts, true_rate * np.exp(np.random.default_rng(1).normal(0, 1.5, true_rate.size)), exposure)
    backwards = gini(counts, 1 / true_rate, exposure)
    assert informed > noisy > 0 > backwards


def test_lorenz_curve_runs_corner_to_corner(book):
    counts, true_rate, exposure = book
    x, y = lorenz_curve(counts, true_rate, exposure)
    assert (x[0], y[0]) == (0.0, 0.0)
    assert x[-1] == pytest.approx(1.0) and y[-1] == pytest.approx(1.0)
    assert np.all(np.diff(x) >= 0) and np.all(np.diff(y) >= 0)


def test_balance():
    assert balance([1.0, 1.0], [1.5, 1.5]) == pytest.approx(1.5)


def test_actual_vs_expected_bands_partition_the_book(book):
    counts, true_rate, exposure = book
    table = actual_vs_expected(counts, true_rate * exposure, exposure, n_bands=10)
    assert len(table) == 10
    assert table["exposure"].sum() == pytest.approx(exposure.sum())
    assert table["actual"].sum() == pytest.approx(counts.sum())
    # equal-exposure bands
    assert table["exposure"].std() / table["exposure"].mean() < 0.01
    # with the true rate as the model, A/E sits near one in every band
    assert table["a_over_e"].between(0.85, 1.15).all()


def test_double_lift_relative_columns_average_to_one(book):
    counts, true_rate, exposure = book
    other = np.full_like(true_rate, true_rate.mean())
    table = double_lift(counts, true_rate * exposure, other * exposure, exposure)
    for col in ("actual_relative", "a_relative", "b_relative"):
        assert np.average(table[col], weights=table["exposure"]) == pytest.approx(1.0, rel=1e-9)
    # sorted by the ratio of a to b, the informed model's relative rate rises
    assert table["a_relative"].is_monotonic_increasing


def test_murphy_decomposition_identity_and_signs(book):
    counts, true_rate, exposure = book
    y = counts / exposure
    for pred in (true_rate, true_rate * 1.3, np.full_like(true_rate, y.mean())):
        out = murphy(y, pred, exposure, power=1.0)
        assert out["score"] == pytest.approx(out["miscalibration"] - out["discrimination"] + out["uncertainty"])
        assert out["miscalibration"] >= -1e-12
        assert out["discrimination"] >= -1e-12
    scaled = murphy(y, true_rate * 1.3, exposure, power=1.0)
    exact = murphy(y, true_rate, exposure, power=1.0)
    assert scaled["miscalibration"] > exact["miscalibration"]
    assert scaled["discrimination"] == pytest.approx(exact["discrimination"], rel=1e-9)


def test_group_bootstrap_resamples_whole_groups():
    groups = np.repeat(np.arange(50), 4)
    values = np.repeat(np.arange(50, dtype=float), 4)
    seen_sizes = []

    def stat(idx):
        seen_sizes.append(idx.size)
        # every resampled group arrives complete: four identical values in a row
        assert np.all(values[idx].reshape(-1, 4).std(axis=1) == 0)
        return values[idx].mean()

    draws = group_bootstrap(stat, groups, n_boot=30, seed=0)
    assert draws.shape == (30,)
    assert set(seen_sizes) == {200}
    again = group_bootstrap(lambda idx: values[idx].mean(), groups, n_boot=30, seed=0)
    assert np.allclose(draws, again)
