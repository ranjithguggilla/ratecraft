from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ratecraft import config as cfg
from ratecraft.data import attach_losses
from ratecraft.features import FeatureSpec
from ratecraft.glm import factors_of, fit_component, formula, relativities, search_interactions, select_alpha
from ratecraft.synth import TRUE_EFFECTS, make_portfolio


@pytest.fixture(scope="module")
def big():
    policies, claims = make_portfolio(60_000, seed=5)
    table = attach_losses(policies, claims, 50_000.0, 1.0)
    frame = FeatureSpec.from_policies(table).glm_frame(table)
    return table, frame


def test_formula_reads_like_a_rating_plan():
    rhs = formula(("drv_age", "veh_gas"), ("log_density",), (("drv_age", "veh_gas"),))
    assert rhs == "C(drv_age) + C(veh_gas) + log_density + C(drv_age):C(veh_gas)"
    assert factors_of(rhs) == ["drv_age", "veh_gas", "log_density"]


def test_offset_poisson_glm_recovers_the_simulated_effects(big):
    table, frame = big
    counts = table["n_claims"].to_numpy(float)
    comp = fit_component(frame, counts, power=1.0, factors=cfg.GLM_FACTORS, numeric=cfg.GLM_NUMERIC,
                         alpha=0.0, offset=np.log(table["exposure"].to_numpy()))
    coefs = comp.coefficients()
    assert coefs["C(veh_gas)[Diesel]"] == pytest.approx(TRUE_EFFECTS["diesel"], abs=0.05)
    assert coefs["log_density"] == pytest.approx(TRUE_EFFECTS["log_density_slope"], abs=0.015)
    assert coefs["C(drv_age)[21-25]"] == pytest.approx(TRUE_EFFECTS["young_driver"], abs=0.15)
    # bonus-malus enters as 1.6 * log(BM / 50); the 100-125 band should land in that range
    assert 1.6 * np.log(100 / 50) - 0.2 < coefs["C(bonus_malus)[100-125]"] < 1.6 * np.log(125 / 50) + 0.2


def test_canonical_poisson_glm_balances_on_its_training_data(big):
    table, frame = big
    counts = table["n_claims"].to_numpy(float)
    exposure = table["exposure"].to_numpy()
    comp = fit_component(frame, counts, power=1.0, factors=cfg.GLM_FACTORS, numeric=cfg.GLM_NUMERIC,
                         alpha=0.0, offset=np.log(exposure))
    assert np.sum(comp.predict(frame) * exposure) == pytest.approx(counts.sum(), rel=1e-4)


def test_glum_matches_statsmodels_on_the_same_design():
    sm = pytest.importorskip("statsmodels.formula.api")
    import statsmodels.api as sm_api

    rng = np.random.default_rng(2)
    n = 4_000
    frame = pd.DataFrame({
        "zone": pd.Categorical(rng.choice(["mid", "low", "high"], n), categories=["mid", "low", "high"]),
        "x": rng.normal(size=n),
    })
    exposure = rng.uniform(0.1, 1.0, n)
    rate = np.exp(-1.5 + 0.4 * (frame["zone"] == "high") - 0.3 * (frame["zone"] == "low") + 0.2 * frame["x"])
    y = rng.poisson(rate * exposure).astype(float)
    ours = fit_component(frame, y, power=1.0, factors=("zone",), numeric=("x",), alpha=0.0,
                         offset=np.log(exposure)).coefficients()
    ref = sm.glm("y ~ C(zone) + x", data=frame.assign(y=y), family=sm_api.families.Poisson(),
                 offset=np.log(exposure)).fit()
    assert ours["C(zone)[high]"] == pytest.approx(ref.params["C(zone)[T.high]"], abs=1e-4)
    assert ours["C(zone)[low]"] == pytest.approx(ref.params["C(zone)[T.low]"], abs=1e-4)
    assert ours["x"] == pytest.approx(ref.params["x"], abs=1e-4)


def test_contributions_add_up_to_the_linear_predictor(big):
    table, frame = big
    counts = table["n_claims"].to_numpy(float)
    comp = fit_component(frame, counts, power=1.0, factors=cfg.GLM_FACTORS, numeric=cfg.GLM_NUMERIC,
                         interactions=[("drv_age", "veh_gas")], alpha=1e-4,
                         offset=np.log(table["exposure"].to_numpy()))
    sample = frame.iloc[:500]
    contrib = comp.contributions(sample)
    assert set(contrib.columns) == {*cfg.GLM_FACTORS, *cfg.GLM_NUMERIC}
    lp = contrib.sum(axis=1).to_numpy() + comp.model.intercept_
    assert np.allclose(lp, np.log(comp.predict(sample)), atol=1e-9)


def test_relativities_have_intervals_around_them(big):
    table, frame = big
    counts = table["n_claims"].to_numpy(float)
    offset = np.log(table["exposure"].to_numpy())
    comp = fit_component(frame, counts, power=1.0, factors=cfg.GLM_FACTORS, numeric=cfg.GLM_NUMERIC,
                         alpha=0.0, offset=offset)
    rel = relativities(comp, frame, counts, offset=offset)
    assert set(rel["factor"]) == set(cfg.GLM_FACTORS)
    assert "41-50" not in set(rel.loc[rel["factor"] == "drv_age", "level"])  # base level is implied
    assert (rel["ci_low"] < rel["relativity"]).all() and (rel["relativity"] < rel["ci_high"]).all()


def test_alpha_is_picked_from_the_grid(big):
    table, frame = big
    counts = table["n_claims"].to_numpy(float)
    off = np.log(table["exposure"].to_numpy())
    tr, va = np.arange(0, 40_000), np.arange(40_000, 60_000)
    alpha, scores = select_alpha(frame.iloc[tr], frame.iloc[va], power=1.0, factors=cfg.GLM_FACTORS,
                                 numeric=cfg.GLM_NUMERIC, alphas=(1e-5, 1e-2), y_train=counts[tr],
                                 y_valid=counts[va], off_train=off[tr], off_valid=off[va])
    assert alpha in (1e-5, 1e-2)
    assert len(scores) == 2 and min(s["deviance"] for s in scores) == next(
        s["deviance"] for s in scores if s["alpha"] == alpha)


def test_interaction_search_finds_a_planted_interaction():
    rng = np.random.default_rng(9)
    n = 40_000
    frame = pd.DataFrame({
        "age": pd.Categorical(rng.choice(["mid", "young", "old"], n), categories=["mid", "young", "old"]),
        "fuel": pd.Categorical(rng.choice(["petrol", "diesel"], n), categories=["petrol", "diesel"]),
        "power": pd.Categorical(rng.choice(["low", "high"], n), categories=["low", "high"]),
        "z": rng.normal(size=n),
    })
    exposure = rng.uniform(0.2, 1.0, n)
    young_diesel = (frame["age"] == "young") & (frame["fuel"] == "diesel")
    rate = np.exp(-2.0 + 0.2 * (frame["power"] == "high") + 0.9 * young_diesel)
    y = rng.poisson(rate * exposure).astype(float)
    off = np.log(exposure)
    tr, va = np.arange(30_000), np.arange(30_000, n)
    chosen, history = search_interactions(
        frame.iloc[tr], frame.iloc[va], y_train=y[tr], y_valid=y[va], off_train=off[tr], off_valid=off[va],
        factors=("age", "fuel", "power"), numeric=("z",), candidates=("age", "fuel", "power"),
        alpha=1e-6, max_terms=2, min_gain=1e-4,
    )
    assert chosen[0] == ("age", "fuel")
    assert len(history) >= 3
