"""Shared fixtures. Everything here is synthetic: no test needs the network or
the GPL-licensed source data."""

from __future__ import annotations

import numpy as np
import pytest

from ratecraft import config as C
from ratecraft.data import attach_losses
from ratecraft.features import FeatureSpec
from ratecraft.glm import fit_component
from ratecraft.largeloss import LargeLossLoading
from ratecraft.pricing import Pricer, PurePremiumModel
from ratecraft.synth import make_portfolio


@pytest.fixture(scope="session")
def portfolio():
    return make_portfolio(8_000, seed=3)


@pytest.fixture(scope="session")
def table(portfolio):
    policies, claims = portfolio
    return attach_losses(policies, claims, 50_000.0, C.EXPOSURE_CAP)


@pytest.fixture(scope="session")
def spec(table):
    return FeatureSpec.from_policies(table)


@pytest.fixture(scope="session")
def glm_frame(spec, table):
    return spec.glm_frame(table)


@pytest.fixture(scope="session")
def pricer(table, spec, glm_frame, portfolio):
    """A small but real GLM pricer fitted on the synthetic book."""
    counts = table["n_claims"].to_numpy(dtype=float)
    exposure = table["exposure"].to_numpy()
    freq = fit_component(glm_frame, counts, power=1.0, factors=C.GLM_FACTORS, numeric=C.GLM_NUMERIC,
                         alpha=1e-4, offset=np.log(exposure))
    has = counts > 0
    sev_y = table["loss_capped"].to_numpy()[has] / counts[has]
    sev = fit_component(glm_frame[has], sev_y, power=2.0, factors=C.GLM_FACTORS, numeric=C.GLM_NUMERIC,
                        alpha=0.1, weights=counts[has])
    for comp in (freq, sev):
        comp.set_reference(glm_frame, exposure)
    model = PurePremiumModel("GLM tariff", "freq_sev", frequency=freq, severity=sev)
    loading = LargeLossLoading.fit(portfolio[1]["ClaimAmount"].to_numpy(), 50_000.0)
    return Pricer(model=model, spec=spec, loading=loading, metadata={"shipped": "GLM tariff"})


@pytest.fixture
def profile():
    return {
        "DrivAge": 22, "VehAge": 3, "VehPower": 8, "BonusMalus": 90, "VehBrand": "B12",
        "VehGas": "Diesel", "Density": 3_000, "Region": "Ile-de-France",
    }
