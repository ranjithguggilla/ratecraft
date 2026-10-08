"""A synthetic motor book with the same schema as the real one.

The tests use it so they never need the network or the GPL data, and the
pipeline smoke test runs end to end on it in a few seconds. The true
frequency structure is log-linear and known, which lets the tests check that
a fitted GLM recovers it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BRANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14")
REGIONS = (
    "Centre", "Rhone-Alpes", "Provence-Alpes-Cotes-D'Azur", "Ile-de-France",
    "Bretagne", "Aquitaine", "Pays-de-la-Loire", "Nord-Pas-de-Calais",
)

#: True log-rate effects used to simulate claim counts.
TRUE_EFFECTS = {
    "intercept": np.log(0.07),
    "young_driver": 0.55,  # DrivAge < 26
    "log_bm_slope": 1.6,  # coefficient on log(BonusMalus / 50)
    "diesel": 0.10,
    "log_density_slope": 0.06,
}


def make_portfolio(n: int = 20_000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (policies, claims) in the normalised layout of ``ratecraft.data``."""
    rng = np.random.default_rng(seed)
    drv_age = np.clip(rng.normal(46, 15, n).round(), 18, 95).astype(np.int64)
    veh_age = np.clip(rng.gamma(2.0, 3.5, n).round(), 0, 40).astype(np.int64)
    veh_power = rng.integers(4, 14, n)
    bonus_malus = np.where(rng.random(n) < 0.55, 50, rng.integers(51, 160, n)).astype(np.int64)
    density = np.clip(np.exp(rng.normal(6.0, 1.9, n)), 1, 27_000).round().astype(np.int64)
    area = pd.cut(density, [0, 50, 100, 500, 2000, 10_000, np.inf], labels=list("ABCDEF")).astype(str)
    gas = np.where(rng.random(n) < 0.49, "Diesel", "Regular")
    brand = rng.choice(BRANDS, n)
    region = rng.choice(REGIONS, n)
    exposure = np.where(rng.random(n) < 0.35, 1.0, rng.uniform(0.01, 1.0, n))
    # The real table has a few rows running past one policy-year; keep some.
    over = rng.random(n) < 0.002
    exposure[over] = rng.uniform(1.0, 2.0, over.sum())

    log_rate = (
        TRUE_EFFECTS["intercept"]
        + TRUE_EFFECTS["young_driver"] * (drv_age < 26)
        + TRUE_EFFECTS["log_bm_slope"] * np.log(bonus_malus / 50)
        + TRUE_EFFECTS["diesel"] * (gas == "Diesel")
        + TRUE_EFFECTS["log_density_slope"] * np.log(density)
    )
    counts = rng.poisson(np.exp(log_rate) * np.minimum(exposure, 1.0))

    policies = pd.DataFrame(
        {
            "IDpol": np.arange(1, n + 1, dtype=np.int64),
            "ClaimNb": counts.astype(np.int64),
            "Exposure": exposure,
            "VehPower": veh_power.astype(np.int64),
            "VehAge": veh_age,
            "DrivAge": drv_age,
            "BonusMalus": bonus_malus,
            "VehBrand": pd.Categorical(brand),
            "VehGas": pd.Categorical(gas),
            "Area": pd.Categorical(area),
            "Density": density,
            "Region": pd.Categorical(region),
        }
    )

    owners = np.repeat(policies["IDpol"].to_numpy(), counts)
    power_for_claim = np.repeat(veh_power, counts)
    # Attritional claims around 1,500 that grow a little with engine power,
    # plus a thin layer of large losses.
    amounts = rng.gamma(1.4, 1100.0 * (1 + 0.04 * (power_for_claim - 6)), owners.size)
    large = rng.random(owners.size) < 0.004
    amounts[large] = 50_000 * (1 + rng.pareto(1.5, large.sum()))
    claims = pd.DataFrame({"IDpol": owners, "ClaimAmount": np.round(amounts, 2)})
    return policies, claims
