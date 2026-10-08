from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ratecraft.config import BANDS, CATEGORICAL_BASE, GBM_CATEGORICAL, GBM_NUMERIC, Band
from ratecraft.features import band


@pytest.mark.parametrize(
    ("value", "expected"),
    [(18, "18-20"), (20, "18-20"), (21, "21-25"), (40, "31-40"), (41, "41-50"), (80, "71-80"), (81, "81+"),
     (100, "81+")],
)
def test_driver_age_band_edges(value, expected):
    assert band([value], BANDS["drv_age"])[0] == expected


@pytest.mark.parametrize(
    ("factor", "value", "expected"),
    [("bonus_malus", 50, "50"), ("bonus_malus", 51, "51-57"), ("bonus_malus", 100, "100-125"),
     ("bonus_malus", 230, "126+"), ("veh_age", 0, "0"), ("veh_age", 4, "2-4"), ("veh_age", 100, "20+"),
     ("veh_power", 12, "12+"), ("veh_power", 15, "12+"), ("density_band", 50, "1-50"),
     ("density_band", 51, "51-100"), ("density_band", 27_000, "10001+")],
)
def test_other_band_edges(factor, value, expected):
    assert band([value], BANDS[factor])[0] == expected


def test_base_level_comes_first_so_treatment_coding_drops_it():
    for spec in BANDS.values():
        cats = band([spec.edges[0]], spec).categories
        assert cats[0] == spec.base
        assert sorted(cats) == sorted(spec.labels)


def test_values_below_the_first_edge_are_an_error():
    with pytest.raises(ValueError, match="below the first band edge"):
        band([17], BANDS["drv_age"])


def test_band_definition_is_checked():
    with pytest.raises(ValueError, match="one label per left edge"):
        Band("x", (0, 1), ("a",), base="a")
    with pytest.raises(ValueError, match="not a label"):
        Band("x", (0, 1), ("a", "b"), base="c")


def test_spec_puts_each_base_level_first(spec):
    for name, (_, base) in CATEGORICAL_BASE.items():
        assert spec.levels[name][0] == base


def test_unknown_category_is_rejected(spec, table):
    one = table.head(1).copy()
    one["Region"] = pd.Categorical(["Atlantis"])
    with pytest.raises(ValueError, match="unknown region"):
        spec.glm_frame(one)


def test_glm_frame_is_complete_and_uses_fixed_levels(spec, table):
    frame = spec.glm_frame(table)
    assert not frame.isna().any().any()
    assert np.allclose(frame["log_density"], np.log(table["Density"]))
    subset = spec.glm_frame(table.head(5))
    for col in ("region", "veh_brand", "drv_age"):
        assert list(subset[col].cat.categories) == list(frame[col].cat.categories)


def test_gbm_frame_columns(spec, table):
    frame = spec.gbm_frame(table)
    assert list(frame.columns) == [*GBM_NUMERIC, *GBM_CATEGORICAL]
    for col in GBM_CATEGORICAL:
        assert isinstance(frame[col].dtype, pd.CategoricalDtype)
    assert frame["BonusMalus"].dtype == float
