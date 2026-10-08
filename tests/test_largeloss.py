from __future__ import annotations

import numpy as np
import pytest

from ratecraft.largeloss import LargeLossLoading


def test_loading_is_excess_over_capped_cost():
    amounts = np.array([1_000.0, 2_000.0, 80_000.0])
    layer = LargeLossLoading.fit(amounts, 50_000.0)
    # capped total 53,000, excess 30,000
    assert layer.loading == pytest.approx(30_000 / 53_000)
    assert layer.cap(amounts).tolist() == [1_000.0, 2_000.0, 50_000.0]
    # charging the loading back on attritional cost recovers the full total
    assert layer.apply(layer.cap(amounts).sum()) == pytest.approx(amounts.sum())


def test_no_threshold_is_a_no_op():
    amounts = np.array([1.0, 1e6])
    layer = LargeLossLoading.fit(amounts, None)
    assert layer.loading == 0.0
    assert np.array_equal(layer.cap(amounts), amounts)


def test_threshold_above_every_claim_gives_zero_loading():
    assert LargeLossLoading.fit([10.0, 20.0], 1e9).loading == 0.0
