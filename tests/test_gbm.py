from __future__ import annotations

import numpy as np
import pytest

from ratecraft import config as C
from ratecraft.gbm import fit_component

SMALL = {"grid": ({"num_leaves": 7, "min_data_in_leaf": 50},), "max_rounds": 80, "early_stopping": 15}
PARAMS = {**C.GBM_BASE_PARAMS, "num_threads": 1}


@pytest.fixture(scope="module")
def frequency_booster(table, spec):
    frame = spec.gbm_frame(table)
    counts = table["n_claims"].to_numpy(float)
    off = np.log(table["exposure"].to_numpy())
    tr, va = np.arange(0, 6_000), np.arange(6_000, len(table))
    comp = fit_component("poisson", frame.iloc[tr], counts[tr], frame.iloc[va], counts[va], base_params=PARAMS,
                         seed=1, init_train=off[tr], init_valid=off[va], **SMALL)
    return comp, frame, counts, off, tr


def test_booster_balances_on_its_training_fold(frequency_booster):
    comp, frame, counts, off, tr = frequency_booster
    expected = comp.predict(frame.iloc[tr]) * np.exp(off[tr])
    assert expected.sum() == pytest.approx(counts[tr].sum(), rel=1e-9)
    assert comp.search and comp.best_iteration >= 1


def test_a_worse_bonus_malus_never_makes_a_quote_cheaper(frequency_booster):
    comp, frame, *_ = frequency_booster
    row = frame.iloc[[0]].copy()
    rates = []
    for bm in range(50, 231, 10):
        row["BonusMalus"] = float(bm)
        rates.append(comp.predict(row)[0])
    assert np.all(np.diff(rates) >= -1e-12)


def test_tree_contributions_add_up_to_the_raw_score(frequency_booster):
    comp, frame, *_ = frequency_booster
    sample = frame.iloc[:200]
    full = comp.booster.predict(sample, pred_contrib=True, num_iteration=comp.best_iteration)
    contrib = comp.contributions(sample)
    assert list(contrib.columns) == list(sample.columns)
    assert np.allclose(contrib.sum(axis=1).to_numpy() + full[:, -1], comp.raw(sample))


def test_severity_booster_uses_claim_count_weights(table, spec):
    frame = spec.gbm_frame(table)
    counts = table["n_claims"].to_numpy(float)
    idx = np.flatnonzero(counts > 0)
    y = table["loss_capped"].to_numpy()[idx] / counts[idx]
    half = idx.size // 2
    comp = fit_component("gamma", frame.iloc[idx[:half]], y[:half], frame.iloc[idx[half:]], y[half:],
                         base_params=PARAMS, seed=1, w_train=counts[idx[:half]], w_valid=counts[idx[half:]], **SMALL)
    pred = comp.predict(frame.iloc[idx[:half]])
    assert np.sum(pred * counts[idx[:half]]) == pytest.approx(np.sum(y[:half] * counts[idx[:half]]), rel=1e-9)
