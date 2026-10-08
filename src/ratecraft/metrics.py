"""Scores a pricing actuary would ask for, beyond the ones sklearn ships.

Deviance answers "how close", the ordered Lorenz curve and Gini answer "does
it rank risks", actual-over-expected by band and the double-lift chart answer
"where is it wrong", and the Murphy decomposition splits a score into the
part a recalibration could fix and the part it could not.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_tweedie_deviance


def tweedie_deviance(y, mu, weights=None, power: float = 1.5) -> float:
    """Mean Tweedie unit deviance. power=1 is Poisson, power=2 is gamma."""
    return float(mean_tweedie_deviance(np.asarray(y, float), np.asarray(mu, float), sample_weight=weights, power=power))


def poisson_deviance(counts, expected) -> float:
    """Mean Poisson deviance on claim counts against expected counts."""
    return tweedie_deviance(counts, expected, power=1.0)


def unit_deviance(y, mu, power: float) -> np.ndarray:
    """Per-row Tweedie unit deviance, so a bootstrap can resum rows cheaply.

    Matches sklearn's mean_tweedie_deviance when averaged with the same weights."""
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    if power == 1.0:
        safe = np.where(y > 0, y, 1.0)
        return 2.0 * (np.where(y > 0, y * np.log(safe / mu), 0.0) - y + mu)
    if power == 2.0:
        return 2.0 * (np.log(mu / y) + y / mu - 1.0)
    p = power
    term = np.where(y > 0, np.power(np.where(y > 0, y, 1.0), 2 - p) / ((1 - p) * (2 - p)), 0.0)
    return 2.0 * (term - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))


def improvement(model_score: float, flat_score: float) -> float:
    """Share of the flat rate's deviance a model removes (D-squared style)."""
    return float((flat_score - model_score) / flat_score)


def _tied_groups(predicted: np.ndarray, *columns: np.ndarray) -> tuple[np.ndarray, ...]:
    """Sum columns within groups of identical predictions, sorted ascending.

    Treating ties as one block keeps the Lorenz curve well defined: a model
    that predicts the same value for everyone gets the diagonal and a Gini of
    exactly zero instead of whatever the row order happens to be."""
    uniq, inverse = np.unique(predicted, return_inverse=True)
    return tuple(np.bincount(inverse, weights=c, minlength=uniq.size) for c in columns)


def lorenz_curve(actual, predicted, exposure) -> tuple[np.ndarray, np.ndarray]:
    """Ordered Lorenz curve: policies sorted from cheapest to dearest predicted
    rate; x is the share of exposure, y the share of actual losses."""
    actual = np.asarray(actual, float)
    exposure = np.asarray(exposure, float)
    exp_sum, loss_sum = _tied_groups(np.asarray(predicted, float), exposure, actual)
    x = np.concatenate([[0.0], np.cumsum(exp_sum) / exp_sum.sum()])
    y = np.concatenate([[0.0], np.cumsum(loss_sum) / loss_sum.sum()])
    return x, y


def gini(actual, predicted, exposure) -> float:
    """Gini of the ordered Lorenz curve. Zero for a flat rate, larger is better."""
    x, y = lorenz_curve(actual, predicted, exposure)
    return float(1.0 - 2.0 * np.trapezoid(y, x))


def balance(actual, expected) -> float:
    """Total expected over total actual. 1.0 means the book is priced in total."""
    return float(np.sum(expected) / np.sum(actual))


def _equal_exposure_bands(rank_by: np.ndarray, exposure: np.ndarray, n_bands: int) -> np.ndarray:
    order = np.argsort(rank_by, kind="mergesort")
    cum = np.cumsum(exposure[order]) / exposure.sum()
    band_sorted = np.minimum((cum * n_bands - 1e-12).astype(int), n_bands - 1)
    bands = np.empty_like(band_sorted)
    bands[order] = band_sorted
    return bands


def actual_vs_expected(actual, expected, exposure, rank_by=None, n_bands: int = 10) -> pd.DataFrame:
    """A/E in equal-exposure bands of the model's own predicted rate."""
    actual = np.asarray(actual, float)
    expected = np.asarray(expected, float)
    exposure = np.asarray(exposure, float)
    rank_by = expected / exposure if rank_by is None else np.asarray(rank_by, float)
    bands = _equal_exposure_bands(rank_by, exposure, n_bands)
    frame = pd.DataFrame({"band": bands + 1, "exposure": exposure, "actual": actual, "expected": expected})
    out = frame.groupby("band").sum()
    out["actual_per_year"] = out["actual"] / out["exposure"]
    out["expected_per_year"] = out["expected"] / out["exposure"]
    out["a_over_e"] = out["actual"] / out["expected"]
    return out.reset_index()


def double_lift(actual, pred_a, pred_b, exposure, n_bands: int = 10) -> pd.DataFrame:
    """The classic champion/challenger chart.

    Sort by the ratio of the two models' predictions, band by exposure, and in
    each band compare actual losses with what each model expected. Where the
    models disagree most, the one whose line tracks the actual line wins.
    Every column is expressed relative to its own book-wide average."""
    actual = np.asarray(actual, float)
    pred_a = np.asarray(pred_a, float)
    pred_b = np.asarray(pred_b, float)
    exposure = np.asarray(exposure, float)
    bands = _equal_exposure_bands(pred_a / pred_b, exposure, n_bands)
    frame = pd.DataFrame({"band": bands + 1, "exposure": exposure, "actual": actual, "a": pred_a, "b": pred_b})
    out = frame.groupby("band").sum()
    for col in ("actual", "a", "b"):
        out[f"{col}_relative"] = (out[col] / out["exposure"]) / (frame[col].sum() / frame["exposure"].sum())
    return out.reset_index()


def murphy(y, predicted, weights=None, power: float = 1.5) -> dict[str, float]:
    """Score = miscalibration - discrimination + uncertainty.

    The recalibrated forecast is the isotonic regression of the outcome on the
    prediction. Isotonic blocks with no claims at all have a mean of exactly
    zero, which the Tweedie and Poisson scores do not accept, so they are
    floored at a negligible positive value; the limit of the score there is
    finite, so the floor changes nothing measurable.
    """
    y = np.asarray(y, float)
    predicted = np.asarray(predicted, float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)
    flat = np.average(y, weights=w)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    recalibrated = np.maximum(iso.fit(predicted, y, sample_weight=w).predict(predicted), flat * 1e-9)

    def score(p):
        return tweedie_deviance(y, p, w, power)

    total = score(predicted)
    recal = score(recalibrated)
    uncertainty = score(np.full_like(y, flat))
    return {
        "score": total,
        "miscalibration": total - recal,
        "discrimination": uncertainty - recal,
        "uncertainty": uncertainty,
    }


def bias_by_factor(observed, predicted, feature, weights) -> pd.DataFrame:
    """Mean of (predicted - observed) per level, with a standard error and a
    p-value for "no bias", using model-diagnostics' identification function."""
    from model_diagnostics.calibration import compute_bias

    table = compute_bias(
        y_obs=np.asarray(observed, float),
        y_pred=np.asarray(predicted, float),
        feature=pd.Series(feature).astype(str).to_numpy(),
        weights=np.asarray(weights, float),
    )
    return table.to_pandas()


def group_bootstrap(
    statistic: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Resample whole risk profiles with replacement and recompute a statistic.

    ``statistic`` receives an index array into the evaluation rows."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = np.argsort(inverse, kind="mergesort")
    starts = np.searchsorted(inverse[members], np.arange(uniq.size))
    lengths = np.diff(np.append(starts, members.size))
    out = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, uniq.size, uniq.size)
        lens = lengths[picked]
        # Position of each resampled row inside ``members``: the group's start
        # plus a running offset within the group.
        first = np.repeat(starts[picked], lens)
        within = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
        out[b] = statistic(members[first + within])
    return out
