"""Gradient-boosted challengers, with exposure handled the same way the GLM does.

Frequency boosters get ``log(exposure)`` as an init score, which LightGBM adds
to the raw prediction and never fits, i.e. a true offset. Severity is a gamma
booster weighted by claim count, and the direct pure-premium booster is a
Tweedie on loss per policy-year weighted by exposure. Bonus-malus is
constrained to push prices one way only: a worse claims record never makes a
quote cheaper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ratecraft.metrics import tweedie_deviance

MONOTONE_INCREASING = ("BonusMalus",)


@dataclass
class GBMComponent:
    booster: lgb.Booster
    objective: str
    params: dict
    best_iteration: int
    #: Multiplier that makes training predictions balance with training
    #: actuals; trees are not guaranteed to, unlike a canonical-link GLM.
    scale: float = 1.0
    search: list[dict] = field(default_factory=list)

    def raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(frame, raw_score=True, num_iteration=self.best_iteration)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Annual value per row (the init score is not part of raw_score)."""
        return self.scale * np.exp(self.raw(frame))

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions to the log prediction, one column per feature.

        They are measured against the booster's average raw score on its
        training data, so they already read as "versus an average policy"."""
        contrib = self.booster.predict(frame, pred_contrib=True, num_iteration=self.best_iteration)
        return pd.DataFrame(contrib[:, :-1], columns=list(frame.columns), index=frame.index)


def _monotone(frame: pd.DataFrame) -> list[int]:
    return [1 if c in MONOTONE_INCREASING else 0 for c in frame.columns]


def fit_component(
    objective: str,
    train: pd.DataFrame,
    y_train,
    valid: pd.DataFrame,
    y_valid,
    *,
    grid,
    base_params: dict,
    seed: int,
    max_rounds: int,
    early_stopping: int,
    w_train=None,
    w_valid=None,
    init_train=None,
    init_valid=None,
    extra_params: dict | None = None,
) -> GBMComponent:
    """Grid over tree shapes, early stopping on validation, keep the best."""
    power = {"poisson": 1.0, "gamma": 2.0}.get(objective, (extra_params or {}).get("tweedie_variance_power", 1.5))
    dtrain = lgb.Dataset(train, np.asarray(y_train, float), weight=w_train, init_score=init_train,
                         free_raw_data=False)
    dvalid = lgb.Dataset(valid, np.asarray(y_valid, float), weight=w_valid, init_score=init_valid,
                         reference=dtrain, free_raw_data=False)
    best: GBMComponent | None = None
    best_score = np.inf
    search = []
    for shape in grid:
        params = {
            **base_params, **(extra_params or {}), **shape,
            "objective": objective, "seed": seed, "monotone_constraints": _monotone(train),
        }
        booster = lgb.train(
            params, dtrain, num_boost_round=max_rounds, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        comp = GBMComponent(booster, objective, params, booster.best_iteration)
        pred = comp.predict(valid)
        if init_valid is not None:
            score = tweedie_deviance(y_valid, pred * np.exp(init_valid), None, power)
        else:
            score = tweedie_deviance(y_valid, pred, w_valid, power)
        search.append({**shape, "rounds": int(booster.best_iteration), "valid_deviance": score})
        if score < best_score:
            best, best_score = comp, score
    assert best is not None
    best.search = search
    # Balance on the training fold.
    pred_train = best.predict(train)
    if init_train is not None:
        expected = pred_train * np.exp(init_train)
        best.scale = float(np.sum(y_train) / np.sum(expected))
    else:
        w = np.ones(len(train)) if w_train is None else np.asarray(w_train, float)
        best.scale = float(np.sum(np.asarray(y_train, float) * w) / np.sum(pred_train * w))
    return best
