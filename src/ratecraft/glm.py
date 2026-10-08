"""The tariff: offset Poisson frequency, gamma severity and Tweedie GLMs.

All fitting goes through glum with a one-sided formula, so the model is
written down the way a rating plan is: ``C(drv_age) + C(bonus_malus) + ...``.
Base levels are the first category of each factor (see ``config.BANDS``), and
glum's treatment coding drops exactly those, so ``exp(coef)`` is the
relativity of a level against its base.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution

from ratecraft.metrics import tweedie_deviance

_TERM = re.compile(r"^C\((?P<factor>[a-z_]+)\)\[(?P<level>.*)\]$")


def formula(factors, numeric=(), interactions=()) -> str:
    terms = [f"C({f})" for f in factors] + list(numeric) + [f"C({a}):C({b})" for a, b in interactions]
    return " + ".join(terms)


def family_for(power: float):
    if power == 1.0:
        return "poisson"
    if power == 2.0:
        return "gamma"
    return TweedieDistribution(power)


def factors_of(rhs: str) -> list[str]:
    """Factor and numeric names in a right-hand side, in order of appearance."""
    seen: list[str] = []
    for term in rhs.split("+"):
        for piece in term.strip().split(":"):
            name = piece.strip()
            name = name[2:-1] if name.startswith("C(") else name
            if name and name not in seen:
                seen.append(name)
    return seen


@dataclass
class GLMComponent:
    """One fitted GLM plus what is needed to score and explain new policies."""

    model: GeneralizedLinearRegressor
    rhs: str
    power: float
    alpha: float
    interactions: tuple[tuple[str, str], ...] = ()
    #: Weighted mean contribution of each factor over a reference book, so a
    #: quote can be explained against an average policy rather than against
    #: the base levels nobody actually has all of.
    reference: dict[str, float] = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Annual value (rate, severity or pure premium) per row."""
        return np.asarray(self.model.predict(frame), dtype=float)

    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.model.feature_names_, map(float, self.model.coef_)))

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Each factor's additive share of the linear predictor, per row.

        An interaction term is credited half to each of its two factors, so the
        columns always add up to ``linear predictor - intercept``."""
        main: dict[str, dict[str, float]] = defaultdict(dict)
        pairs: dict[tuple[str, str], dict[tuple[str, str], float]] = defaultdict(dict)
        numeric: dict[str, float] = {}
        for name, value in self.coefficients().items():
            parts = [_TERM.match(p) for p in name.split(":")]
            if len(parts) == 1 and parts[0] is None:
                numeric[name] = value
            elif len(parts) == 1:
                main[parts[0]["factor"]][parts[0]["level"]] = value
            else:
                a, b = parts
                pairs[(a["factor"], b["factor"])][(a["level"], b["level"])] = value

        out = pd.DataFrame(0.0, index=frame.index, columns=factors_of(self.rhs))
        as_text = {c: frame[c].astype(str) for c in out.columns if c not in numeric}
        for factor, table in main.items():
            out[factor] += as_text[factor].map(table).fillna(0.0).to_numpy()
        for name, value in numeric.items():
            out[name] += value * frame[name].to_numpy(dtype=float)
        for (fa, fb), table in pairs.items():
            keyed = {f"{la}\x1f{lb}": v for (la, lb), v in table.items()}
            joint = (as_text[fa] + "\x1f" + as_text[fb]).map(keyed).fillna(0.0).to_numpy()
            out[fa] += joint / 2
            out[fb] += joint / 2
        return out

    def set_reference(self, frame: pd.DataFrame, weights) -> None:
        contrib = self.contributions(frame)
        w = np.asarray(weights, float)
        self.reference = {c: float(np.average(contrib[c], weights=w)) for c in contrib.columns}


def fit_component(
    frame: pd.DataFrame,
    y,
    *,
    power: float,
    factors,
    numeric=(),
    interactions=(),
    alpha: float,
    weights=None,
    offset=None,
) -> GLMComponent:
    rhs = formula(factors, numeric, interactions)
    model = GeneralizedLinearRegressor(
        family=family_for(power), alpha=alpha, formula=rhs, drop_first=True, max_iter=200,
    )
    model.fit(frame, np.asarray(y, dtype=float), sample_weight=weights, offset=offset)
    return GLMComponent(model, rhs, power, alpha, tuple(interactions))


def relativities(component: GLMComponent, frame, y, *, weights=None, offset=None) -> pd.DataFrame:
    """Main-effect relativities against the base level, with 95% intervals."""
    table = component.model.coef_table(X=frame, y=np.asarray(y, float), sample_weight=weights, offset=offset)
    rows = []
    for name, row in table.iterrows():
        m = _TERM.match(str(name))
        if m is None:
            continue
        rows.append(
            {
                "factor": m["factor"],
                "level": m["level"],
                "coef": float(row["coef"]),
                "se": float(row["se"]),
                "relativity": float(np.exp(row["coef"])),
                "ci_low": float(np.exp(row["ci_lower"])),
                "ci_high": float(np.exp(row["ci_upper"])),
                "p_value": float(row["p_value"]),
            }
        )
    return pd.DataFrame(rows)


def deviance(y, annual_pred, *, power, weights=None, offset=None) -> float:
    """Validation score for a GLM's annual prediction. With an offset the
    prediction is scaled up to the policy's own exposure first."""
    annual_pred = np.asarray(annual_pred, float)
    if offset is not None:
        return tweedie_deviance(y, annual_pred * np.exp(offset), None, power)
    return tweedie_deviance(y, annual_pred, weights, power)


def select_alpha(train, valid, *, power, factors, numeric, alphas, y_train, y_valid,
                 w_train=None, w_valid=None, off_train=None, off_valid=None):
    """Pick the ridge strength on the validation fold."""
    scores = []
    for alpha in alphas:
        comp = fit_component(train, y_train, power=power, factors=factors, numeric=numeric,
                             alpha=alpha, weights=w_train, offset=off_train)
        dev = deviance(y_valid, comp.predict(valid), power=power, weights=w_valid, offset=off_valid)
        scores.append({"alpha": float(alpha), "deviance": dev})
    best = min(scores, key=lambda s: s["deviance"])
    return best["alpha"], scores


def search_interactions(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    y_train,
    y_valid,
    off_train,
    off_valid,
    factors,
    numeric,
    candidates,
    alpha: float,
    max_terms: int,
    min_gain: float,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """Greedy forward selection of two-way interactions for the frequency GLM.

    Each round tries every remaining pair of candidate factors, keeps the one
    that lowers validation deviance the most, and stops when the best gain is
    below ``min_gain`` or ``max_terms`` pairs are in."""
    def score(interactions) -> float:
        comp = fit_component(train, y_train, power=1.0, factors=factors, numeric=numeric,
                             interactions=interactions, alpha=alpha, offset=off_train)
        return deviance(y_valid, comp.predict(valid), power=1.0, offset=off_valid)

    chosen: list[tuple[str, str]] = []
    history: list[dict] = []
    current = score([])
    for round_no in range(1, max_terms + 1):
        trials = []
        for pair in combinations(candidates, 2):
            if pair in chosen:
                continue
            dev = score([*chosen, pair])
            trials.append((current - dev, pair, dev))
            history.append({"round": round_no, "pair": f"{pair[0]} x {pair[1]}", "gain": current - dev})
        gain, pair, dev = max(trials, key=lambda t: t[0])
        if gain < min_gain:
            break
        chosen.append(pair)
        current = dev
    return chosen, history
