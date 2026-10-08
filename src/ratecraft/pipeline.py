"""The whole study, stage by stage.

Each stage caches what it fits under ``data/interim`` so a stage can be rerun
on its own; ``run_all`` runs them in order. Model choices are made on the
validation fold only. The test fold is touched once, in ``evaluate``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ratecraft import config as C
from ratecraft import gbm as G
from ratecraft import glm as L
from ratecraft.audit import audit
from ratecraft.data import attach_losses
from ratecraft.features import FeatureSpec
from ratecraft.largeloss import LargeLossLoading
from ratecraft.metrics import poisson_deviance, tweedie_deviance
from ratecraft.split import Split, grouped_split, profile_groups, random_split

STAGES = ("frequency_glm", "frequency_gbm", "severity", "direct", "studies")


@dataclass
class Context:
    policies: pd.DataFrame
    claims: pd.DataFrame
    spec: FeatureSpec
    glm: pd.DataFrame
    gbm: pd.DataFrame
    groups: np.ndarray
    split: Split
    loading: LargeLossLoading
    audit: dict
    interim: Path
    seed: int
    quick: bool = False

    # --- targets -----------------------------------------------------------
    @property
    def counts(self) -> np.ndarray:
        return self.policies["n_claims"].to_numpy(dtype=float)

    @property
    def exposure(self) -> np.ndarray:
        return self.policies["exposure"].to_numpy(dtype=float)

    @property
    def offset(self) -> np.ndarray:
        return np.log(self.exposure)

    @property
    def pure_premium(self) -> np.ndarray:
        """Attritional (capped) claims cost per policy-year."""
        return self.policies["loss_capped"].to_numpy() / self.exposure

    def claimants(self, idx: np.ndarray) -> np.ndarray:
        return idx[self.counts[idx] > 0]

    def severity(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean capped claim size and claim count, for policies with claims."""
        n = self.counts[idx]
        return self.policies["loss_capped"].to_numpy()[idx] / n, n

    def cache(self, name: str) -> Path:
        return self.interim / f"stage_{name}.joblib"

    def load_stage(self, name: str) -> dict:
        path = self.cache(name)
        if not path.exists():
            raise FileNotFoundError(f"stage {name!r} has not been run yet ({path})")
        return joblib.load(path)


def prepare(policies: pd.DataFrame, claims: pd.DataFrame, interim: Path, *, seed: int = C.SEED,
            threshold: float | None = C.LARGE_LOSS_THRESHOLD, quick: bool = False) -> Context:
    report = audit(policies, claims, exposure_cap=C.EXPOSURE_CAP, threshold=threshold or np.inf)
    table = attach_losses(policies, claims, threshold, C.EXPOSURE_CAP)
    spec = FeatureSpec.from_policies(table)
    groups = profile_groups(table)
    split = grouped_split(groups, C.SPLIT_FRACTIONS, seed)
    train_ids = set(table["IDpol"].to_numpy()[split.train])
    train_amounts = claims.loc[claims["IDpol"].isin(train_ids), "ClaimAmount"].to_numpy()
    interim.mkdir(parents=True, exist_ok=True)
    return Context(
        policies=table,
        claims=claims,
        spec=spec,
        glm=spec.glm_frame(table),
        gbm=spec.gbm_frame(table),
        groups=groups,
        split=split,
        loading=LargeLossLoading.fit(train_amounts, threshold),
        audit=report,
        interim=interim,
        seed=seed,
        quick=quick,
    )


def _gbm_settings(ctx: Context) -> dict:
    if ctx.quick:
        return {"grid": C.GBM_GRID[:1], "max_rounds": 60, "early_stopping": 10}
    return {"grid": C.GBM_GRID, "max_rounds": C.GBM_MAX_ROUNDS, "early_stopping": C.GBM_EARLY_STOPPING}


# --- stages ------------------------------------------------------------------

def stage_frequency_glm(ctx: Context) -> dict:
    tr, va = ctx.split.train, ctx.split.valid
    y, off, e = ctx.counts, ctx.offset, ctx.exposure
    alpha, alpha_scores = L.select_alpha(
        ctx.glm.iloc[tr], ctx.glm.iloc[va], power=1.0, factors=C.GLM_FACTORS, numeric=C.GLM_NUMERIC,
        alphas=C.GLM_ALPHAS, y_train=y[tr], y_valid=y[va], off_train=off[tr], off_valid=off[va],
    )
    main = L.fit_component(ctx.glm.iloc[tr], y[tr], power=1.0, factors=C.GLM_FACTORS,
                           numeric=C.GLM_NUMERIC, alpha=alpha, offset=off[tr])
    flat_rate = float(y[tr].sum() / e[tr].sum())
    flat_dev = poisson_deviance(y[va], flat_rate * e[va])
    main_dev = poisson_deviance(y[va], main.predict(ctx.glm.iloc[va]) * e[va])
    min_gain = C.INTERACTION_MIN_GAIN * (flat_dev - main_dev)
    candidates = C.INTERACTION_CANDIDATES[:3] if ctx.quick else C.INTERACTION_CANDIDATES
    chosen, history = L.search_interactions(
        ctx.glm.iloc[tr], ctx.glm.iloc[va], y_train=y[tr], y_valid=y[va], off_train=off[tr],
        off_valid=off[va], factors=C.GLM_FACTORS, numeric=C.GLM_NUMERIC, candidates=candidates,
        alpha=alpha, max_terms=1 if ctx.quick else C.INTERACTION_MAX_TERMS, min_gain=min_gain,
    )
    hybrid = L.fit_component(ctx.glm.iloc[tr], y[tr], power=1.0, factors=C.GLM_FACTORS,
                             numeric=C.GLM_NUMERIC, interactions=chosen, alpha=alpha, offset=off[tr])
    return {
        "alpha": alpha, "alpha_scores": alpha_scores, "flat_rate": flat_rate,
        "main": main, "hybrid": hybrid, "interactions": chosen, "history": history,
        "valid": {"flat": flat_dev, "main": main_dev, "min_gain": min_gain},
    }


def stage_frequency_gbm(ctx: Context) -> dict:
    tr, va = ctx.split.train, ctx.split.valid
    y, off = ctx.counts, ctx.offset
    comp = G.fit_component(
        "poisson", ctx.gbm.iloc[tr], y[tr], ctx.gbm.iloc[va], y[va], base_params=C.GBM_BASE_PARAMS,
        seed=ctx.seed, init_train=off[tr], init_valid=off[va], **_gbm_settings(ctx),
    )
    return {"gbm": comp}


def stage_severity(ctx: Context) -> dict:
    tr, va = ctx.claimants(ctx.split.train), ctx.claimants(ctx.split.valid)
    y_tr, n_tr = ctx.severity(tr)
    y_va, n_va = ctx.severity(va)
    alpha, alpha_scores = L.select_alpha(
        ctx.glm.iloc[tr], ctx.glm.iloc[va], power=2.0, factors=C.GLM_FACTORS, numeric=C.GLM_NUMERIC,
        alphas=C.SEVERITY_ALPHAS, y_train=y_tr, y_valid=y_va, w_train=n_tr, w_valid=n_va,
    )
    glm = L.fit_component(ctx.glm.iloc[tr], y_tr, power=2.0, factors=C.GLM_FACTORS,
                          numeric=C.GLM_NUMERIC, alpha=alpha, weights=n_tr)
    gbm = G.fit_component(
        "gamma", ctx.gbm.iloc[tr], y_tr, ctx.gbm.iloc[va], y_va, base_params=C.GBM_BASE_PARAMS,
        seed=ctx.seed, w_train=n_tr, w_valid=n_va, **_gbm_settings(ctx),
    )
    flat = float(np.sum(y_tr * n_tr) / np.sum(n_tr))
    return {
        "alpha": alpha, "alpha_scores": alpha_scores, "glm": glm, "gbm": gbm, "flat": flat,
        "valid": {
            "flat": tweedie_deviance(y_va, np.full_like(y_va, flat), n_va, 2.0),
            "glm": tweedie_deviance(y_va, glm.predict(ctx.glm.iloc[va]), n_va, 2.0),
            "gbm": tweedie_deviance(y_va, gbm.predict(ctx.gbm.iloc[va]), n_va, 2.0),
        },
    }


def stage_direct(ctx: Context) -> dict:
    tr, va = ctx.split.train, ctx.split.valid
    y, e = ctx.pure_premium, ctx.exposure
    alpha = ctx.load_stage("frequency_glm")["alpha"]
    power_scores = []
    best = None
    for power in C.TWEEDIE_POWERS[2:3] if ctx.quick else C.TWEEDIE_POWERS:
        comp = L.fit_component(ctx.glm.iloc[tr], y[tr], power=power, factors=C.GLM_FACTORS,
                               numeric=C.GLM_NUMERIC, alpha=alpha, weights=e[tr])
        dev = tweedie_deviance(y[va], comp.predict(ctx.glm.iloc[va]), e[va], C.EVAL_POWER)
        power_scores.append({"power": power, "valid_deviance": dev})
        if best is None or dev < best[0]:
            best = (dev, power, comp)
    _, power, glm = best
    gbm = G.fit_component(
        "tweedie", ctx.gbm.iloc[tr], y[tr], ctx.gbm.iloc[va], y[va], base_params=C.GBM_BASE_PARAMS,
        seed=ctx.seed, w_train=e[tr], w_valid=e[va],
        extra_params={"tweedie_variance_power": power}, **_gbm_settings(ctx),
    )
    return {"power": power, "power_scores": power_scores, "glm": glm, "gbm": gbm}


def _lift(y, mu, e_flat_rate, e) -> float:
    flat = poisson_deviance(y, e_flat_rate * e)
    return (flat - poisson_deviance(y, mu)) / flat


def leakage_study(ctx: Context, alpha: float, gbm_params: dict, rounds: int, seeds) -> pd.DataFrame:
    """Same models, same data, grouped versus row-level splits, several seeds.

    The boosters use the tree shape and round count chosen on the main
    validation fold, with no early stopping, so the only thing that changes
    between the two arms is how rows were dealt into folds."""
    import lightgbm as lgb

    y, e, off = ctx.counts, ctx.exposure, ctx.offset
    n = len(y)
    rows = []
    for seed in seeds:
        for kind in ("grouped", "random"):
            sp = grouped_split(ctx.groups, C.SPLIT_FRACTIONS, seed) if kind == "grouped" else random_split(
                n, C.SPLIT_FRACTIONS, seed)
            tr = np.concatenate([sp.train, sp.valid])
            te = sp.test
            flat_rate = y[tr].sum() / e[tr].sum()
            glm = L.fit_component(ctx.glm.iloc[tr], y[tr], power=1.0, factors=C.GLM_FACTORS,
                                  numeric=C.GLM_NUMERIC, alpha=alpha, offset=off[tr])
            mu_glm = glm.predict(ctx.glm.iloc[te]) * e[te]
            booster = lgb.train(gbm_params, lgb.Dataset(ctx.gbm.iloc[tr], y[tr], init_score=off[tr]),
                                num_boost_round=rounds)
            mu_gbm = np.exp(booster.predict(ctx.gbm.iloc[te], raw_score=True) + off[te])
            seen = np.isin(ctx.groups[te], np.unique(ctx.groups[tr]))
            rows.append({
                "seed": seed, "split": kind,
                "glm_improvement": _lift(y[te], mu_glm, flat_rate, e[te]),
                "gbm_improvement": _lift(y[te], mu_gbm, flat_rate, e[te]),
                "test_rows_with_profile_in_train": float(seen.mean()),
            })
    return pd.DataFrame(rows)


EXPOSURE_BANDS = (0.0, 1 / 12, 0.25, 0.5, 0.75, 0.999)
EXPOSURE_LABELS = ("under 1 month", "1-3 months", "3-6 months", "6-9 months", "9-12 months", "full year")


def exposure_band(e: np.ndarray) -> pd.Categorical:
    idx = np.searchsorted(np.asarray(EXPOSURE_BANDS), e, side="left") - 1
    idx = np.clip(idx, 0, len(EXPOSURE_LABELS) - 1)
    labels = np.asarray(EXPOSURE_LABELS, dtype=object)[idx]
    return pd.Categorical(labels, categories=[EXPOSURE_LABELS[-1], *EXPOSURE_LABELS[:-1]])


def exposure_study(ctx: Context, alpha: float) -> dict:
    """Three ways of treating exposure in the same frequency GLM."""
    tr, te = ctx.split.train, ctx.split.test
    y, e, off = ctx.counts, ctx.exposure, ctx.offset
    frame = ctx.glm.copy()
    frame["duration"] = exposure_band(e)

    offset_glm = L.fit_component(frame.iloc[tr], y[tr], power=1.0, factors=C.GLM_FACTORS,
                                 numeric=C.GLM_NUMERIC, alpha=alpha, offset=off[tr])
    # The shortcut: annualise each policy's claims and fit that, unweighted.
    ratio_glm = L.fit_component(frame.iloc[tr], y[tr] / e[tr], power=1.0, factors=C.GLM_FACTORS,
                                numeric=C.GLM_NUMERIC, alpha=alpha)
    duration_glm = L.fit_component(frame.iloc[tr], y[tr], power=1.0, factors=(*C.GLM_FACTORS, "duration"),
                                   numeric=C.GLM_NUMERIC, alpha=alpha, offset=off[tr])

    test = frame.iloc[te]
    full_year = test.copy()
    full_year["duration"] = pd.Categorical([EXPOSURE_LABELS[-1]] * len(test),
                                           categories=frame["duration"].cat.categories)
    expected = {
        "offset": offset_glm.predict(test) * e[te],
        "annualised, unweighted": ratio_glm.predict(test) * e[te],
        "offset + duration factor": duration_glm.predict(test) * e[te],
    }
    annual_full_year = {
        "offset": float(np.mean(offset_glm.predict(test))),
        "annualised, unweighted": float(np.mean(ratio_glm.predict(test))),
        "offset + duration factor": float(np.mean(duration_glm.predict(full_year))),
    }
    bands = frame["duration"].iloc[te].astype(str).to_numpy()
    rows = []
    for name, mu in expected.items():
        for label in EXPOSURE_LABELS:
            m = bands == label
            rows.append({"model": name, "duration": label, "policies": int(m.sum()),
                         "exposure": float(e[te][m].sum()), "actual": float(y[te][m].sum()),
                         "expected": float(mu[m].sum())})
    by_band = pd.DataFrame(rows)
    by_band["a_over_e"] = by_band["actual"] / by_band["expected"]
    raw = (
        pd.DataFrame({"duration": frame["duration"].astype(str), "e": e, "y": y})
        .groupby("duration").agg(policies=("y", "size"), exposure=("e", "sum"), claims=("y", "sum"))
        .reindex(EXPOSURE_LABELS)
    )
    raw["claims_per_year"] = raw["claims"] / raw["exposure"]
    summary = {
        name: {
            "test_deviance": poisson_deviance(y[te], mu),
            "balance": float(mu.sum() / y[te].sum()),
            "mean_full_year_rate": annual_full_year[name],
        }
        for name, mu in expected.items()
    }
    return {"by_band": by_band, "raw": raw.reset_index(), "summary": summary,
            "duration_relativities": L.relativities(duration_glm, frame.iloc[tr], y[tr], offset=off[tr])}


def stage_studies(ctx: Context) -> dict:
    freq = ctx.load_stage("frequency_glm")
    gbm = ctx.load_stage("frequency_gbm")["gbm"]
    params = {k: v for k, v in gbm.params.items() if k != "metric"}
    seeds = (1,) if ctx.quick else (1, 2, 3, 4, 5)
    leakage = leakage_study(ctx, freq["alpha"], params, max(gbm.best_iteration, 1), seeds)
    return {"leakage": leakage, "exposure": exposure_study(ctx, freq["alpha"])}


RUNNERS = {
    "frequency_glm": stage_frequency_glm,
    "frequency_gbm": stage_frequency_gbm,
    "severity": stage_severity,
    "direct": stage_direct,
    "studies": stage_studies,
}


def run_stage(ctx: Context, name: str, log=print) -> dict:
    started = time.time()
    result = RUNNERS[name](ctx)
    result["seconds"] = round(time.time() - started, 1)
    joblib.dump(result, ctx.cache(name))
    log(f"  {name}: {result['seconds']}s")
    return result
