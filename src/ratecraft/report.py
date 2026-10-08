"""Choose the model on validation, score everything on test once, write it down.

Everything the README quotes comes out of this module, into
``results/metrics.json`` and a set of CSVs, and every chart in ``assets/`` is
drawn from the same numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ratecraft import config as C
from ratecraft import plots
from ratecraft.config import Paths
from ratecraft.data import attach_losses
from ratecraft.glm import fit_component, relativities
from ratecraft.largeloss import LargeLossLoading
from ratecraft.market import simulate, to_frame
from ratecraft.metrics import (
    actual_vs_expected,
    bias_by_factor,
    double_lift,
    gini,
    group_bootstrap,
    murphy,
    poisson_deviance,
    tweedie_deviance,
    unit_deviance,
)
from ratecraft.pipeline import STAGES, Context
from ratecraft.pricing import Frames, Pricer, PurePremiumModel, quote

FLAT = "Flat rate"
GLM_TARIFF = "GLM tariff"
TWEEDIE_GLM = "Tweedie GLM"
HYBRID = "GLM with interactions"
BOOSTED = "Boosted frequency x severity"
TWEEDIE_GBM = "Tweedie GBM"

#: Complexity tiers, simplest first. Within a tier the better validation score
#: wins; moving up a tier has to earn SHIP_MARGIN more improvement.
TIERS = ((GLM_TARIFF, TWEEDIE_GLM), (HYBRID,), (BOOSTED, TWEEDIE_GBM))
GLM_MODELS = (GLM_TARIFF, TWEEDIE_GLM, HYBRID)

BIAS_FACTORS = ("drv_age", "bonus_malus", "veh_age", "veh_power", "veh_brand", "region")

DEMO_PROFILES = {
    "new driver, city, powerful car": {
        "DrivAge": 19, "VehAge": 2, "VehPower": 9, "BonusMalus": 100, "VehBrand": "B12",
        "VehGas": "Regular", "Density": 9_000, "Region": "Ile-de-France",
    },
    "experienced driver, small town": {
        "DrivAge": 47, "VehAge": 6, "VehPower": 6, "BonusMalus": 50, "VehBrand": "B1",
        "VehGas": "Diesel", "Density": 300, "Region": "Bretagne",
    },
    "driver with a claims record": {
        "DrivAge": 35, "VehAge": 10, "VehPower": 7, "BonusMalus": 95, "VehBrand": "B2",
        "VehGas": "Diesel", "Density": 2_500, "Region": "Provence-Alpes-Cotes-D'Azur",
    },
}


def _r(x, nd: int = 6):
    return None if x is None else float(round(float(x), nd))


def assemble(ctx: Context, st: dict) -> dict[str, PurePremiumModel]:
    f, g, s, d = st["frequency_glm"], st["frequency_gbm"], st["severity"], st["direct"]
    tr = ctx.split.train
    flat = float(ctx.policies["loss_capped"].to_numpy()[tr].sum() / ctx.exposure[tr].sum())
    models = {
        FLAT: PurePremiumModel(FLAT, "flat", flat_rate=flat),
        GLM_TARIFF: PurePremiumModel(GLM_TARIFF, "freq_sev", frequency=f["main"], severity=s["glm"]),
        TWEEDIE_GLM: PurePremiumModel(TWEEDIE_GLM, "direct", direct=d["glm"]),
        HYBRID: PurePremiumModel(HYBRID, "freq_sev", frequency=f["hybrid"], severity=s["glm"]),
        BOOSTED: PurePremiumModel(BOOSTED, "freq_sev", frequency=g["gbm"], severity=s["gbm"]),
        TWEEDIE_GBM: PurePremiumModel(TWEEDIE_GBM, "direct", direct=d["gbm"]),
    }
    # Explanations are quoted against the average training policy, weighted
    # by exposure, for every GLM component alike.
    for comp in (f["main"], f["hybrid"], s["glm"], d["glm"]):
        comp.set_reference(ctx.glm.iloc[tr], ctx.exposure[tr])
    return models


def select(valid: dict[str, float]) -> tuple[str, list[dict]]:
    lift = {name: valid[FLAT] - score for name, score in valid.items()}
    shipped = max(TIERS[0], key=lift.get)
    decisions = []
    for tier in TIERS[1:]:
        challenger = max(tier, key=lift.get)
        needed = (1 + C.SHIP_MARGIN) * lift[shipped]
        promoted = lift[challenger] >= needed
        extra = lift[challenger] / lift[shipped] - 1 if lift[shipped] > 0 else None
        decisions.append({
            "incumbent": shipped, "challenger": challenger,
            "incumbent_lift": _r(lift[shipped], 8), "challenger_lift": _r(lift[challenger], 8),
            "needed": _r(needed, 8), "extra_lift": _r(extra, 4), "promoted": bool(promoted),
        })
        if promoted:
            shipped = challenger
    return shipped, decisions


def _frequency_table(ctx, st, te) -> pd.DataFrame:
    f, g = st["frequency_glm"], st["frequency_gbm"]
    y, e = ctx.counts[te], ctx.exposure[te]
    rates = {
        FLAT: np.full(te.size, f["flat_rate"]),
        "GLM, main effects": f["main"].predict(ctx.glm.iloc[te]),
        HYBRID: f["hybrid"].predict(ctx.glm.iloc[te]),
        "Boosted": g["gbm"].predict(ctx.gbm.iloc[te]),
    }
    flat_dev = poisson_deviance(y, rates[FLAT] * e)
    rows = []
    for name, rate in rates.items():
        dev = poisson_deviance(y, rate * e)
        rows.append({
            "model": name, "poisson_deviance": dev, "improvement": (flat_dev - dev) / flat_dev,
            "gini": gini(y, rate, e), "balance": float(np.sum(rate * e) / y.sum()),
        })
    return pd.DataFrame(rows)


def _severity_table(ctx, st, te) -> pd.DataFrame:
    s = st["severity"]
    idx = ctx.claimants(te)
    y, n = ctx.severity(idx)
    preds = {
        FLAT: np.full(idx.size, s["flat"]),
        "GLM": s["glm"].predict(ctx.glm.iloc[idx]),
        "Boosted": s["gbm"].predict(ctx.gbm.iloc[idx]),
    }
    flat_dev = tweedie_deviance(y, preds[FLAT], n, 2.0)
    rows = []
    for name, pred in preds.items():
        dev = tweedie_deviance(y, pred, n, 2.0)
        rows.append({
            "model": name, "gamma_deviance": dev, "improvement": (flat_dev - dev) / flat_dev,
            "gini": gini(y * n, pred, n), "claims": int(n.sum()),
        })
    return pd.DataFrame(rows)


def _large_loss_sensitivity(ctx: Context, st: dict, te: np.ndarray) -> pd.DataFrame:
    """Refit only what the threshold touches (severity and the loading)."""
    tr = ctx.split.train
    freq = st["frequency_glm"]["main"]
    alpha = st["severity"]["alpha"]
    rate = freq.predict(ctx.glm)
    loss_total = ctx.policies["loss"].to_numpy()
    train_ids = set(ctx.policies["IDpol"].to_numpy()[tr])
    train_amounts = ctx.claims.loc[ctx.claims["IDpol"].isin(train_ids), "ClaimAmount"].to_numpy()
    rows = []
    for threshold in C.LARGE_LOSS_SENSITIVITY:
        table = attach_losses(ctx.policies[list(ctx.policies.columns[:12])], ctx.claims, threshold, C.EXPOSURE_CAP)
        loading = LargeLossLoading.fit(train_amounts, threshold)
        claimants = tr[table["n_claims"].to_numpy()[tr] > 0]
        n = table["n_claims"].to_numpy()[claimants].astype(float)
        sev_y = table["loss_capped"].to_numpy()[claimants] / n
        sev = fit_component(ctx.glm.iloc[claimants], sev_y, power=2.0, factors=C.GLM_FACTORS,
                            numeric=C.GLM_NUMERIC, alpha=alpha, weights=n)
        pure = rate * sev.predict(ctx.glm) * (1 + loading.loading)
        e = ctx.exposure
        rows.append({
            "threshold": "none" if threshold is None else f"{int(threshold):,}",
            "loading": loading.loading,
            "train_claims_over_threshold": int(0 if threshold is None else (train_amounts > threshold).sum()),
            "test_gini_total_cost": gini(loss_total[te], pure[te], e[te]),
            "test_balance_total_cost": float(np.sum(pure[te] * e[te]) / loss_total[te].sum()),
        })
    return pd.DataFrame(rows)


def _market(preds: dict, ctx: Context, te: np.ndarray, matchups) -> pd.DataFrame:
    actual = ctx.policies["loss_capped"].to_numpy()[te]
    e = ctx.exposure[te]
    frames = []
    eta = C.COSTS.price_elasticity
    scenarios = (("cheapest quote wins", None), (f"logit demand, elasticity {eta:g}", eta))
    for incumbent, challenger in matchups:
        for scenario, elasticity in scenarios:
            books = simulate((incumbent, challenger), (preds[incumbent][te], preds[challenger][te]),
                             actual, e, C.COSTS, elasticity)
            frame = to_frame(books, scenario, C.COSTS)
            frame.insert(0, "matchup", f"{incumbent} vs {challenger}")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _bias_counts(ctx: Context, preds: dict, names, te) -> pd.DataFrame:
    y = ctx.pure_premium[te]
    e = ctx.exposure[te]
    rows = []
    for name in names:
        for factor in BIAS_FACTORS:
            table = bias_by_factor(y, preds[name][te], ctx.glm[factor].iloc[te], e)
            for _, row in table.iterrows():
                rows.append({"model": name, "factor": factor, "level": row["feature"],
                             "bias_mean": row["bias_mean"], "bias_stderr": row["bias_stderr"],
                             "p_value": row["p_value"], "weight": row["bias_weights"]})
    return pd.DataFrame(rows)


def _bootstrap(ctx: Context, preds: dict, a: str, b: str, te: np.ndarray, n_boot: int) -> dict:
    y = ctx.pure_premium[te]
    w = ctx.exposure[te]
    d_flat = unit_deviance(y, preds[FLAT][te], C.EVAL_POWER) * w
    d_a = unit_deviance(y, preds[a][te], C.EVAL_POWER) * w
    d_b = unit_deviance(y, preds[b][te], C.EVAL_POWER) * w

    def stat(idx):
        flat = d_flat[idx].sum()
        return ((flat - d_a[idx].sum()) - (flat - d_b[idx].sum())) / flat

    draws = group_bootstrap(stat, ctx.groups[te], n_boot, ctx.seed)
    return {
        "a": a, "b": b, "replicates": n_boot,
        "improvement_gap": _r(stat(np.arange(te.size)), 6),
        "ci_low": _r(np.percentile(draws, 2.5), 6), "ci_high": _r(np.percentile(draws, 97.5), 6),
        "share_a_better": _r(np.mean(draws > 0), 4),
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.6g")


def build_report(ctx: Context, paths: Paths, *, synthetic: bool = False) -> dict:
    st = {name: ctx.load_stage(name) for name in STAGES}
    models = assemble(ctx, st)
    frames = Frames(ctx.spec, ctx.policies, glm=ctx.glm, gbm=ctx.gbm)
    preds = {name: model.predict(frames) for name, model in models.items()}

    va, te = ctx.split.valid, ctx.split.test
    y, e = ctx.pure_premium, ctx.exposure
    loss_capped = ctx.policies["loss_capped"].to_numpy()
    loss = ctx.policies["loss"].to_numpy()

    valid = {name: tweedie_deviance(y[va], p[va], e[va], C.EVAL_POWER) for name, p in preds.items()}
    shipped, decisions = select(valid)
    best_glm = max(GLM_MODELS, key=lambda n: valid[FLAT] - valid[n])
    challenger = max((BOOSTED, TWEEDIE_GBM), key=lambda n: valid[FLAT] - valid[n])

    flat_test = tweedie_deviance(y[te], preds[FLAT][te], e[te], C.EVAL_POWER)
    rows = []
    for name, p in preds.items():
        test = tweedie_deviance(y[te], p[te], e[te], C.EVAL_POWER)
        mur = murphy(y[te], p[te], e[te], C.EVAL_POWER)
        rows.append({
            "model": name,
            "valid_deviance": valid[name],
            "valid_improvement": (valid[FLAT] - valid[name]) / valid[FLAT],
            "test_deviance": test,
            "test_improvement": (flat_test - test) / flat_test,
            "gini_attritional": gini(loss_capped[te], p[te], e[te]),
            "gini_total": gini(loss[te], p[te], e[te]),
            "balance_attritional": float(np.sum(p[te] * e[te]) / loss_capped[te].sum()),
            "balance_total": float(np.sum(ctx.loading.apply(p[te]) * e[te]) / loss[te].sum()),
            "miscalibration": mur["miscalibration"],
            "discrimination": mur["discrimination"],
            "uncertainty": mur["uncertainty"],
            "shipped": name == shipped,
        })
    scores = pd.DataFrame(rows)

    frequency = _frequency_table(ctx, st, te)
    severity = _severity_table(ctx, st, te)
    sensitivity = _large_loss_sensitivity(ctx, st, te)
    rel = relativities(st["frequency_glm"]["main"], ctx.glm.iloc[ctx.split.train],
                       ctx.counts[ctx.split.train], offset=ctx.offset[ctx.split.train])
    ae = pd.concat([
        actual_vs_expected(loss_capped[te], preds[n][te] * e[te], e[te]).assign(model=n)
        for n in dict.fromkeys([shipped, best_glm])
    ])
    lift = double_lift(loss_capped[te], preds[challenger][te] * e[te], preds[best_glm][te] * e[te], e[te])
    bias = _bias_counts(ctx, preds, list(dict.fromkeys([shipped, best_glm])), te)
    n_boot = 40 if synthetic else 200
    boot = _bootstrap(ctx, preds, challenger, best_glm, te, n_boot)
    matchups = [(FLAT, best_glm), (best_glm, challenger)]
    market = _market(preds, ctx, te, matchups)
    studies = st["studies"]
    leakage = studies["leakage"]
    exposure = studies["exposure"]

    out = paths.results
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(scores, out / "model_scores.csv")
    _write_csv(frequency, out / "frequency_models.csv")
    _write_csv(severity, out / "severity_models.csv")
    _write_csv(sensitivity, out / "large_loss_sensitivity.csv")
    _write_csv(rel, out / "frequency_relativities.csv")
    _write_csv(ae, out / "actual_vs_expected.csv")
    _write_csv(lift, out / "double_lift.csv")
    _write_csv(bias, out / "bias_by_factor.csv")
    _write_csv(market, out / "market_simulation.csv")
    _write_csv(leakage, out / "split_leakage.csv")
    _write_csv(exposure["by_band"], out / "exposure_treatments.csv")
    _write_csv(exposure["raw"], out / "frequency_by_duration.csv")
    _write_csv(pd.DataFrame(st["frequency_glm"]["history"]), out / "interaction_search.csv")

    pricer = Pricer(
        model=models[shipped], spec=ctx.spec, loading=ctx.loading, costs=C.COSTS,
        reference=None if shipped in GLM_MODELS else models[best_glm],
        metadata={"shipped": shipped, "reference": best_glm, "eval_power": C.EVAL_POWER},
    )
    demo = {label: {"shipped": quote(pricer, profile),
                    "reference": None if pricer.reference is None else quote(pricer, profile, use_reference=True)}
            for label, profile in DEMO_PROFILES.items()} if not synthetic else {}

    metrics = _metrics(ctx, st, scores, frequency, severity, sensitivity, rel, bias, boot, market,
                       leakage, exposure, decisions, shipped, best_glm, challenger, demo)
    closer = (lift["a_relative"] - lift["actual_relative"]).abs() < (lift["b_relative"] - lift["actual_relative"]).abs()
    metrics["double_lift"] = {
        "challenger": challenger, "glm": best_glm,
        "bands_where_challenger_is_closer": int(closer.sum()), "bands": int(len(lift)),
        "bands_detail": lift[["band", "actual_relative", "a_relative", "b_relative"]].round(4).to_dict("records"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(pricer, out / "model.joblib")
    if demo:
        (out / "app_demo.md").write_text(plots.demo_markdown(demo, DEMO_PROFILES))

    plots.draw_all(paths.assets, ctx=ctx, scores=scores, preds=preds, te=te, rel=rel, ae=ae, lift=lift,
                   market=market, exposure=exposure, shipped=shipped, best_glm=best_glm,
                   challenger=challenger, demo=demo)
    return metrics


def _metrics(ctx, st, scores, frequency, severity, sensitivity, rel, bias, boot, market, leakage,
             exposure, decisions, shipped, best_glm, challenger, demo) -> dict:
    by_model = {row["model"]: {k: (_r(v, 6) if isinstance(v, float) else v) for k, v in row.items() if k != "model"}
                for row in scores.to_dict("records")}
    lk = leakage.groupby("split")[["glm_improvement", "gbm_improvement"]].agg(["mean", "std"])
    sig = bias.assign(significant=bias["p_value"] < 0.01).groupby("model")["significant"].agg(["sum", "size"])
    f_glm = st["frequency_glm"]
    freq_gbm = st["frequency_gbm"]["gbm"]
    return {
        "data": {k: ctx.audit[k] for k in (
            "policies", "claims", "policies_with_claims", "exposure_years_raw", "exposure_years_capped",
            "claim_frequency", "exposure_over_cap", "exposure_max", "exposure_below_one_month",
            "policies_with_more_than_4_claims", "claims_on_those_policies", "risk_profiles",
            "rows_sharing_a_profile", "share_rows_sharing_a_profile", "largest_profile",
            "consecutive_id_share_within_profiles", "vehicle_age_over_30", "driver_age_over_90")},
        "reconciliation": ctx.audit["reconciliation"],
        "severity_profile": ctx.audit["severity"],
        "area_density_ranges": ctx.audit["area_density_ranges"],
        "split": {**ctx.split.sizes(),
                  "test_exposure_years": _r(ctx.exposure[ctx.split.test].sum(), 2),
                  "test_claims": int(ctx.counts[ctx.split.test].sum())},
        "large_loss": {"threshold": ctx.loading.threshold, "loading_on_train": _r(ctx.loading.loading, 6)},
        "frequency_glm": {
            "alpha": f_glm["alpha"], "interactions": [f"{a} x {b}" for a, b in f_glm["interactions"]],
            "valid_flat": _r(f_glm["valid"]["flat"], 8), "valid_main": _r(f_glm["valid"]["main"], 8),
            "min_gain": _r(f_glm["valid"]["min_gain"], 8),
            "alpha_scores": [{"alpha": s["alpha"], "deviance": _r(s["deviance"], 8)} for s in f_glm["alpha_scores"]],
        },
        "frequency_gbm": {"rounds": int(freq_gbm.best_iteration),
                          "params": {k: freq_gbm.params[k] for k in ("num_leaves", "min_data_in_leaf")},
                          "balance_scale": _r(freq_gbm.scale, 6)},
        "severity_models": {"alpha": st["severity"]["alpha"],
                            "flat_mean_capped_claim": _r(st["severity"]["flat"], 2),
                            "valid_deviance": {k: _r(v, 8) for k, v in st["severity"]["valid"].items()},
                            "alpha_scores": [{"alpha": s["alpha"], "deviance": _r(s["deviance"], 8)}
                                             for s in st["severity"]["alpha_scores"]]},
        "tweedie": {"power": st["direct"]["power"],
                    "gbm_balance_scale": _r(st["direct"]["gbm"].scale, 6)},
        "frequency_models": frequency.round(6).to_dict("records"),
        "severity_models_test": severity.round(6).to_dict("records"),
        "pure_premium_models": by_model,
        "selection": {"shipped": shipped, "best_glm": best_glm, "challenger": challenger,
                      "ship_margin": C.SHIP_MARGIN, "decisions": decisions},
        "bootstrap": boot,
        "bias_levels_p_below_0_01": {m: {"significant": int(r["sum"]), "levels": int(r["size"])}
                                    for m, r in sig.iterrows()},
        "large_loss_sensitivity": sensitivity.round(6).to_dict("records"),
        "relativities_frequency": rel.round(6).to_dict("records"),
        "market": market.round(6).to_dict("records"),
        "split_leakage": {
            **{
                split: {"glm_mean": _r(lk.loc[split, ("glm_improvement", "mean")]),
                        "glm_sd": _r(lk.loc[split, ("glm_improvement", "std")]),
                        "gbm_mean": _r(lk.loc[split, ("gbm_improvement", "mean")]),
                        "gbm_sd": _r(lk.loc[split, ("gbm_improvement", "std")])}
                for split in lk.index
            },
            "seeds": int(leakage["seed"].nunique()),
            "random_split_test_rows_with_profile_in_train": _r(
                leakage.loc[leakage["split"] == "random", "test_rows_with_profile_in_train"].mean(), 4),
        },
        "exposure": {
            "summary": {k: {kk: _r(vv) for kk, vv in v.items()} for k, v in exposure["summary"].items()},
            "frequency_by_duration": exposure["raw"].round(6).to_dict("records"),
            "a_over_e": exposure["by_band"][["model", "duration", "a_over_e"]].round(4).to_dict("records"),
            "duration_relativities": exposure["duration_relativities"]
            .query("factor == 'duration'")[["level", "relativity", "ci_low", "ci_high"]].round(4).to_dict("records"),
        },
        "demo_quotes": {
            label: {
                which: None if q is None else {
                    "frequency": _r(q.frequency), "severity": _r(q.severity, 2),
                    "technical": _r(q.technical, 2), "office": _r(q.office, 2),
                    "reasons": [r.sentence() for r in q.reasons],
                }
                for which, q in quotes.items()
            }
            for label, quotes in demo.items()
        },
        "costs": {"target_loss_ratio": C.COSTS.target_loss_ratio, "expense_ratio": C.COSTS.expense_ratio,
                  "price_elasticity": C.COSTS.price_elasticity, "book_policy_years": C.COSTS.book_policy_years},
        "stage_seconds": {name: st[name]["seconds"] for name in STAGES},
    }
