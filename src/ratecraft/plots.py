"""Charts for the README. Drawing only: every number comes in from the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ratecraft.config import BANDS, COSTS  # noqa: E402
from ratecraft.metrics import lorenz_curve  # noqa: E402

INK = "#1f2933"
MUTED = "#7b8794"
COLORS = {
    "Flat rate": "#9aa5b1",
    "GLM tariff": "#1f4e79",
    "Tweedie GLM": "#4f81bd",
    "GLM with interactions": "#2a9d8f",
    "Boosted frequency x severity": "#c0504d",
    "Tweedie GBM": "#e9a03b",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#cbd2d9",
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.color": "#e4e7eb",
    "grid.linewidth": 0.7,
})

FACTOR_TITLES = {
    "drv_age": "Driver age",
    "bonus_malus": "Bonus-malus level",
    "veh_age": "Vehicle age",
    "veh_power": "Vehicle power band",
}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def audit_chart(amounts: np.ndarray, threshold: float, raw_duration: pd.DataFrame, severity: dict, path: Path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    bins = np.logspace(0, np.log10(amounts.max() * 1.05), 70)
    left.hist(amounts, bins=bins, color="#4f81bd", edgecolor="white", linewidth=0.3)
    left.set_xscale("log")
    left.set_yscale("log")
    left.axvline(threshold, color="#c0504d", lw=1.2, ls="--")
    top = severity["most_common_amounts"][0]
    left.annotate(
        f"EUR {top['amount']:,.0f} exactly:\n{top['share']:.0%} of all claims",
        xy=(top["amount"], top["claims"]), xytext=(8, 2_500), fontsize=8.5, color=INK,
        arrowprops={"arrowstyle": "->", "color": MUTED},
    )
    left.text(threshold * 1.15, 1_200,
              f"cap at EUR {threshold:,.0f}\n{severity['claims_over_threshold']} claims go past it;\n"
              f"the excess is {severity['excess_share_of_cost']:.1%}\nof all claim cost",
              fontsize=8.5, color="#c0504d", va="top")
    left.set_xlabel("claim amount (EUR, log scale)")
    left.set_ylabel("claims (log scale)")
    left.set_title("Claim sizes: a fixed-price spike and a long tail")

    freq = raw_duration["claims_per_year"].to_numpy()
    labels = raw_duration["duration"].tolist()
    bars = right.bar(range(len(freq)), freq, color=["#c0504d"] * (len(freq) - 1) + ["#1f4e79"])
    for b, v in zip(bars, freq):
        right.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=8.5)
    right.set_xticks(range(len(freq)), labels, rotation=25, ha="right")
    right.set_ylabel("claims per policy-year")
    right.set_title("Short policies claim far more per year")
    right.set_ylim(0, freq.max() * 1.18)
    _save(fig, path)


def lorenz_chart(actual, exposure, preds: dict, names, path: Path, scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], color=COLORS["Flat rate"], lw=1, ls="--", label="flat rate (Gini 0)")
    gini_of = scores.set_index("model")["gini_attritional"]
    for name in names:
        x, y = lorenz_curve(actual, preds[name], exposure)
        ax.plot(x, y, color=COLORS[name], lw=1.6, label=f"{name} (Gini {gini_of[name]:.3f})")
    ax.set_xlabel("share of exposure, cheapest predicted first")
    ax.set_ylabel("share of attritional claims cost")
    ax.set_title("Ordered Lorenz curves, held-out policies")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, path)


def murphy_chart(scores: pd.DataFrame, path: Path):
    frame = scores[scores["model"] != "Flat rate"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(frame))
    ax.barh(y + 0.18, frame["discrimination"], height=0.34, color="#1f4e79", label="discrimination (higher is better)")
    ax.barh(y - 0.18, frame["miscalibration"], height=0.34, color="#c0504d", label="miscalibration (lower is better)")
    ax.set_yticks(y, frame["model"])
    ax.set_xlabel("Tweedie deviance units (power 1.5), held-out policies")
    ax.set_title("What each model's score is made of")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=8)
    _save(fig, path)


def calibration_chart(ae: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, part in ae.groupby("model", sort=False):
        ax.plot(part["band"], part["a_over_e"], marker="o", lw=1.5, color=COLORS[name], label=name)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("decile of predicted pure premium (equal exposure)")
    ax.set_ylabel("actual / expected")
    ax.set_title("Calibration across the price range")
    ax.legend(fontsize=8)
    _save(fig, path)


def double_lift_chart(lift: pd.DataFrame, challenger: str, glm: str, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(lift["band"], lift["actual_relative"], color=INK, marker="o", lw=1.8, label="actual")
    ax.plot(lift["band"], lift["a_relative"], color=COLORS[challenger], marker="s", lw=1.4, label=challenger)
    ax.plot(lift["band"], lift["b_relative"], color=COLORS[glm], marker="^", lw=1.4, label=glm)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel(f"decile of {challenger} / {glm} price ratio")
    ax.set_ylabel("cost per policy-year, relative to book")
    ax.set_title("Double lift: where the two models disagree")
    ax.legend(fontsize=8)
    _save(fig, path)


def relativity_chart(rel: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, factor in zip(axes.flat, FACTOR_TITLES):
        spec = BANDS[factor]
        part = rel[rel["factor"] == factor].set_index("level")
        levels = list(spec.labels)
        values = [1.0 if lv == spec.base else part.loc[lv, "relativity"] for lv in levels]
        low = [1.0 if lv == spec.base else part.loc[lv, "ci_low"] for lv in levels]
        high = [1.0 if lv == spec.base else part.loc[lv, "ci_high"] for lv in levels]
        x = np.arange(len(levels))
        ax.errorbar(x, values, yerr=[np.subtract(values, low), np.subtract(high, values)], fmt="o",
                    color="#1f4e79", ecolor="#9aa5b1", capsize=3, ms=5)
        ax.axhline(1.0, color=MUTED, lw=1, ls="--")
        ax.set_xticks(x, levels, rotation=30 if len(levels) > 7 else 0)
        ax.set_title(f"{FACTOR_TITLES[factor]} (base {spec.base})")
        ax.set_ylabel("frequency relativity")
    _save(fig, path)


def exposure_chart(by_band: pd.DataFrame, path: Path):
    order = ["under 1 month", "1-3 months", "3-6 months", "6-9 months", "9-12 months", "full year"]
    models = list(dict.fromkeys(by_band["model"]))
    palette = ["#1f4e79", "#c0504d", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    width = 0.8 / len(models)
    for i, (name, color) in enumerate(zip(models, palette)):
        part = by_band[by_band["model"] == name].set_index("duration").reindex(order)
        ax.bar(np.arange(len(order)) + (i - (len(models) - 1) / 2) * width, part["a_over_e"], width,
               color=color, label=name)
    ax.axhline(1.0, color=INK, lw=1, ls="--")
    ax.set_xticks(range(len(order)), order)
    ax.set_ylabel("actual / expected claims")
    ax.set_title("Three ways to handle exposure, scored by policy duration")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    _save(fig, path)


def market_chart(market: pd.DataFrame, target: float, path: Path):
    matchups = list(dict.fromkeys(market["matchup"]))
    scenarios = list(dict.fromkeys(market["scenario"]))
    fig, axes = plt.subplots(1, len(matchups), figsize=(5.2 * len(matchups), 3.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, matchup in zip(axes, matchups):
        part = market[market["matchup"] == matchup]
        insurers = list(dict.fromkeys(part["insurer"]))
        x = np.arange(len(scenarios))
        for i, insurer in enumerate(insurers):
            rows = part[part["insurer"] == insurer].set_index("scenario").reindex(scenarios)
            vals = rows["loss_ratio"].tolist()
            shares = rows["share_of_customers"].tolist()
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=COLORS.get(insurer, MUTED), label=insurer)
            for b, v, s in zip(bars, vals, shares):
                ax.text(b.get_x() + b.get_width() / 2, v - 0.03, f"{v:.0%}", ha="center", va="top",
                        fontsize=9, color="white", fontweight="bold")
                ax.text(b.get_x() + b.get_width() / 2, 0.04, f"wins\n{s:.0%}", ha="center", va="bottom",
                        fontsize=7.5, color="white")
        ax.axhline(target, color=INK, lw=1, ls="--", label=f"target loss ratio {target:.0%}")
        ax.set_xticks(x, [s.replace(", elasticity", "\nelasticity") for s in scenarios])
        ax.set_title(matchup, fontsize=9.5)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("loss ratio on the book each insurer wins")
    top = market["loss_ratio"].max()
    axes[0].set_ylim(0, top * 1.35)
    _save(fig, path)


def demo_markdown(demo: dict, profiles: dict) -> str:
    lines = ["# Quotes from the shipped pricer", "",
             "Produced by `scripts/run_all.py` through the same `quote()` function the app calls.", ""]
    for label, quotes in demo.items():
        p = profiles[label]
        lines += [f"## {label}", "",
                  f"Driver {p['DrivAge']}, bonus-malus {p['BonusMalus']}, vehicle age {p['VehAge']}, "
                  f"power band {p['VehPower']}, brand {p['VehBrand']}, {p['VehGas'].lower()} fuel, "
                  f"{p['Density']:,} people per km2, {p['Region']}.", ""]
        for which, q in quotes.items():
            if q is None:
                continue
            title = "shipped model" if which == "shipped" else "best GLM (with interactions)"
            freq = "" if q.frequency is None else f", {q.frequency:.4f} claims a year"
            lines.append(f"- {title}: office premium EUR {q.office:,.0f} (technical EUR {q.technical:,.0f}{freq})")
        lines += ["", "Biggest factors in the shipped price:", ""]
        lines += [f"- {r.sentence()}" for r in quotes["shipped"].reasons]
        lines.append("")
    return "\n".join(lines)


def demo_chart(demo: dict, path: Path):
    n = len(demo)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 3.6))
    for ax, (label, quotes) in zip(np.atleast_1d(axes), demo.items()):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#f5f7fa", edgecolor="#cbd2d9", lw=1))
        q = quotes["shipped"]
        ax.text(0.05, 0.9, label, fontsize=10, fontweight="bold", color=INK)
        ax.text(0.05, 0.76, f"EUR {q.office:,.0f}", fontsize=19, fontweight="bold", color="#1f4e79")
        ref = quotes.get("reference")
        sub = "office premium, shipped model"
        if ref is not None:
            sub += f"\nthe best GLM would charge EUR {ref.office:,.0f}"
        ax.text(0.05, 0.62, sub, fontsize=8, color=MUTED, va="top")
        y = 0.44
        for r in q.reasons[:4]:
            pct = (r.effect - 1) * 100
            color = "#c0504d" if pct >= 0 else "#2a9d8f"
            ax.text(0.05, y, f"{'+' if pct >= 0 else '-'}{abs(pct):.0f}%", fontsize=9.5, color=color,
                    fontweight="bold")
            ax.text(0.24, y, f"{r.factor} {r.value}", fontsize=8.5, color=INK)
            y -= 0.1
    _save(fig, path)


def draw_all(assets: Path, *, ctx, scores, preds, te, rel, ae, lift, market, exposure, shipped, best_glm,
             challenger, demo) -> None:
    assets.mkdir(parents=True, exist_ok=True)
    severity = ctx.audit["severity"]
    audit_chart(ctx.claims["ClaimAmount"].to_numpy(), severity["threshold"], exposure["raw"], severity,
                assets / "data_audit.png")
    loss_capped = ctx.policies["loss_capped"].to_numpy()[te]
    names = list(dict.fromkeys(["GLM tariff", best_glm, challenger, shipped]))
    lorenz_chart(loss_capped, ctx.exposure[te], {n: preds[n][te] for n in names}, names,
                 assets / "lorenz.png", scores)
    murphy_chart(scores, assets / "murphy.png")
    calibration_chart(ae, assets / "calibration.png")
    double_lift_chart(lift, challenger, best_glm, assets / "double_lift.png")
    relativity_chart(rel, assets / "relativities.png")
    exposure_chart(exposure["by_band"], assets / "exposure.png")
    market_chart(market, COSTS.target_loss_ratio, assets / "market.png")
    if demo:
        demo_chart(demo, assets / "app_demo.png")
