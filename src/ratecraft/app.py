"""A quoting screen: type a customer in, get a price and the reasons for it.

    python scripts/run_all.py      # fits and saves results/model.joblib
    ratecraft-app                  # or: python -m ratecraft.app

The screen is a thin shell. All the pricing happens in ``render_quote``, which
calls the same ``quote()`` the report used for its demo, and that function is
what the tests exercise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from ratecraft.pricing import Pricer, quote

DEFAULT_MODEL = Path("results/model.joblib")

DEFAULT_PROFILE = {
    "DrivAge": 35, "VehAge": 5, "VehPower": 6, "BonusMalus": 50, "VehBrand": "B12",
    "VehGas": "Regular", "Density": 1_000, "Region": "Centre",
}


def load_pricer(path: Path = DEFAULT_MODEL) -> Pricer:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/run_all.py` first to fit and save the model")
    return joblib.load(path)


def render_quote(pricer: Pricer, profile: dict) -> str:
    """Markdown for one customer. Bad input comes back as a message, not a crash."""
    try:
        q = quote(pricer, profile)
    except ValueError as exc:
        return f"Cannot price this profile: {exc}"
    costs = pricer.costs
    lines = [f"### Office premium: EUR {q.office:,.0f} a year", ""]
    if q.frequency is not None and q.severity is not None:
        lines.append(f"Expected claims: {q.frequency:.3f} a year at EUR {q.severity:,.0f} each (capped)")
    lines += [
        f"Attritional cost EUR {q.attritional:,.0f} + large-loss allowance EUR {q.large_loss_allowance:,.0f} "
        f"= technical price EUR {q.technical:,.0f}",
        f"Office premium = technical price / {costs.target_loss_ratio:.0%} target loss ratio",
        "",
    ]
    if pricer.reference is not None:
        ref = quote(pricer, profile, use_reference=True)
        name = pricer.metadata.get("reference", "the reference GLM")
        lines += [f"The {name} would charge EUR {ref.office:,.0f}.", ""]
    lines.append("What moved the price, against an average policy:")
    lines += [f"- {r.sentence()}" for r in q.reasons]
    return "\n".join(lines)


def build_app(pricer: Pricer):
    import gradio as gr

    levels = pricer.spec.levels

    def price(drv_age, veh_age, veh_power, bonus_malus, brand, gas, density, region):
        return render_quote(pricer, {
            "DrivAge": drv_age, "VehAge": veh_age, "VehPower": veh_power, "BonusMalus": bonus_malus,
            "VehBrand": brand, "VehGas": gas, "Density": density, "Region": region,
        })

    shipped = pricer.metadata.get("shipped", "shipped model")
    with gr.Blocks(title="ratecraft quote") as demo:
        gr.Markdown(f"## Motor TPL quote\nPriced by the {shipped}, with the reasons in plain words.")
        with gr.Row():
            with gr.Column():
                drv_age = gr.Slider(18, 100, value=DEFAULT_PROFILE["DrivAge"], step=1, label="Driver age")
                bonus_malus = gr.Slider(50, 230, value=DEFAULT_PROFILE["BonusMalus"], step=1,
                                        label="Bonus-malus level (50 is the best record, 100 is a new driver)")
                veh_age = gr.Slider(0, 30, value=DEFAULT_PROFILE["VehAge"], step=1, label="Vehicle age")
                veh_power = gr.Slider(4, 15, value=DEFAULT_PROFILE["VehPower"], step=1, label="Vehicle power band")
            with gr.Column():
                brand = gr.Dropdown(list(levels["veh_brand"]), value=DEFAULT_PROFILE["VehBrand"], label="Vehicle brand")
                gas = gr.Radio(list(levels["veh_gas"]), value=DEFAULT_PROFILE["VehGas"], label="Fuel")
                density = gr.Number(value=DEFAULT_PROFILE["Density"], minimum=1, maximum=27_000,
                                    label="Population density where the car is kept (people per km2)")
                region = gr.Dropdown(list(levels["region"]), value=DEFAULT_PROFILE["Region"], label="Region")
        button = gr.Button("Price it", variant="primary")
        output = gr.Markdown(render_quote(pricer, DEFAULT_PROFILE))
        button.click(price, [drv_age, veh_age, veh_power, bonus_malus, brand, gas, density, region], output)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the quoting screen.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(load_pricer(args.model)).launch(server_port=args.port)


if __name__ == "__main__":
    main()
