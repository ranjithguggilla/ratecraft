"""End to end on a synthetic book: every stage, the report, the charts and the
saved pricer, through the same entry point CI uses on the real data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import pytest

from ratecraft.pricing import quote

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    _load_script("run_all").main(["--synthetic", "6000", "--root", str(root)])
    return root


def test_report_files_are_written(synthetic_run):
    results = synthetic_run / "results"
    for name in ("metrics.json", "model_scores.csv", "frequency_models.csv", "severity_models.csv",
                 "market_simulation.csv", "exposure_treatments.csv", "split_leakage.csv", "model.joblib"):
        assert (results / name).exists(), name
    for chart in ("lorenz.png", "murphy.png", "calibration.png", "double_lift.png", "relativities.png",
                  "exposure.png", "market.png", "data_audit.png"):
        assert (synthetic_run / "assets" / chart).stat().st_size > 10_000, chart


def test_metrics_have_the_numbers_the_readme_quotes(synthetic_run):
    metrics = json.loads((synthetic_run / "results" / "metrics.json").read_text())
    for key in ("data", "split", "pure_premium_models", "selection", "bootstrap", "market", "exposure",
                "frequency_models", "severity_models_test", "large_loss_sensitivity", "split_leakage"):
        assert key in metrics, key
    assert metrics["selection"]["shipped"] in metrics["pure_premium_models"]
    flat = metrics["pure_premium_models"]["Flat rate"]
    assert flat["test_improvement"] == 0.0 and flat["gini_attritional"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["data"]["policies"] == 6000


def test_saved_pricer_quotes(synthetic_run, profile):
    pricer = joblib.load(synthetic_run / "results" / "model.joblib")
    q = quote(pricer, profile)
    assert q.office > q.technical > q.attritional > 0


def test_verify_metrics_accepts_identical_runs_and_flags_drift(synthetic_run, tmp_path):
    verify = _load_script("verify_metrics")
    committed = synthetic_run / "results" / "metrics.json"
    assert verify.compare(json.loads(committed.read_text()), json.loads(committed.read_text())) == []
    drifted = json.loads(committed.read_text())
    drifted["selection"]["shipped"] = "something else"
    assert verify.compare(json.loads(committed.read_text()), drifted)
