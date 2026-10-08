from __future__ import annotations

import pytest

from ratecraft.app import DEFAULT_PROFILE, load_pricer, render_quote


def test_render_quote_shows_the_price_and_the_reasons(pricer, profile):
    text = render_quote(pricer, profile)
    assert text.startswith("### Office premium: EUR ")
    assert "What moved the price" in text
    assert text.count("\n- ") == 4


def test_bad_input_comes_back_as_a_message(pricer, profile):
    text = render_quote(pricer, {**profile, "DrivAge": 12})
    assert text.startswith("Cannot price this profile")


def test_the_default_profile_prices(pricer):
    assert "EUR" in render_quote(pricer, DEFAULT_PROFILE)


def test_missing_model_file_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_all.py"):
        load_pricer(tmp_path / "nope.joblib")


def test_the_screen_builds_without_launching(pricer):
    gr = pytest.importorskip("gradio")
    from ratecraft.app import build_app

    assert isinstance(build_app(pricer), gr.Blocks)
