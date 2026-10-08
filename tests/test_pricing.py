from __future__ import annotations

import numpy as np
import pytest

from ratecraft.pricing import Frames, PurePremiumModel, Reason, profile_frame, quote


def test_profile_validation(profile):
    frame = profile_frame(profile)
    assert frame.shape == (1, 8)
    with pytest.raises(ValueError, match="missing DrivAge"):
        profile_frame({k: v for k, v in profile.items() if k != "DrivAge"})
    with pytest.raises(ValueError, match="outside"):
        profile_frame({**profile, "BonusMalus": 40})
    with pytest.raises(ValueError, match="missing Region"):
        profile_frame({**profile, "Region": ""})


def test_quote_adds_up(pricer, profile):
    q = quote(pricer, profile)
    assert q.attritional == pytest.approx(q.frequency * q.severity)
    assert q.technical == pytest.approx(q.attritional * (1 + pricer.loading.loading))
    assert q.large_loss_allowance == pytest.approx(q.technical - q.attritional)
    assert q.office == pytest.approx(q.technical / pricer.costs.target_loss_ratio)
    assert len(q.reasons) == 4
    sizes = [abs(np.log(r.effect)) for r in q.reasons]
    assert sizes == sorted(sizes, reverse=True)


def test_reasons_multiply_back_to_the_price(pricer, profile):
    """Every factor's effect against the average policy, multiplied together and
    applied to the average policy's price, has to give the quoted price."""
    q = quote(pricer, profile, top=50)
    freq, sev = pricer.model.frequency, pricer.model.severity
    average = np.exp(
        freq.model.intercept_ + sum(freq.reference.values()) + sev.model.intercept_ + sum(sev.reference.values())
    )
    product = np.prod([r.effect for r in q.reasons])
    assert average * product == pytest.approx(q.attritional, rel=1e-9)


def test_a_worse_record_costs_more(pricer, profile):
    clean = quote(pricer, {**profile, "BonusMalus": 50}).office
    marked = quote(pricer, {**profile, "BonusMalus": 150}).office
    assert marked > clean


def test_reason_sentences():
    assert Reason("Driver age", "19", 1.25).sentence() == "Driver age 19: 25% dearer than an average policy"
    assert Reason("Fuel", "Diesel", 0.8).sentence() == "Fuel Diesel: 20% cheaper than an average policy"


def test_reference_quote_requires_a_reference(pricer, profile):
    with pytest.raises(ValueError, match="no reference"):
        quote(pricer, profile, use_reference=True)


def test_flat_model_and_frame_subsets(pricer, table):
    frames = Frames(pricer.spec, table)
    flat = PurePremiumModel("Flat rate", "flat", flat_rate=123.0)
    assert np.all(flat.predict(frames.subset(np.arange(10))) == 123.0)
    assert flat.frequency_rate(frames) is None
    vectorised = pricer.office(table.head(25))
    assert vectorised.shape == (25,)
    assert np.allclose(vectorised, pricer.technical(table.head(25)) / pricer.costs.target_loss_ratio)
