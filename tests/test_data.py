from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ratecraft.config import DataSource
from ratecraft.data import (
    DataIntegrityError,
    attach_losses,
    fetch,
    normalise_claims,
    normalise_policies,
    parse_policy_ids,
    sha256,
)


def test_policy_ids_written_in_scientific_notation_are_parsed():
    ids = parse_policy_ids(pd.Series(["99999", "1e+05", "100001"], dtype="category"))
    assert ids.tolist() == [99_999, 100_000, 100_001]
    assert ids.dtype == np.int64


def test_non_integral_policy_id_is_rejected():
    with pytest.raises(ValueError, match="non-integral"):
        parse_policy_ids(pd.Series(["1.5"]))


def _raw_policy_frame(n=4):
    """Shaped like rdata's output: 1-based string row names, Int32 and factors."""
    index = pd.Index([str(i) for i in range(1, n + 1)])
    return pd.DataFrame(
        {
            "IDpol": pd.Categorical([str(i * 10) for i in range(1, n + 1)]),
            "ClaimNb": np.arange(n, dtype=float) % 2,
            "Exposure": np.linspace(0.1, 1.2, n),
            "VehPower": pd.array([5, 6, 7, 8][:n], dtype="Int32"),
            "VehAge": pd.array([0, 3, 10, 25][:n], dtype="Int32"),
            "DrivAge": pd.array([18, 40, 60, 90][:n], dtype="Int32"),
            "BonusMalus": pd.array([50, 50, 80, 120][:n], dtype="Int32"),
            "VehBrand": pd.Categorical(["B12", "B1", "B2", "B12"][:n]),
            "VehGas": pd.Categorical(["Regular", "Diesel", "Diesel", "Regular"][:n]),
            "Area": pd.Categorical(["A", "C", "D", "F"][:n]),
            "Density": pd.array([20, 300, 1500, 20000][:n], dtype="Int32"),
            "Region": pd.Categorical(["Centre", "Bretagne", "Centre", "Ile-de-France"][:n]),
        },
        index=index,
    )


def test_normalise_keeps_factor_columns_aligned_with_their_rows():
    # Regression test: assigning a Series built on the 1-based index used to
    # shift every factor by one row and leave the first as NaN.
    out = normalise_policies(_raw_policy_frame())
    assert out["VehBrand"].astype(str).tolist() == ["B12", "B1", "B2", "B12"]
    assert out["Region"].astype(str).tolist() == ["Centre", "Bretagne", "Centre", "Ile-de-France"]
    assert out.isna().sum().sum() == 0
    assert out["IDpol"].tolist() == [10, 20, 30, 40]
    assert out["ClaimNb"].dtype == np.int64


def test_normalise_rejects_missing_columns_and_duplicate_ids():
    raw = _raw_policy_frame()
    with pytest.raises(ValueError, match="missing"):
        normalise_policies(raw.drop(columns=["Region"]))
    dup = raw.copy()
    dup["IDpol"] = pd.Categorical(["10", "10", "30", "40"])
    with pytest.raises(ValueError, match="duplicate"):
        normalise_policies(dup)


def test_attach_losses_caps_each_claim_and_the_exposure():
    policies = normalise_policies(_raw_policy_frame())
    claims = normalise_claims(pd.DataFrame({"IDpol": ["20", "20", "40"], "ClaimAmount": [1_000.0, 80_000.0, 500.0]}))
    out = attach_losses(policies, claims, threshold=50_000.0, exposure_cap=1.0)
    row = out.set_index("IDpol")
    assert row.loc[20, "n_claims"] == 2
    assert row.loc[20, "loss"] == pytest.approx(81_000.0)
    assert row.loc[20, "loss_capped"] == pytest.approx(51_000.0)
    assert row.loc[10, "loss"] == 0.0 and row.loc[10, "n_claims"] == 0
    assert out["exposure"].max() == pytest.approx(1.0)
    assert out["Exposure"].max() > 1.0  # the raw column is left alone


def test_no_threshold_means_no_cap():
    policies = normalise_policies(_raw_policy_frame())
    claims = normalise_claims(pd.DataFrame({"IDpol": ["20"], "ClaimAmount": [80_000.0]}))
    out = attach_losses(policies, claims, threshold=None, exposure_cap=1.0)
    assert out["loss_capped"].sum() == pytest.approx(80_000.0)


def test_fetch_trusts_files_that_match_their_digest(tmp_path):
    (tmp_path / "a.rda").write_bytes(b"policy table")
    source = DataSource(files={"a.rda": sha256(tmp_path / "a.rda")}, repo="https://example.invalid/none")
    # A verified file means no download is attempted, so the bogus repo is never touched.
    assert fetch(tmp_path, source) == {"a.rda": tmp_path / "a.rda"}


def test_fetch_refuses_a_file_that_does_not_match(tmp_path, monkeypatch):
    (tmp_path / "a.rda").write_bytes(b"tampered")
    source = DataSource(files={"a.rda": "0" * 64}, repo="https://example.invalid/none")
    import ratecraft.data as data_module

    monkeypatch.setattr(data_module, "_fetch_with_git", lambda *a, **k: None)
    with pytest.raises(DataIntegrityError, match="expected sha256"):
        fetch(tmp_path, source)


def test_raw_url_points_at_the_pinned_commit():
    source = DataSource()
    url = source.raw_url("freMTPL2freq.rda")
    assert source.commit in url and url.endswith("/data/freMTPL2freq.rda")
