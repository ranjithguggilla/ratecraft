from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ratecraft.audit import audit, consecutive_id_share, reconcile, severity_profile


def test_reconcile_counts_every_kind_of_disagreement():
    policies = pd.DataFrame({"IDpol": [1, 2, 3, 4], "ClaimNb": [1, 2, 0, 1]})
    claims = pd.DataFrame({
        "IDpol": [1, 2, 3, 9],  # 2 has one row for two claims, 3 has an amount but no count, 9 is unknown
        "ClaimAmount": [100.0, 0.0, 50.0, 10.0],
    })
    out = reconcile(policies, claims)
    assert out == {
        "claims_without_policy": 1,
        "policies_counting_claims_with_no_amounts": 1,  # policy 4
        "policies_with_amounts_but_zero_count": 1,  # policy 3
        "policies_where_counts_disagree": 1,  # policy 2
        "non_positive_amounts": 1,
    }


def test_severity_profile_on_a_known_sample():
    amounts = np.array([1204.0] * 6 + [1000.0, 2000.0, 100_000.0, 300_000.0])
    out = severity_profile(amounts, threshold=50_000.0)
    assert out["claims"] == 10
    assert out["claims_over_threshold"] == 2
    assert out["excess_share_of_cost"] == pytest.approx((50_000 + 250_000) / amounts.sum(), abs=1e-4)
    assert out["most_common_amounts"][0] == {"amount": 1204.0, "claims": 6, "share": 0.6}
    assert out["max"] == 300_000.0


def test_consecutive_ids_inside_profiles():
    groups = np.array([0, 0, 0, 1, 1, 2])
    ids = np.array([10, 11, 15, 30, 32, 40])
    # pairs inside a group: (10,11) yes, (11,15) no, (30,32) no
    assert consecutive_id_share(ids, groups) == pytest.approx(1 / 3)
    assert consecutive_id_share(np.array([1, 2]), np.array([0, 1])) == 0.0


def test_audit_on_the_synthetic_book(portfolio):
    policies, claims = portfolio
    out = audit(policies, claims, exposure_cap=1.0, threshold=50_000.0)
    assert out["policies"] == len(policies)
    assert out["claims"] == int(policies["ClaimNb"].sum())
    assert out["exposure_over_cap"] == int((policies["Exposure"] > 1).sum()) > 0
    assert out["reconciliation"]["policies_where_counts_disagree"] == 0
    assert out["exposure_years_capped"] <= out["exposure_years_raw"]
    assert set(out["area_density_ranges"]) <= set("ABCDEF")
