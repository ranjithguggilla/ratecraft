"""Measure the data before trusting it.

Every check returns a number rather than silently fixing something, so the
README can say exactly what was wrong with the table and how much of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ratecraft.split import profile_groups


def _native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def reconcile(policies: pd.DataFrame, claims: pd.DataFrame) -> dict:
    """Do the policy-level claim counts agree with the claim-level table?"""
    rows_per_policy = claims.groupby("IDpol").size()
    known = claims["IDpol"].isin(policies["IDpol"])
    recorded = rows_per_policy.reindex(policies["IDpol"].to_numpy()).fillna(0).to_numpy()
    counts = policies["ClaimNb"].to_numpy()
    return {
        "claims_without_policy": int((~known).sum()),
        "policies_counting_claims_with_no_amounts": int(((counts > 0) & (recorded == 0)).sum()),
        "policies_with_amounts_but_zero_count": int(((counts == 0) & (recorded > 0)).sum()),
        "policies_where_counts_disagree": int(((counts > 0) & (recorded > 0) & (counts != recorded)).sum()),
        "non_positive_amounts": int((claims["ClaimAmount"] <= 0).sum()),
    }


def severity_profile(amounts: np.ndarray, threshold: float) -> dict:
    amounts = np.asarray(amounts, dtype=float)
    total = amounts.sum()
    top_k = max(1, int(round(0.01 * amounts.size)))
    top_share = np.sort(amounts)[-top_k:].sum() / total
    values, freq = np.unique(np.round(amounts, 2), return_counts=True)
    order = np.argsort(freq)[::-1][:3]
    over = amounts > threshold
    return {
        "claims": int(amounts.size),
        "total": float(round(total, 2)),
        "mean": float(round(amounts.mean(), 2)),
        "median": float(round(np.median(amounts), 2)),
        "quantiles": {
            str(q): float(round(np.quantile(amounts, q), 2)) for q in (0.9, 0.99, 0.995, 0.999)
        },
        "max": float(round(amounts.max(), 2)),
        "top_1pct_share_of_cost": float(round(top_share, 4)),
        "most_common_amounts": [
            {"amount": float(values[i]), "claims": int(freq[i]), "share": float(round(freq[i] / amounts.size, 4))}
            for i in order
        ],
        "share_between_1100_and_1300": float(round(((amounts > 1100) & (amounts < 1300)).mean(), 4)),
        "threshold": float(threshold),
        "claims_over_threshold": int(over.sum()),
        "excess_share_of_cost": float(round(np.maximum(amounts - threshold, 0).sum() / total, 4)),
    }


def consecutive_id_share(policy_ids: np.ndarray, groups: np.ndarray) -> float:
    """Among neighbouring rows of the same risk profile (sorted by policy id),
    the share whose ids differ by exactly one. High values mean the rows look
    like one contract cut into pieces rather than strangers who happen to match."""
    frame = pd.DataFrame({"g": groups, "id": policy_ids}).sort_values(["g", "id"], kind="mergesort")
    g = frame["g"].to_numpy()
    ids = frame["id"].to_numpy()
    same_group = g[1:] == g[:-1]
    if not same_group.any():
        return 0.0
    return float(np.mean(np.diff(ids)[same_group] == 1))


def audit(policies: pd.DataFrame, claims: pd.DataFrame, *, exposure_cap: float, threshold: float) -> dict:
    counts = policies["ClaimNb"].to_numpy()
    exposure = policies["Exposure"].to_numpy()
    capped_exposure = np.minimum(exposure, exposure_cap)

    groups = profile_groups(policies)
    sizes = np.bincount(groups)
    rows_in_shared = int((sizes[groups] > 1).sum())

    area_density = (
        policies.groupby("Area", observed=True)["Density"].agg(["min", "max"]).astype(int).to_dict("index")
    )

    return {
        "policies": int(len(policies)),
        "claims": int(counts.sum()),
        "policies_with_claims": int((counts > 0).sum()),
        "exposure_years_raw": float(round(exposure.sum(), 2)),
        "exposure_years_capped": float(round(capped_exposure.sum(), 2)),
        "claim_frequency": float(round(counts.sum() / capped_exposure.sum(), 5)),
        "exposure_over_cap": int((exposure > exposure_cap).sum()),
        "exposure_max": float(exposure.max()),
        "exposure_below_one_month": int((exposure < 1 / 12).sum()),
        "claim_count_distribution": {str(int(k)): int(v) for k, v in zip(*np.unique(counts, return_counts=True))},
        "policies_with_more_than_4_claims": int((counts > 4).sum()),
        "claims_on_those_policies": int(counts[counts > 4].sum()),
        "reconciliation": reconcile(policies, claims),
        "severity": severity_profile(claims["ClaimAmount"].to_numpy(), threshold),
        "area_density_ranges": {str(k): {kk: _native(vv) for kk, vv in v.items()} for k, v in area_density.items()},
        "risk_profiles": int(sizes.size),
        "rows_sharing_a_profile": rows_in_shared,
        "share_rows_sharing_a_profile": float(round(rows_in_shared / len(policies), 4)),
        "largest_profile": int(sizes.max()),
        "consecutive_id_share_within_profiles": round(
            consecutive_id_share(policies["IDpol"].to_numpy(), groups), 4
        ),
        "vehicle_age_over_30": int((policies["VehAge"] > 30).sum()),
        "driver_age_over_90": int((policies["DrivAge"] > 90).sum()),
    }
