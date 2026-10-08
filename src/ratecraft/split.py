"""Train / validation / test splits that respect repeated risk profiles.

38% of the rows in freMTPL2freq share every rating factor with at least one
other row, and 41% of neighbouring rows inside such a profile have
consecutive policy ids. That looks like one customer's contract split across
several records far more often than chance would allow. A
row-level random split puts those siblings on both sides of the line and lets
a flexible model grade itself on people it has already seen, so the split here
is by profile: every row with the same rating factors lands in the same fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_COLUMNS = (
    "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region",
)


def profile_groups(policies: pd.DataFrame) -> np.ndarray:
    """Integer id per distinct combination of rating factors, 0..k-1."""
    return (
        policies.groupby(list(RISK_COLUMNS), observed=True, sort=False)
        .ngroup()
        .to_numpy(dtype=np.int64)
    )


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": int(self.train.size), "valid": int(self.valid.size), "test": int(self.test.size)}


def _counts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to one")
    n_train = int(round(fractions[0] * n))
    n_valid = int(round(fractions[1] * n))
    return n_train, n_valid


def grouped_split(groups: np.ndarray, fractions: tuple[float, float, float], seed: int) -> Split:
    """Assign whole groups to folds at random; no group straddles two folds."""
    unique, inverse = np.unique(groups, return_inverse=True)
    order = np.random.default_rng(seed).permutation(unique.size)
    n_train, n_valid = _counts(unique.size, fractions)
    fold_of_group = np.full(unique.size, 2, dtype=np.int8)
    fold_of_group[order[:n_train]] = 0
    fold_of_group[order[n_train : n_train + n_valid]] = 1
    fold = fold_of_group[inverse]
    return Split(np.flatnonzero(fold == 0), np.flatnonzero(fold == 1), np.flatnonzero(fold == 2))


def random_split(n: int, fractions: tuple[float, float, float], seed: int) -> Split:
    """Plain row-level split, kept only to measure how optimistic it is."""
    order = np.random.default_rng(seed).permutation(n)
    n_train, n_valid = _counts(n, fractions)
    return Split(
        np.sort(order[:n_train]),
        np.sort(order[n_train : n_train + n_valid]),
        np.sort(order[n_train + n_valid :]),
    )
