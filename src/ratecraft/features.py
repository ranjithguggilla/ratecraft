"""Rating factors for the GLMs and raw columns for the boosters.

Nothing here looks at claims. Bands use fixed cut points from ``config`` and
category levels come from the whole policy table, which is fine because the
set of brands or regions an insurer writes is known before any claim is
counted. So a feature frame can be built once for every fold without leaking
anything from the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ratecraft.config import BANDS, CATEGORICAL_BASE, GBM_CATEGORICAL, GBM_NUMERIC, Band


def band(values, spec: Band) -> pd.Categorical:
    """Map values onto left-closed bands, base level first."""
    values = np.asarray(values, dtype=float)
    edges = np.asarray(spec.edges, dtype=float)
    idx = np.searchsorted(edges, values, side="right") - 1
    if (idx < 0).any():
        bad = values[idx < 0][0]
        raise ValueError(f"{spec.source}={bad} is below the first band edge {edges[0]}")
    labels = np.asarray(spec.labels, dtype=object)[idx]
    categories = [spec.base, *[label for label in spec.labels if label != spec.base]]
    return pd.Categorical(labels, categories=categories)


def ordered_levels(spec: Band) -> list[str]:
    """Band labels in their natural order (for tables and charts)."""
    return list(spec.labels)


@dataclass(frozen=True)
class FeatureSpec:
    """The category levels every frame is built with, so that a model fitted
    on one fold and a quote typed into the app see identical encodings."""

    levels: dict[str, tuple[str, ...]]

    @classmethod
    def from_policies(cls, policies: pd.DataFrame) -> FeatureSpec:
        levels = {}
        for name, (column, base) in CATEGORICAL_BASE.items():
            observed = sorted(map(str, pd.unique(policies[column].astype(str))))
            if base not in observed:
                raise ValueError(f"base level {base!r} for {column} not present in the data")
            levels[name] = (base, *[v for v in observed if v != base])
        return cls(levels)

    def _categorical(self, name: str, values) -> pd.Categorical:
        cats = list(self.levels[name])
        values = pd.Series(values, copy=False).astype(str).to_numpy()
        unknown = set(np.unique(values)) - set(cats)
        if unknown:
            raise ValueError(f"unknown {name} level(s): {sorted(unknown)}")
        return pd.Categorical(values, categories=cats)

    def glm_frame(self, policies: pd.DataFrame) -> pd.DataFrame:
        """Banded rating factors plus log density, ready for a formula."""
        frame = pd.DataFrame(index=policies.index)
        for name, spec in BANDS.items():
            frame[name] = band(policies[spec.source].to_numpy(), spec)
        for name, (column, _) in CATEGORICAL_BASE.items():
            frame[name] = self._categorical(name, policies[column])
        frame["log_density"] = np.log(policies["Density"].to_numpy(dtype=float))
        return frame

    def gbm_frame(self, policies: pd.DataFrame) -> pd.DataFrame:
        """Unbanded columns for the boosters; trees find their own cut points."""
        frame = pd.DataFrame(index=policies.index)
        for col in GBM_NUMERIC:
            if col == "log_density":
                frame[col] = np.log(policies["Density"].to_numpy(dtype=float))
            else:
                frame[col] = policies[col].to_numpy(dtype=float)
        name_of = {column: name for name, (column, _) in CATEGORICAL_BASE.items()}
        for col in GBM_CATEGORICAL:
            frame[col] = self._categorical(name_of[col], policies[col])
        return frame
