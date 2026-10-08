"""Large losses: cap them for modelling, charge them back as a flat loading.

One claim in this table is worth more than 3,400 median claims. Left in, a
handful of bodily-injury claims decide the severity relativities for whatever
cells they happen to land in. The usual fix is to model the attritional layer
(every claim capped at a threshold) and spread the excess over the whole book
in proportion to attritional cost. The loading is fitted on training claims
only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LargeLossLoading:
    threshold: float | None
    loading: float

    @classmethod
    def fit(cls, amounts, threshold: float | None) -> LargeLossLoading:
        amounts = np.asarray(amounts, dtype=float)
        if threshold is None:
            return cls(None, 0.0)
        capped = np.minimum(amounts, threshold)
        excess = amounts - capped
        return cls(float(threshold), float(excess.sum() / capped.sum()))

    def cap(self, amounts) -> np.ndarray:
        amounts = np.asarray(amounts, dtype=float)
        return amounts if self.threshold is None else np.minimum(amounts, self.threshold)

    def apply(self, attritional) -> np.ndarray:
        """Attritional cost to expected total cost."""
        return np.asarray(attritional, dtype=float) * (1.0 + self.loading)
