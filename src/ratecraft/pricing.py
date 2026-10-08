"""From fitted components to a price, and from a price to a reason.

A pure-premium model is either frequency x severity or a single direct
model, built from GLM or booster components. ``quote`` turns one customer
profile into an office premium plus the handful of factors that moved it most,
in words.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ratecraft.config import COSTS, CostModel
from ratecraft.features import FeatureSpec
from ratecraft.gbm import GBMComponent
from ratecraft.glm import GLMComponent
from ratecraft.largeloss import LargeLossLoading

#: Human wording for every factor, keyed by the raw column it comes from.
LABELS = {
    "DrivAge": "Driver age",
    "VehAge": "Vehicle age",
    "VehPower": "Vehicle power band",
    "BonusMalus": "Bonus-malus level",
    "VehBrand": "Vehicle brand",
    "VehGas": "Fuel",
    "Density": "Population density",
    "Region": "Region",
}

#: GLM factor or booster column -> raw column it is derived from.
SOURCE_OF = {
    "drv_age": "DrivAge", "veh_age": "VehAge", "veh_power": "VehPower", "bonus_malus": "BonusMalus",
    "veh_brand": "VehBrand", "veh_gas": "VehGas", "region": "Region", "log_density": "Density",
    "density_band": "Density",
    "DrivAge": "DrivAge", "VehAge": "VehAge", "VehPower": "VehPower", "BonusMalus": "BonusMalus",
    "VehBrand": "VehBrand", "VehGas": "VehGas", "Region": "Region",
}

Component = GLMComponent | GBMComponent


class Frames:
    """Builds each kind of feature frame once per batch of policies.

    Frames that were already built (the pipeline builds them once for the
    whole table) can be handed in to skip the work."""

    def __init__(self, spec: FeatureSpec, policies: pd.DataFrame,
                 glm: pd.DataFrame | None = None, gbm: pd.DataFrame | None = None):
        self.spec = spec
        self.policies = policies
        self._glm = glm
        self._gbm = gbm

    def subset(self, idx: np.ndarray) -> Frames:
        return Frames(
            self.spec,
            self.policies.iloc[idx],
            None if self._glm is None else self._glm.iloc[idx],
            None if self._gbm is None else self._gbm.iloc[idx],
        )

    @property
    def glm(self) -> pd.DataFrame:
        if self._glm is None:
            self._glm = self.spec.glm_frame(self.policies)
        return self._glm

    @property
    def gbm(self) -> pd.DataFrame:
        if self._gbm is None:
            self._gbm = self.spec.gbm_frame(self.policies)
        return self._gbm

    def for_component(self, component: Component) -> pd.DataFrame:
        return self.glm if isinstance(component, GLMComponent) else self.gbm


def _contributions(component: Component, frames: Frames) -> pd.DataFrame:
    """Log-scale contributions against an average policy, keyed by raw column."""
    frame = frames.for_component(component)
    contrib = component.contributions(frame)
    if isinstance(component, GLMComponent):
        contrib = contrib - pd.Series(component.reference)
    out = pd.DataFrame(index=frame.index)
    for column in contrib.columns:
        source = SOURCE_OF[column]
        out[source] = out.get(source, 0.0) + contrib[column]
    return out


@dataclass
class PurePremiumModel:
    """Expected attritional claims cost per policy-year."""

    name: str
    kind: str  # "flat", "freq_sev" or "direct"
    frequency: Component | None = None
    severity: Component | None = None
    direct: Component | None = None
    flat_rate: float | None = None
    notes: dict = field(default_factory=dict)

    def frequency_rate(self, frames: Frames) -> np.ndarray | None:
        if self.frequency is None:
            return None
        return self.frequency.predict(frames.for_component(self.frequency))

    def severity_mean(self, frames: Frames) -> np.ndarray | None:
        if self.severity is None:
            return None
        return self.severity.predict(frames.for_component(self.severity))

    def predict(self, frames: Frames) -> np.ndarray:
        if self.kind == "flat":
            return np.full(len(frames.policies), float(self.flat_rate))
        if self.kind == "freq_sev":
            return self.frequency_rate(frames) * self.severity_mean(frames)
        return self.direct.predict(frames.for_component(self.direct))

    def contributions(self, frames: Frames) -> pd.DataFrame:
        """Per-factor log effects against an average policy, summed over components."""
        parts = [c for c in (self.frequency, self.severity, self.direct) if c is not None]
        if not parts:
            return pd.DataFrame(index=frames.policies.index)
        total = _contributions(parts[0], frames)
        for comp in parts[1:]:
            total = total.add(_contributions(comp, frames), fill_value=0.0)
        return total


@dataclass(frozen=True)
class Reason:
    factor: str
    value: str
    effect: float  # multiplicative, 1.25 means 25% dearer than an average policy

    def sentence(self) -> str:
        pct = (self.effect - 1.0) * 100
        direction = "dearer" if pct >= 0 else "cheaper"
        return f"{self.factor} {self.value}: {abs(pct):.0f}% {direction} than an average policy"


@dataclass(frozen=True)
class Quote:
    frequency: float | None
    severity: float | None
    attritional: float
    large_loss_allowance: float
    technical: float
    office: float
    reasons: tuple[Reason, ...]


@dataclass
class Pricer:
    """What the app loads: the shipped model, the loading and the cost model,
    plus the best plain GLM so a quote can be shown next to a filed tariff."""

    model: PurePremiumModel
    spec: FeatureSpec
    loading: LargeLossLoading
    costs: CostModel = COSTS
    reference: PurePremiumModel | None = None
    metadata: dict = field(default_factory=dict)

    def technical(self, policies: pd.DataFrame) -> np.ndarray:
        """Expected total claims cost per policy-year, large losses included."""
        return self.loading.apply(self.model.predict(Frames(self.spec, policies)))

    def office(self, policies: pd.DataFrame) -> np.ndarray:
        return self.technical(policies) / self.costs.target_loss_ratio


PROFILE_FIELDS = {
    "DrivAge": (18, 100),
    "VehAge": (0, 100),
    "VehPower": (4, 15),
    "BonusMalus": (50, 230),
    "Density": (1, 27_000),
}


def profile_frame(profile: dict) -> pd.DataFrame:
    """One customer as a one-row policy table, with ranges checked."""
    for name, (low, high) in PROFILE_FIELDS.items():
        value = profile.get(name)
        if value is None:
            raise ValueError(f"missing {name}")
        if not (low <= float(value) <= high):
            raise ValueError(f"{name}={value} is outside {low}-{high}")
    row = {name: int(profile[name]) for name in PROFILE_FIELDS}
    for name in ("VehBrand", "VehGas", "Region"):
        if not profile.get(name):
            raise ValueError(f"missing {name}")
        row[name] = str(profile[name])
    return pd.DataFrame([row])


def _display(source: str, row: pd.Series) -> str:
    value = row[source]
    if source == "Density":
        return f"{int(value):,} people per km2"
    if source == "BonusMalus":
        return str(int(value))
    if source in ("DrivAge", "VehAge"):
        return f"{int(value)}"
    return str(value)


def quote(pricer: Pricer, profile: dict, *, top: int = 4, use_reference: bool = False) -> Quote:
    """Price one customer and say which factors moved the price most."""
    policies = profile_frame(profile)
    frames = Frames(pricer.spec, policies)
    model = pricer.reference if use_reference else pricer.model
    if model is None:
        raise ValueError("this pricer has no reference model")
    attritional = float(model.predict(frames)[0])
    technical = float(pricer.loading.apply(attritional))
    freq = model.frequency_rate(frames)
    sev = model.severity_mean(frames)
    contrib = model.contributions(frames).iloc[0]
    ranked = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top]
    reasons = tuple(
        Reason(LABELS[source], _display(source, policies.iloc[0]), float(np.exp(value)))
        for source, value in ranked.items()
    )
    return Quote(
        frequency=None if freq is None else float(freq[0]),
        severity=None if sev is None else float(sev[0]),
        attritional=attritional,
        large_loss_allowance=technical - attritional,
        technical=technical,
        office=technical / pricer.costs.target_loss_ratio,
        reasons=reasons,
    )
