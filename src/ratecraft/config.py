"""Everything more than one module has to agree on.

Paths are derived from a repository root that the scripts pass in, so the
package works the same from an editable install, a test's temporary directory,
or CI. Every number in here was fixed before the test fold was scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Seeds the split, the boosters and the synthetic generator.
SEED = 7


@dataclass(frozen=True)
class DataSource:
    """Where the policy and claim tables come from, pinned to one commit."""

    repo: str = "https://github.com/dutangc/CASdatasets"
    commit: str = "227fb56b8734bdb7c0327a41180e01d2ddaeaf26"
    subdir: str = "data"
    files: dict[str, str] = field(
        default_factory=lambda: {
            "freMTPL2freq.rda": "82c8598513d9fa78226b6c7d271a9940eca4b8083e9e32320a75d377bdfe15a3",
            "freMTPL2sev.rda": "78e6e5016ae046603b37e88ff8ad8ca327f5d59f827c1f75d140388de09b14b3",
        }
    )

    def raw_url(self, name: str) -> str:
        owner_repo = self.repo.removeprefix("https://github.com/")
        return f"https://raw.githubusercontent.com/{owner_repo}/{self.commit}/{self.subdir}/{name}"


SOURCE = DataSource()


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    def ensure(self) -> Paths:
        for p in (self.raw, self.interim, self.results, self.assets):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --- data treatment -------------------------------------------------------

#: Exposure is in policy-years. 1,224 rows run past one year, which a single
#: annual contract cannot do, so exposure is capped at one.
EXPOSURE_CAP = 1.0

#: Claims above this are treated as large losses: capped for modelling, and the
#: excess is charged back as a flat loading fitted on the training fold only.
LARGE_LOSS_THRESHOLD = 50_000.0

#: Thresholds for the sensitivity table. None means no cap at all.
LARGE_LOSS_SENSITIVITY: tuple[float | None, ...] = (25_000.0, 50_000.0, 100_000.0, None)

#: train / validation / test, by risk profile, never by row.
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)


# --- rating factor bands --------------------------------------------------
# Cut points are domain choices made up front, not fitted, so banding cannot
# leak anything from the target. The first label in each list is the base
# level the relativities are quoted against.

@dataclass(frozen=True)
class Band:
    source: str
    edges: tuple[float, ...]  # left-closed edges; the last band is open ended
    labels: tuple[str, ...]
    base: str

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.labels):
            raise ValueError(f"{self.source}: one label per left edge")
        if self.base not in self.labels:
            raise ValueError(f"{self.source}: base level {self.base!r} is not a label")


BANDS: dict[str, Band] = {
    "drv_age": Band(
        "DrivAge",
        (18, 21, 26, 31, 41, 51, 61, 71, 81),
        ("18-20", "21-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81+"),
        base="41-50",
    ),
    "veh_age": Band(
        "VehAge",
        (0, 1, 2, 5, 10, 15, 20),
        ("0", "1", "2-4", "5-9", "10-14", "15-19", "20+"),
        base="5-9",
    ),
    "veh_power": Band(
        "VehPower",
        (4, 5, 6, 7, 8, 9, 10, 11, 12),
        ("4", "5", "6", "7", "8", "9", "10", "11", "12+"),
        base="6",
    ),
    "bonus_malus": Band(
        "BonusMalus",
        (50, 51, 58, 65, 80, 100, 126),
        ("50", "51-57", "58-64", "65-79", "80-99", "100-125", "126+"),
        base="50",
    ),
    "density_band": Band(
        "Density",
        (1, 51, 101, 501, 2001, 10001),
        ("1-50", "51-100", "101-500", "501-2000", "2001-10000", "10001+"),
        base="101-500",
    ),
}

#: Categorical rating factors taken as they are, with their base level.
CATEGORICAL_BASE: dict[str, tuple[str, str]] = {
    "veh_brand": ("VehBrand", "B12"),
    "veh_gas": ("VehGas", "Regular"),
    "region": ("Region", "Centre"),
}

#: Main effects in the tariff. density_band exists for interaction terms;
#: the main effect uses log density, which is smoother than six steps.
GLM_FACTORS = ("drv_age", "veh_age", "veh_power", "bonus_malus", "veh_brand", "veh_gas", "region")
GLM_NUMERIC = ("log_density",)

#: Factors the interaction search may pair up.
INTERACTION_CANDIDATES = (
    "drv_age", "veh_age", "veh_power", "bonus_malus", "veh_gas", "veh_brand", "density_band", "region",
)

#: Raw columns the boosters see. Area is left out on purpose: it is a coarsened
#: copy of Density (see the audit), so it adds nothing a tree cannot split on.
GBM_NUMERIC = ("DrivAge", "VehAge", "VehPower", "BonusMalus", "log_density")
GBM_CATEGORICAL = ("VehBrand", "VehGas", "Region")


# --- model selection --------------------------------------------------------

#: Every pure-premium model is scored with the same Tweedie deviance, picked
#: before any model was fitted. Any power gives a strictly consistent score
#: for the mean, so this is a fair yardstick for all of them.
EVAL_POWER = 1.5

#: A more complex model has to buy at least this much extra deviance
#: improvement over the flat rate, relative to the simpler model, on the
#: validation fold. Fixed in advance.
SHIP_MARGIN = 0.05

GLM_ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3)
#: Severity has a few thousand claims per fold against 60 coefficients, so its
#: ridge grid reaches much further towards "every policy gets the mean".
SEVERITY_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TWEEDIE_POWERS = (1.2, 1.35, 1.5, 1.65, 1.8)
INTERACTION_MAX_TERMS = 3
#: An interaction must cut validation deviance by at least this share of the
#: main-effects GLM's own improvement over the flat rate to be kept.
INTERACTION_MIN_GAIN = 0.01

GBM_GRID = tuple(
    {"num_leaves": leaves, "min_data_in_leaf": min_leaf}
    for leaves in (15, 31, 63)
    for min_leaf in (200, 1000)
)
GBM_BASE_PARAMS = {
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 4,
    "deterministic": True,
    "force_row_wise": True,
}
GBM_MAX_ROUNDS = 3000
GBM_EARLY_STOPPING = 100


# --- business assumptions ---------------------------------------------------

@dataclass(frozen=True)
class CostModel:
    """The assumptions behind every euro figure. Change one and rerun."""

    #: Share of the office premium expected to go on claims.
    target_loss_ratio: float = 0.70
    #: Acquisition, admin and claims handling, as a share of premium.
    expense_ratio: float = 0.25
    #: Logit demand exponent. At 4, a quote 10% dearer than the rival's wins
    #: 41% of customers instead of 50%. Only used in the softer market scenario.
    price_elasticity: float = 4.0
    #: Results are scaled to a book of this many earned policy-years.
    book_policy_years: float = 100_000.0


COSTS = CostModel()
