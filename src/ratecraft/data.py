"""Fetch, verify and tidy the French motor third-party liability tables.

The source is freMTPL2freq (one row per policy) and freMTPL2sev (one row per
claim) from the CASdatasets R package, pinned to one commit and checked
against SHA-256 digests. The files are R serialisations, read here with the
pure-Python ``rdata`` parser so nothing needs an R install.

The data is GPL-licensed and is never committed to this repository; it is
downloaded on first use into ``data/raw`` and cached as parquet in
``data/interim``.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ratecraft.config import SOURCE, DataSource

POLICY_COLUMNS = (
    "IDpol", "ClaimNb", "Exposure", "VehPower", "VehAge", "DrivAge",
    "BonusMalus", "VehBrand", "VehGas", "Area", "Density", "Region",
)
INTEGER_COLUMNS = ("ClaimNb", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density")
CATEGORY_COLUMNS = ("VehBrand", "VehGas", "Area", "Region")


class DataIntegrityError(RuntimeError):
    """A downloaded file does not match its pinned digest."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_verified(path: Path, expected: str) -> bool:
    return path.exists() and sha256(path) == expected


def _fetch_with_git(dest: Path, source: DataSource, names: list[str]) -> None:
    """Sparse, blobless fetch of just the files we need at the pinned commit."""
    with tempfile.TemporaryDirectory() as tmp:
        def git(*args: str) -> None:
            subprocess.run(["git", "-C", tmp, *args], check=True, capture_output=True, timeout=600)

        git("init", "-q")
        git("remote", "add", "origin", source.repo)
        git("sparse-checkout", "set", "--no-cone", *[f"{source.subdir}/{n}" for n in names])
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", source.commit)
        git("checkout", "-q", "FETCH_HEAD")
        for name in names:
            shutil.copyfile(Path(tmp) / source.subdir / name, dest / name)


def _fetch_with_http(dest: Path, source: DataSource, names: list[str]) -> None:
    for name in names:
        with urllib.request.urlopen(source.raw_url(name), timeout=300) as resp, open(dest / name, "wb") as out:
            shutil.copyfileobj(resp, out)


def fetch(raw_dir: Path, source: DataSource = SOURCE) -> dict[str, Path]:
    """Make sure both source files are present and match their digests.

    Tries git first (works anywhere github.com is reachable), then plain HTTPS
    against raw.githubusercontent.com. Either way the digest decides.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n, digest in source.files.items() if not _is_verified(raw_dir / n, digest)]
    if missing:
        try:
            _fetch_with_git(raw_dir, source, missing)
        except (OSError, subprocess.SubprocessError):
            _fetch_with_http(raw_dir, source, missing)
    for name, digest in source.files.items():
        actual = sha256(raw_dir / name)
        if actual != digest:
            raise DataIntegrityError(f"{name}: expected sha256 {digest}, got {actual}")
    return {name: raw_dir / name for name in source.files}


def read_rda(path: Path) -> pd.DataFrame:
    """Read the single data frame stored in an .rda file."""
    import rdata  # imported lazily: only needed when parsing the raw files

    converted = rdata.conversion.convert(rdata.parser.parse_file(path))
    if len(converted) != 1:
        raise ValueError(f"{path.name}: expected one object, found {list(converted)}")
    return next(iter(converted.values()))


def parse_policy_ids(values: pd.Series) -> np.ndarray:
    """Policy ids arrive as an R factor of strings, and R wrote at least one of
    them in scientific notation ("1e+05"). Parse through float, then insist the
    result is integral."""
    as_float = pd.to_numeric(pd.Series(values).astype(str), errors="raise").to_numpy(dtype=float)
    as_int = as_float.astype(np.int64)
    if not np.array_equal(as_int, as_float):
        raise ValueError("non-integral policy id")
    return as_int


def normalise_policies(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in POLICY_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"policy table is missing {missing}")
    out = pd.DataFrame(index=pd.RangeIndex(len(raw)))
    out["IDpol"] = parse_policy_ids(raw["IDpol"])
    for col in INTEGER_COLUMNS:
        values = pd.to_numeric(raw[col]).to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError(f"{col} has missing values")
        out[col] = values.astype(np.int64)
    out["Exposure"] = raw["Exposure"].to_numpy(dtype=float)
    # .to_numpy() everywhere: the R row names become a 1-based index on the
    # raw frame, and assigning a Series would silently align on it.
    for col in CATEGORY_COLUMNS:
        out[col] = pd.Categorical(raw[col].astype(str).to_numpy())
    if out["IDpol"].duplicated().any():
        raise ValueError("duplicate policy ids")
    return out[list(POLICY_COLUMNS)]


def normalise_claims(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IDpol": parse_policy_ids(raw["IDpol"]),
            "ClaimAmount": raw["ClaimAmount"].to_numpy(dtype=float),
        }
    )


def load(raw_dir: Path, interim_dir: Path, *, fetch_if_missing: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (policies, claims), parsing the raw files once and caching."""
    policies_cache = interim_dir / "policies.parquet"
    claims_cache = interim_dir / "claims.parquet"
    if policies_cache.exists() and claims_cache.exists():
        policies = pd.read_parquet(policies_cache)
        for col in CATEGORY_COLUMNS:
            policies[col] = policies[col].astype("category")
        return policies, pd.read_parquet(claims_cache)

    if fetch_if_missing:
        fetch(raw_dir)
    policies = normalise_policies(read_rda(raw_dir / "freMTPL2freq.rda"))
    claims = normalise_claims(read_rda(raw_dir / "freMTPL2sev.rda"))
    interim_dir.mkdir(parents=True, exist_ok=True)
    policies.to_parquet(policies_cache, index=False)
    claims.to_parquet(claims_cache, index=False)
    return policies, claims


def attach_losses(
    policies: pd.DataFrame,
    claims: pd.DataFrame,
    threshold: float | None,
    exposure_cap: float,
) -> pd.DataFrame:
    """Policy table plus capped exposure and per-policy loss totals.

    ``loss`` is the full claim cost, ``loss_capped`` caps each individual
    claim at ``threshold`` (no cap when it is None), and ``n_claims`` counts
    the claim rows. Policies without claims get zeros.
    """
    amounts = claims["ClaimAmount"].to_numpy()
    capped = amounts if threshold is None else np.minimum(amounts, threshold)
    per_policy = (
        pd.DataFrame({"IDpol": claims["IDpol"].to_numpy(), "loss": amounts, "loss_capped": capped})
        .groupby("IDpol", sort=False)
        .agg(n_claims=("loss", "size"), loss=("loss", "sum"), loss_capped=("loss_capped", "sum"))
    )
    out = policies.copy()
    out["exposure"] = np.minimum(out["Exposure"].to_numpy(), exposure_cap)
    joined = per_policy.reindex(out["IDpol"].to_numpy())
    out["n_claims"] = joined["n_claims"].fillna(0).to_numpy(dtype=np.int64)
    out["loss"] = joined["loss"].fillna(0.0).to_numpy()
    out["loss_capped"] = joined["loss_capped"].fillna(0.0).to_numpy()
    return out
