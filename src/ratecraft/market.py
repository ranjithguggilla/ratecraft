"""Two insurers, one market, the held-out policies as customers.

Each insurer quotes every customer its own technical price grossed up to the
same target loss ratio, and customers choose. In the hard scenario everyone
takes the cheaper quote; in the soft one a logit demand curve splits each
customer between the two by price ratio. Each insurer then pays the actual
claims of the customers it wrote. This is the mechanism behind adverse
selection: the insurer with the cruder tariff wins the risks it underprices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ratecraft.config import CostModel


@dataclass(frozen=True)
class Book:
    name: str
    share_of_customers: float
    earned_years: float
    premium: float
    claims: float
    expenses: float

    @property
    def loss_ratio(self) -> float:
        return self.claims / self.premium if self.premium else float("nan")

    @property
    def combined_ratio(self) -> float:
        return (self.claims + self.expenses) / self.premium if self.premium else float("nan")

    @property
    def result(self) -> float:
        return self.premium - self.claims - self.expenses


def challenger_share(price_incumbent, price_challenger, elasticity: float | None) -> np.ndarray:
    """Probability each customer buys from the challenger.

    ``elasticity=None`` means every customer takes the cheaper quote (ties are
    split evenly). Otherwise share = 1 / (1 + (p_challenger / p_incumbent)^e)."""
    a = np.asarray(price_incumbent, float)
    b = np.asarray(price_challenger, float)
    if elasticity is None:
        return np.where(b < a, 1.0, np.where(b > a, 0.0, 0.5))
    return 1.0 / (1.0 + np.power(b / a, elasticity))


def simulate(
    names: tuple[str, str],
    technical: tuple[np.ndarray, np.ndarray],
    actual_claims: np.ndarray,
    exposure: np.ndarray,
    costs: CostModel,
    elasticity: float | None,
) -> list[Book]:
    """Run the market once. Premiums are annual; earned premium is premium times
    the exposure actually observed, which is what the claims were incurred over."""
    office = [t / costs.target_loss_ratio for t in technical]
    share_b = challenger_share(office[0], office[1], elasticity)
    shares = (1.0 - share_b, share_b)
    books = []
    for name, prem, share in zip(names, office, shares):
        earned = np.sum(share * prem * exposure)
        books.append(
            Book(
                name=name,
                share_of_customers=float(share.mean()),
                earned_years=float(np.sum(share * exposure)),
                premium=float(earned),
                claims=float(np.sum(share * actual_claims)),
                expenses=float(costs.expense_ratio * earned),
            )
        )
    return books


def to_frame(books: list[Book], scenario: str, costs: CostModel) -> pd.DataFrame:
    rows = []
    for b in books:
        per_book = costs.book_policy_years / b.earned_years if b.earned_years else float("nan")
        rows.append(
            {
                "scenario": scenario,
                "insurer": b.name,
                "share_of_customers": b.share_of_customers,
                "earned_policy_years": b.earned_years,
                "premium": b.premium,
                "claims": b.claims,
                "loss_ratio": b.loss_ratio,
                "combined_ratio": b.combined_ratio,
                "underwriting_result": b.result,
                "result_per_book": b.result * per_book,
            }
        )
    return pd.DataFrame(rows)
