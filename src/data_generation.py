"""
Synthetic loan-approval data generation.

We build a dataset that mimics real lending dynamics:
- Numerical features drawn from realistic distributions (truncated normals, log-normals).
- Categorical features with realistic prior probabilities.
- An approval label generated from a logistic model whose coefficients reflect
  domain intuition (credit score helps; high DTI and prior defaults hurt; etc.),
  with added Gaussian noise so the label isn't perfectly recoverable.

Why synthetic?
- Reproducible without external downloads.
- Lets us know the ground-truth data-generating process for sanity checks.
- Easy to swap out later: just match the column names in `FEATURES`
  and a real CSV (e.g. UCI German Credit, Lending Club) drops in unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import Generator

# Feature lists — kept here so other modules can import the canonical schema
NUMERICAL_FEATURES = [
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "dti_ratio",
    "age",
    "num_credit_lines",
    "savings_balance",
]
CATEGORICAL_FEATURES = ["loan_purpose", "home_ownership", "education"]
ENGINEERED_FEATURES = ["loan_to_income", "savings_to_loan"]
TARGET = "approved"


def _truncated_normal(rng: Generator, mu: float, sigma: float, lo: float, hi: float, n: int) -> np.ndarray:
    """Draw from N(mu, sigma) truncated to [lo, hi]. Simple rejection sampling, fine at this scale."""
    out = np.empty(n)
    needed = n
    filled = 0
    while needed > 0:
        # over-sample to reduce loop iterations
        draw = rng.normal(mu, sigma, size=int(needed * 1.5) + 16)
        ok = draw[(draw >= lo) & (draw <= hi)]
        take = min(len(ok), needed)
        out[filled:filled + take] = ok[:take]
        filled += take
        needed -= take
    return out


def generate_loan_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic loan-approval dataset.

    Parameters
    ----------
    n : int
        Number of records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns NUMERICAL_FEATURES + CATEGORICAL_FEATURES +
    ENGINEERED_FEATURES + [TARGET].
    """
    rng = np.random.default_rng(seed)

    # ---- Numerical features ----------------------------------------------
    annual_income = np.exp(_truncated_normal(rng, mu=10.9, sigma=0.45, lo=9.0, hi=12.5, n=n))
    annual_income = np.round(annual_income, -2)  # round to nearest $100

    loan_amount = np.exp(_truncated_normal(rng, mu=11.0, sigma=0.7, lo=8.5, hi=13.5, n=n))
    loan_amount = np.round(loan_amount, -2)

    credit_score = _truncated_normal(rng, mu=685, sigma=85, lo=300, hi=850, n=n).astype(int)
    employment_years = np.clip(rng.gamma(shape=2.0, scale=4.0, size=n), 0, 40).astype(int)
    age = _truncated_normal(rng, mu=38, sigma=11, lo=18, hi=75, n=n).astype(int)
    num_credit_lines = np.clip(rng.poisson(lam=4.5, size=n), 0, 25)
    savings_balance = np.exp(_truncated_normal(rng, mu=8.5, sigma=1.4, lo=4, hi=13, n=n))
    savings_balance = np.round(savings_balance, -1)

    # DTI is correlated with loan/income — generate it that way rather than independently
    base_dti = (loan_amount / annual_income) * 18 + rng.normal(8, 4, n)
    dti_ratio = np.clip(base_dti, 1, 65).round(1)

    # ---- Categorical features --------------------------------------------
    loan_purpose = rng.choice(
        ["debt_consolidation", "home_improvement", "major_purchase", "medical", "education", "other"],
        size=n,
        p=[0.40, 0.18, 0.15, 0.08, 0.09, 0.10],
    )
    home_ownership = rng.choice(
        ["own", "mortgage", "rent", "other"],
        size=n,
        p=[0.18, 0.42, 0.36, 0.04],
    )
    education = rng.choice(
        ["high_school", "some_college", "bachelors", "graduate"],
        size=n,
        p=[0.28, 0.30, 0.30, 0.12],
    )

    # ---- Engineered features ---------------------------------------------
    loan_to_income = (loan_amount / annual_income).round(3)
    savings_to_loan = (savings_balance / loan_amount).round(3)

    # ---- Approval label --------------------------------------------------
    # Logistic model with hand-chosen coefficients. These reflect typical
    # credit-policy intuitions: credit score and DTI dominate, supported by
    # income, employment, and home ownership. Noise added at the end so the
    # problem is not trivially separable (otherwise every model gets ~99% AUC
    # and the comparison becomes meaningless).
    z = (
        0.50                                        # intercept → ~60% baseline approval
        + 0.012 * (credit_score - 685)              # strong: +1 SD credit ≈ +1.0 logit
        + 0.000015 * (annual_income - 60_000)       # mild positive effect of income
        - 0.060 * (dti_ratio - 28)                  # DTI is a strong negative signal
        - 0.0000040 * (loan_amount - 60_000)        # larger loans slightly harder to approve
        + 0.060 * employment_years                  # tenure as stability proxy
        + 0.0000020 * savings_balance               # savings cushion
        - 1.50 * (loan_to_income > 4).astype(int)   # hard cliff for unaffordable loans
        + 0.30 * (home_ownership == "own").astype(int)
        + 0.15 * (home_ownership == "mortgage").astype(int)
        + 0.25 * (education == "graduate").astype(int)
        + 0.08 * (education == "bachelors").astype(int)
        - 0.35 * (loan_purpose == "medical").astype(int)
        - 0.15 * (loan_purpose == "debt_consolidation").astype(int)
    )
    # Add label noise so the problem isn't trivially separable
    z = z + rng.normal(0, 0.4, n)

    p_approve = 1.0 / (1.0 + np.exp(-z))
    approved = (rng.uniform(0, 1, n) < p_approve).astype(int)

    df = pd.DataFrame({
        "annual_income": annual_income.astype(int),
        "loan_amount": loan_amount.astype(int),
        "credit_score": credit_score,
        "employment_years": employment_years,
        "dti_ratio": dti_ratio,
        "age": age,
        "num_credit_lines": num_credit_lines,
        "savings_balance": savings_balance.astype(int),
        "loan_purpose": loan_purpose,
        "home_ownership": home_ownership,
        "education": education,
        "loan_to_income": loan_to_income,
        "savings_to_loan": savings_to_loan,
        "approved": approved,
    })

    return df


if __name__ == "__main__":
    df = generate_loan_data(n=10_000, seed=42)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Approval rate: {df['approved'].mean():.1%}")
    print(f"\nClass balance:\n{df['approved'].value_counts()}")
