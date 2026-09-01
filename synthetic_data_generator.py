"""
Synthetic SME Card Churn Dataset Generator
============================================
Generates a realistic 100,000-row dataset simulating small/medium business
(SME) credit card accounts for a card churn prediction and retention
analytics project.

WHY SYNTHETIC: no public dataset combines SME-specific fields (business
type, owner-vs-employee cardholder, industry) with churn + retention-offer
experiment data. This script fabricates that combination with deliberately
realistic, documented causal relationships, so downstream statistical
tests (chi-square, t-test, logistic regression, propensity matching) have
genuine signal to recover -- not just noise.

"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
N = 100_000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_customers(n=N):
    ids = [f"SME{100000 + i}" for i in range(n)]

    # ---- Demographics ----
    business_size = RNG.choice(
        ["micro", "small", "medium"], size=n, p=[0.55, 0.35, 0.10]
    )
    industry = RNG.choice(
        ["retail", "food_service", "professional_services",
         "construction", "healthcare", "tech"],
        size=n, p=[0.22, 0.20, 0.22, 0.14, 0.12, 0.10]
    )
    # Business owners hold the card directly; employees hold company-issued cards
    cardholder_role = RNG.choice(
        ["business_owner", "employee"], size=n, p=[0.4, 0.6]
    )
    business_age_years = np.clip(RNG.exponential(scale=6, size=n), 0.5, 40).round(1)
    tenure_months = np.clip(RNG.exponential(scale=24, size=n), 1, 180).round(0)

    # ---- Card usage / spend ----
    credit_limit = np.clip(RNG.normal(15000, 8000, n), 1000, 100000).round(-2)

    # spend trend: most flat, some declining (early churn signal), some growing
    spend_trend_3mo = RNG.normal(0.0, 0.12, n)  # fractional change, e.g. -0.2 = -20%

    base_spend = np.clip(RNG.normal(4500, 3000, n), 100, None)
    # inject a handful of large-business outliers
    outlier_mask = RNG.random(n) < 0.015
    base_spend[outlier_mask] *= RNG.uniform(4, 10, outlier_mask.sum())
    avg_monthly_spend = np.clip(base_spend * (1 + spend_trend_3mo), 50, None).round(2)

    transaction_count_monthly = np.clip(
        (avg_monthly_spend / RNG.uniform(80, 200, n)), 1, None
    ).round(0)

    utilization_ratio = np.clip(RNG.beta(2, 4, n), 0.01, 0.99).round(3)
    merchant_category_diversity = RNG.integers(1, 15, n)

    # ---- Credit risk ----
    credit_score_band = RNG.choice(
        ["subprime", "near_prime", "prime", "super_prime"],
        size=n, p=[0.12, 0.23, 0.40, 0.25]
    )
    band_risk = {"subprime": 3, "near_prime": 2, "prime": 1, "super_prime": 0}
    risk_level = np.array([band_risk[b] for b in credit_score_band])

    late_payments_last_12mo = RNG.poisson(lam=0.3 + 0.6 * risk_level, size=n)
    late_payments_last_12mo = np.clip(late_payments_last_12mo, 0, 12)

    payment_ratio = np.clip(
        RNG.normal(0.85 - 0.08 * risk_level, 0.15, n), 0.05, 1.2
    ).round(2)

    # ---- Engagement ----
    support_calls_last_90d = RNG.poisson(lam=0.8 + 0.5 * risk_level, size=n)
    login_frequency_monthly = np.clip(RNG.normal(8, 5, n), 0, None).round(0)

    # inject missingness to mimic real-world data pipeline gaps
    missing_support = RNG.random(n) < 0.03
    missing_login = RNG.random(n) < 0.04
    support_calls_last_90d = support_calls_last_90d.astype(float)
    login_frequency_monthly = login_frequency_monthly.astype(float)
    support_calls_last_90d[missing_support] = np.nan
    login_frequency_monthly[missing_login] = np.nan

    df = pd.DataFrame({
        "customer_id": ids,
        "business_size": business_size,
        "industry": industry,
        "cardholder_role": cardholder_role,
        "business_age_years": business_age_years,
        "tenure_months": tenure_months,
        "credit_limit": credit_limit,
        "avg_monthly_spend": avg_monthly_spend,
        "spend_trend_3mo": spend_trend_3mo.round(3),
        "transaction_count_monthly": transaction_count_monthly,
        "utilization_ratio": utilization_ratio,
        "merchant_category_diversity": merchant_category_diversity,
        "credit_score_band": credit_score_band,
        "late_payments_last_12mo": late_payments_last_12mo,
        "payment_ratio": payment_ratio,
        "support_calls_last_90d": support_calls_last_90d,
        "login_frequency_monthly": login_frequency_monthly,
    })


    # CHURN PROBABILITY MODEL -- the causal core of the dataset

    is_owner = (df["cardholder_role"] == "business_owner").astype(int)
    declining_spend = np.clip(-df["spend_trend_3mo"], 0, None)  # only negative trend hurts
    # U-shaped utilization risk: stress at high util, disengagement at very low util
    util_risk = np.abs(df["utilization_ratio"] - 0.35) * 1.2
    support_calls_norm = df["support_calls_last_90d"].fillna(0) / 5
    tenure_norm = np.log1p(df["tenure_months"]) / np.log1p(180)
    late_pay_norm = df["late_payments_last_12mo"] / 12

    churn_logit = (
        -1.6
        + 0.85 * is_owner
        + 2.2 * declining_spend
        + 0.9 * util_risk
        + 0.45 * support_calls_norm
        + 0.55 * late_pay_norm
        - 1.3 * tenure_norm
        + RNG.normal(0, 0.35, n)  # noise
    )
    churn_prob = sigmoid(churn_logit)
    df["churn_prob_true"] = churn_prob.round(4)  # kept for validation/teaching; drop before modeling
    df["churn_flag"] = (RNG.random(n) < churn_prob).astype(int)

    return df


def generate_offer_experiment(df, n_targeted=15000):
    """
    Simulates a retention-offer rollout targeted at higher-risk customers,
    with NON-random assignment (biased toward high-risk) so that propensity
    score matching is actually necessary to recover the true treatment effect.
    """
    rng = RNG

    # Eligible pool: skew toward higher predicted risk, mimic real targeting rules
    weights = df["churn_prob_true"] ** 1.5
    weights = weights / weights.sum()
    targeted_idx = rng.choice(df.index, size=n_targeted, replace=False, p=weights)
    exp_df = df.loc[targeted_idx].copy()

    # Assign offer type; ~50% get an offer, 50% act as control -- but assignment
    # probability still correlates with risk score (non-random / confounded)
    offer_prob = np.clip(0.3 + 0.5 * exp_df["churn_prob_true"], 0.1, 0.9)
    received_offer = (rng.random(len(exp_df)) < offer_prob).astype(int)
    exp_df["received_offer"] = received_offer

    offer_types = np.where(
        received_offer == 1,
        rng.choice(["credit_increase", "waived_fee", "travel_credit"],
                    size=len(exp_df), p=[0.45, 0.35, 0.20]),
        "none"
    )
    exp_df["offer_type"] = offer_types

    # ---- Treatment effect on 90-day retention ----
    # Baseline: without offer, retention = 1 - churn_prob_true
    base_retain_prob = 1 - exp_df["churn_prob_true"]

    is_subprime_or_near = exp_df["credit_score_band"].isin(["subprime", "near_prime"]).astype(int)

    uplift = np.zeros(len(exp_df))
    ci_mask = offer_types == "credit_increase"
    wf_mask = offer_types == "waived_fee"
    tc_mask = offer_types == "travel_credit"

    # Credit increase: strong for healthier credit profiles, weaker for stressed ones
    uplift[ci_mask] = np.where(is_subprime_or_near[ci_mask] == 1, 0.07, 0.20)
    # Waived fee: more effective for lower-credit-band customers (immediate relief)
    uplift[wf_mask] = np.where(is_subprime_or_near[wf_mask] == 1, 0.19, 0.10)
    # Travel credit: modest, fairly flat effect (loyalty-driven, not financial)
    uplift[tc_mask] = 0.12

    final_retain_prob = np.clip(base_retain_prob + uplift, 0.02, 0.98)
    exp_df["retained_post_offer"] = (rng.random(len(exp_df)) < final_retain_prob).astype(int)

    return exp_df.drop(columns=["churn_prob_true"])


if __name__ == "__main__":
    print(f"Generating {N:,} synthetic SME card accounts...")
    customers = generate_customers(N)
    print(f"Overall churn rate: {customers['churn_flag'].mean():.2%}")
    print(
        "Owner vs employee churn rate: "
        f"{customers.groupby('cardholder_role')['churn_flag'].mean().to_dict()}"
    )

    experiment = generate_offer_experiment(customers)
    print(f"\nOffer experiment size: {len(experiment):,}")
    print(
        "Retention by offer status: "
        f"{experiment.groupby('received_offer')['retained_post_offer'].mean().to_dict()}"
    )

    # Drop the ground-truth probability column from the main file --
    # in the real world you'd never have this; keep it only inside this
    # script's output for your own validation if needed.
    customers_out = customers.drop(columns=["churn_prob_true"])
    customers_out.to_csv("card_customers.csv", index=False)
    experiment.to_csv("offer_experiment.csv", index=False)
    print("\nSaved: card_customers.csv, offer_experiment.csv")
