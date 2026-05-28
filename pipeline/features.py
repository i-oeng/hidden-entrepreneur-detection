# Two-phase approach:
#   Phase A - Polars group_by().agg()  : vectorised, fast.
#   Phase B - Python row-level loops   : entropy / Herfindahl / gap stats.

from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl


# Secondary-feature helper functions
# Each has a written behavioural hypothesis.


def shannon_entropy(values) -> float:
    """
    Normalised Shannon entropy over MCC codes.
    Hypothesis: Entrepreneurs buy across more categories (high entropy) than consumers.
    """
    vals = list(values)
    if len(vals) == 0:
        return 0.0
    counts = pd.Series(vals).value_counts()
    probs  = counts / counts.sum()
    raw    = -(probs * np.log2(probs + 1e-9)).sum()
    return raw / np.log2(len(counts) + 1)   # normalise to [0, 1]


def herfindahl(values) -> float:
    """
    Herfindahl-Hirschman Index (HHI) of merchant concentration.
    Hypothesis: Entrepreneurs spread spend across many merchants (low HHI).
    """
    vals = list(values)
    if len(vals) == 0:
        return 1.0
    counts = pd.Series(vals).value_counts()
    shares = counts / counts.sum()
    return float((shares ** 2).sum())


def monthly_spend_cv(months, amounts) -> float:
    """
    Coefficient of variation of monthly spend totals.
    Hypothesis: Entrepreneurs show irregular spending cycles (higher CV).
    """
    monthly: dict = {}
    for m, a in zip(months, amounts):
        monthly[m] = monthly.get(m, 0) + a
    vals = list(monthly.values())
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals) / (np.mean(vals) + 1e-6))


def gap_stats(dates) -> tuple[float, float]:
    """
    Mean and standard deviation of inter-transaction day gaps.
    Hypothesis: Entrepreneurs have clustered spend patterns (high gap variance).
    """
    sorted_d = sorted(set(dates))
    if len(sorted_d) < 2:
        return 0.0, 0.0
    gaps = [int((sorted_d[i + 1] - sorted_d[i]) / np.timedelta64(1, "D")) for i in range(len(sorted_d) - 1)]
    return float(np.mean(gaps)), float(np.std(gaps))


# Main aggregation function

def build_features(all_txns: pl.DataFrame, features: list[str]) -> pd.DataFrame:
    """Aggregate transaction-level data to card-level features."""
    print("\n" + "=" * 60)
    print("SECTION 3: Feature Engineering")
    print("=" * 60)

    max_date = all_txns.select(pl.col("date").max()).item()
    rolling_exprs = []
    for window in (30, 60, 90):
        recent = pl.col("date") >= (max_date - timedelta(days=window - 1))
        rolling_exprs.extend([
            pl.col("transaction_amount_kzt").filter(recent).count().alias(f"tx_count_{window}d"),
            pl.col("transaction_amount_kzt").filter(recent).sum().alias(f"total_spend_{window}d"),
            pl.col("is_b2b_mcc").filter(recent).mean().alias(f"b2b_ratio_{window}d"),
            pl.col("merchant_id").filter(recent).n_unique().alias(f"unique_merchants_{window}d"),
            pl.col("date").filter(recent).n_unique().alias(f"active_days_{window}d"),
        ])

    # Phase A: Vectorised Polars aggregation 
    card_agg = (
        all_txns
        .group_by("card_number")
        .agg([
            pl.col("label").first(),
            pl.col("card_tier").first(),
            pl.col("bank_name").first(),

            # Volume 
            pl.len().alias("tx_count"),
            pl.col("transaction_amount_kzt").sum().alias("total_spend"),
            pl.col("transaction_amount_kzt").mean().alias("mean_tx"),
            pl.col("transaction_amount_kzt").std().alias("std_tx"),
            pl.col("transaction_amount_kzt").quantile(0.95).alias("p95_tx"),
            pl.col("transaction_amount_kzt").max().alias("max_tx"),
            pl.col("transaction_amount_kzt").median().alias("median_tx"),

            # Merchant diversity 
            pl.col("merchant_id").n_unique().alias("unique_merchants"),
            pl.col("mcc").n_unique().alias("unique_mccs"),

            # MCC semantics 
            pl.col("is_b2b_mcc").mean().alias("b2b_ratio"),
            pl.col("is_mixed_mcc").mean().alias("mixed_mcc_ratio"),

            # Temporal 
            pl.col("is_business_hours").mean().alias("business_hours_ratio"),
            pl.col("is_weekend").mean().alias("weekend_ratio"),
            pl.col("date").n_unique().alias("active_days"),
            pl.col("month").n_unique().alias("active_months"),

            # Channel 
            (pl.col("channel") == "offline").mean().alias("offline_ratio"),
            pl.col("tokenized").mean().alias("tokenized_ratio"),
            pl.col("is_recurring").mean().alias("recurring_ratio"),
            pl.col("is_round_large").mean().alias("round_large_ratio"),

            # Geography 
            pl.col("country").n_unique().alias("unique_countries"),
            (pl.col("country") != "KZ").mean().alias("foreign_ratio"),
            pl.col("merchant_country").n_unique().alias("unique_merchant_countries"),

            # Rolling-window features for faster early detection.
            *rolling_exprs,

            # Store raw lists for Python-level Phase B calculations
            pl.col("mcc").alias("_mcc_list"),
            pl.col("merchant_id").alias("_merchant_list"),
            pl.col("date").alias("_date_list"),
            pl.col("month").alias("_month_list"),
            pl.col("transaction_amount_kzt").alias("_amount_list"),
        ])
        .to_pandas()
    )

    print(f"Card-level aggregation: {card_agg.shape[0]:,} cards, {card_agg.shape[1]} columns")

    # Phase B: Python-level secondary features 
    # These require row-level iteration and cannot be expressed in Polars SQL.
    print("Computing secondary features (entropy, gaps, concentration)...")

    card_agg["mcc_entropy"] = card_agg["_mcc_list"].apply(shannon_entropy)
    card_agg["merchant_concentration"] = card_agg["_merchant_list"].apply(herfindahl)
    card_agg["monthly_spend_cv"] = card_agg.apply(
        lambda r: monthly_spend_cv(list(r["_month_list"]), list(r["_amount_list"])),
        axis=1,
    )

    gap_results          = card_agg["_date_list"].apply(lambda x: gap_stats(list(x)))
    card_agg["gap_mean"] = gap_results.apply(lambda x: x[0])
    card_agg["gap_std"]  = gap_results.apply(lambda x: x[1])

    # Derived ratio features 
    card_agg["tx_amount_cv"]      = card_agg["std_tx"] / (card_agg["mean_tx"] + 1e-6)
    card_agg["tx_per_active_day"] = card_agg["tx_count"] / (card_agg["active_days"] + 1)
    card_agg["spend_per_merchant"] = card_agg["total_spend"] / (card_agg["unique_merchants"] + 1)
    card_agg["spend_share_30d"] = card_agg["total_spend_30d"] / (card_agg["total_spend"] + 1e-6)
    card_agg["tx_share_30d"] = card_agg["tx_count_30d"] / (card_agg["tx_count"] + 1e-6)

    prior_60_spend = card_agg["total_spend_90d"] - card_agg["total_spend_30d"]
    prior_60_tx = card_agg["tx_count_90d"] - card_agg["tx_count_30d"]
    card_agg["spend_accel_30_vs_90"] = np.where(
        prior_60_spend > 0,
        card_agg["total_spend_30d"] / (prior_60_spend / 2),
        0.0,
    )
    card_agg["tx_accel_30_vs_90"] = np.where(
        prior_60_tx > 0,
        card_agg["tx_count_30d"] / (prior_60_tx / 2),
        0.0,
    )

    log_sources = [
        "tx_count", "total_spend", "mean_tx", "p95_tx", "max_tx",
        "unique_merchants", "total_spend_30d", "total_spend_60d",
        "total_spend_90d", "tx_count_30d", "tx_count_60d", "tx_count_90d",
    ]
    for source in log_sources:
        values = pd.to_numeric(card_agg[source], errors="coerce").fillna(0)
        card_agg[f"log_{source}"] = np.sign(values) * np.log1p(np.abs(values))

    # Ordinal-encode categoricals 
    # Tree models are not sensitive to encoding; ordinal is fine here.
    # Monitor via SHAP - if card_tier_enc / bank_name_enc dominate, drop them.
    card_agg["card_tier_enc"] = pd.factorize(card_agg["card_tier"])[0]
    card_agg["bank_name_enc"] = pd.factorize(card_agg["bank_name"])[0]

    # Drop raw list columns used for Phase B
    card_agg.drop(
        columns=[c for c in card_agg.columns if c.startswith("_")],
        inplace=True,
    )

    # Fill NaNs created by edge cases (new card, single txn, etc.)
    card_agg[features] = card_agg[features].fillna(0)

    print(f"Final feature count: {card_agg.shape[1]} columns")
    return card_agg
