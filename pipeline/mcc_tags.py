# mcc_tags.py - MCC Semantic Tagging
# Attach semantic MCC tags to transactions using ISO 18245 codes.

import polars as pl


def apply_mcc_tags(
    df: pl.DataFrame,
    b2b_mccs: set[int],
    mixed_mccs: set[int],
) -> pl.DataFrame:
    """Attach MCC semantic flags and temporal convenience columns to transactions."""
    print("\n" + "=" * 60)
    print("SECTION 2: MCC Semantic Tagging")
    print("=" * 60)

    df = df.with_columns([
        pl.col("mcc").cast(pl.Int64, strict=False)
          .is_in(list(b2b_mccs)).alias("is_b2b_mcc"),
        pl.col("mcc").cast(pl.Int64, strict=False)
          .is_in(list(mixed_mccs)).alias("is_mixed_mcc"),
        pl.col("transaction_timestamp").cast(pl.Datetime).alias("dt"),
        pl.col("transaction_date").cast(pl.Date).alias("date"),
    ])

    df = df.with_columns([
        pl.col("dt").dt.hour().alias("hour"),
        pl.col("dt").dt.weekday().alias("weekday"),   # 0=Mon, 6=Sun
        pl.col("date").dt.month().alias("month"),
    ])

    df = df.with_columns([
        # Business hours: Mon–Fri, 09:00–18:00
        (
            (pl.col("hour") >= 9) & (pl.col("hour") <= 18) &
            (pl.col("weekday") < 5)
        ).alias("is_business_hours"),
        (pl.col("weekday") >= 5).alias("is_weekend"),
        # Round large amounts are common in B2B invoicing (e.g. 50,000 KZT)
        (
            (pl.col("transaction_amount_kzt") % 1000 == 0) &
            (pl.col("transaction_amount_kzt") >= 10_000)
        ).alias("is_round_large"),
    ])

    print("MCC semantic tags applied.")
    return df
