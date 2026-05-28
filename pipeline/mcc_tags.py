# Key Notion:
#   We don't care which merchant was visited - we care what *economic role*
#   that merchant plays. B2B MCC groups signal procurement, logistics, and
#   professional services that ordinary consumers rarely visit.
#
# Standard:
#   MCC codes are drawn from ISO 18245. The full B2B / MIXED sets are
#   defined in config.py so they can be updated without touching this file.
#
# CONTRACT:
#   apply_mcc_tags(df, b2b_mccs, mixed_mccs) -> pl.DataFrame
#   Adds boolean / temporal columns to the transaction DataFrame.

import polars as pl


def apply_mcc_tags(
    df: pl.DataFrame,
    b2b_mccs: set[int],
    mixed_mccs: set[int],
) -> pl.DataFrame:
    """
    Attach MCC semantic flags and temporal convenience columns to the
    transaction DataFrame.
    Note: Polars dt.weekday() is ISO-coded: 1=Mon ... 7=Sun.

    New columns added
    is_b2b_mcc       : bool - MCC is in the ISO 18245 B2B set
    is_mixed_mcc     : bool - MCC is in the ambiguous consumer/B2B set
    dt               : Datetime - parsed transaction_timestamp
    date             : Date - parsed transaction_date
    hour             : Int8  - hour-of-day (0–23)
    weekday          : Int8  - day-of-week (0=Mon … 6=Sun)
    month            : Int8  - calendar month (1–12)
    is_business_hours: bool - Mon–Fri, 09:00–18:00
    is_weekend       : bool - Saturday or Sunday
    is_round_large   : bool - amount ≥ 10,000 KZT and divisible by 1,000
                              (round large amounts are common in B2B invoicing)
    """
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
        pl.col("dt").dt.weekday().alias("weekday"),   # ISO: 1=Mon, 7=Sun
        pl.col("date").dt.month().alias("month"),
    ])

    df = df.with_columns([
        # Business hours: Mon–Fri, 09:00–18:00
        (
            (pl.col("hour") >= 9) & (pl.col("hour") <= 18) &
            (pl.col("weekday") <= 5)
        ).alias("is_business_hours"),
        (pl.col("weekday") >= 6).alias("is_weekend"),
        # Round large amounts are common in B2B invoicing (e.g. 50,000 KZT)
        (
            (pl.col("transaction_amount_kzt") % 1000 == 0) &
            (pl.col("transaction_amount_kzt") >= 10_000)
        ).alias("is_round_large"),
    ])

    print("MCC semantic tags applied.")
    return df
