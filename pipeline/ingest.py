# ingest.py - Data Loading
# Use Polars for fast parquet loading.

from pathlib import Path

import polars as pl


def load_data(data_dir: Path) -> pl.DataFrame:
    """Load raw parquet files, attach labels, and join merchant metadata."""
    print("=" * 60)
    print("SECTION 1: Loading data")
    print("=" * 60)

    business_raw  = pl.read_parquet(data_dir / "business_cards_MDQ.parquet")
    consumer_raw  = pl.read_parquet(data_dir / "consumer_cards_MDQ.parquet")
    merchants_ref = pl.read_parquet(data_dir / "merchants_reference.parquet")

    business_raw = business_raw.with_columns(pl.lit(1).alias("label"))
    consumer_raw = consumer_raw.with_columns(pl.lit(0).alias("label"))

    all_txns = pl.concat([business_raw, consumer_raw], how="diagonal")

    # Join merchant metadata (country, recurring_capable)
    all_txns = all_txns.join(
        merchants_ref.select(["merchant_id", "merchant_country", "recurring_capable"]),
        on="merchant_id",
        how="left",
    )

    print(f"Business transactions : {business_raw.shape[0]:>12,}")
    print(f"Consumer transactions : {consumer_raw.shape[0]:>12,}")
    print(f"Total                 : {all_txns.shape[0]:>12,}")
    print(f"Unique business cards : {business_raw['card_number'].n_unique():>12,}")
    print(f"Unique consumer cards : {consumer_raw['card_number'].n_unique():>12,}")

    return all_txns
