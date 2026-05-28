# quality.py - Data and feature quality gates

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


REQUIRED_TRANSACTION_COLUMNS = [
    "transaction_date",
    "transaction_timestamp",
    "transaction_amount_kzt",
    "mcc",
    "merchant_id",
    "channel",
    "bank_name",
    "country",
    "card_number",
    "card_tier",
    "tokenized",
    "is_recurring",
    "merchant_country",
    "recurring_capable",
    "label",
]


def _write_report(rows: list[dict], out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / filename, index=False)


def validate_transactions(df: pl.DataFrame, out_dir: Path) -> None:
    """Validate raw joined transactions and write a compact quality report."""
    print("\nRunning transaction data quality checks...")

    rows = []
    columns = set(df.columns)
    missing = [c for c in REQUIRED_TRANSACTION_COLUMNS if c not in columns]
    rows.append({
        "check": "required_columns_present",
        "status": "FAIL" if missing else "PASS",
        "value": ", ".join(missing) if missing else "all present",
    })

    if missing:
        _write_report(rows, out_dir, "data_quality_report.csv")
        raise ValueError(f"Missing required transaction columns: {missing}")

    row_count = df.height
    card_count = df["card_number"].n_unique()
    rows.extend([
        {"check": "transaction_rows", "status": "INFO", "value": row_count},
        {"check": "unique_cards", "status": "INFO", "value": card_count},
        {
            "check": "date_range",
            "status": "INFO",
            "value": f"{df['transaction_date'].min()} to {df['transaction_date'].max()}",
        },
    ])

    null_checks = [
        "card_number", "transaction_amount_kzt", "mcc",
        "merchant_id", "merchant_country", "channel",
    ]
    for col in null_checks:
        null_rate = df[col].null_count() / max(row_count, 1)
        rows.append({
            "check": f"null_rate_{col}",
            "status": "WARN" if null_rate > 0 else "PASS",
            "value": f"{null_rate:.6f}",
        })

    non_positive_amounts = (
        df.select((pl.col("transaction_amount_kzt") <= 0).sum()).item()
    )
    rows.append({
        "check": "non_positive_amounts",
        "status": "WARN" if non_positive_amounts else "PASS",
        "value": int(non_positive_amounts),
    })

    merchant_join_miss_rate = df["merchant_country"].null_count() / max(row_count, 1)
    rows.append({
        "check": "merchant_reference_join_miss_rate",
        "status": "WARN" if merchant_join_miss_rate > 0.01 else "PASS",
        "value": f"{merchant_join_miss_rate:.6f}",
    })

    _write_report(rows, out_dir, "data_quality_report.csv")
    print(f"Data quality report saved to {out_dir / 'data_quality_report.csv'}")


def validate_feature_frame(card_agg: pd.DataFrame, features: list[str], out_dir: Path) -> None:
    """Validate card-level features before model training."""
    print("\nRunning feature matrix quality checks...")

    rows = []
    missing_features = [c for c in features if c not in card_agg.columns]
    rows.append({
        "check": "configured_features_present",
        "status": "FAIL" if missing_features else "PASS",
        "value": ", ".join(missing_features) if missing_features else "all present",
    })
    if missing_features:
        _write_report(rows, out_dir, "feature_quality_report.csv")
        raise ValueError(f"Configured features missing from card_agg: {missing_features}")

    duplicate_cards = int(card_agg["card_number"].duplicated().sum())
    rows.append({
        "check": "duplicate_card_rows",
        "status": "FAIL" if duplicate_cards else "PASS",
        "value": duplicate_cards,
    })

    feature_df = card_agg[features]
    nan_count = int(feature_df.isna().sum().sum())
    finite_mask = np.isfinite(feature_df.to_numpy(dtype=float))
    nonfinite_count = int((~finite_mask).sum())
    rows.extend([
        {
            "check": "feature_nan_count",
            "status": "FAIL" if nan_count else "PASS",
            "value": nan_count,
        },
        {
            "check": "feature_nonfinite_count",
            "status": "FAIL" if nonfinite_count else "PASS",
            "value": nonfinite_count,
        },
        {"check": "card_rows", "status": "INFO", "value": len(card_agg)},
        {"check": "feature_count", "status": "INFO", "value": len(features)},
    ])

    label_counts = card_agg["label"].value_counts().to_dict()
    rows.append({
        "check": "label_counts",
        "status": "INFO",
        "value": label_counts,
    })

    _write_report(rows, out_dir, "feature_quality_report.csv")
    if duplicate_cards or nan_count or nonfinite_count:
        raise ValueError("Feature quality checks failed; see feature_quality_report.csv")

    print(f"Feature quality report saved to {out_dir / 'feature_quality_report.csv'}")
