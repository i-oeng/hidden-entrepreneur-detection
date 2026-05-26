# predict.py - Score NEW consumer cards using trained models
#
# USAGE:
#   python predict.py
#   python predict.py --input new_cards.parquet --output scored_output.csv
#
# PREREQUISITES:
#   Run main.py first. It saves trained models to ./output/models/.

import argparse
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

import config
from pipeline.mcc_tags      import apply_mcc_tags
from pipeline.features      import build_features
from pipeline.segmentation  import assign_segments, print_segment_summary


def load_models(model_dir: Path) -> dict:
    """Load all saved model artifacts."""
    models = {}
    for name in ["lgb_model", "catboost_model", "iso_model", "iso_calib", "iso_ref"]:
        path = model_dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Run main.py first to train and save models."
            )
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models


def score_new_data(
    txns: pl.DataFrame,
    models: dict,
    features: list[str],
    ensemble_weights: dict,
) -> pd.DataFrame:
    """Run the full scoring pipeline on new transaction data."""
    # Apply MCC semantic tagging (identical to training)
    txns = apply_mcc_tags(txns, config.B2B_MCCS, config.MIXED_MCCS)

    # Build card-level features (identical to training)
    card_agg = build_features(txns, features)

    X = card_agg[features].values

    # Score with each model
    lgb_scores      = models["lgb_model"].predict_proba(X)[:, 1]
    catboost_scores = models["catboost_model"].predict_proba(X)[:, 1]

    iso_ref = models["iso_ref"]
    iso_raw = models["iso_model"].decision_function(X)
    iso_scores = 1.0 - (iso_raw - iso_ref.min()) / (iso_ref.max() - iso_ref.min() + 1e-9)

    ensemble = (
        ensemble_weights["lgb"]      * lgb_scores +
        ensemble_weights["catboost"] * catboost_scores +
        ensemble_weights["iso"]      * iso_scores
    )

    # Calibrate
    calibrated = models["iso_calib"].predict(lgb_scores)

    # Attach scores
    card_agg["score_lgb"]        = lgb_scores
    card_agg["score_catboost"]   = catboost_scores
    card_agg["score_iso"]        = iso_scores
    card_agg["score_ensemble"]   = ensemble
    card_agg["score_calibrated"] = calibrated

    # Assign segments
    card_agg = assign_segments(card_agg, config.SEGMENT_THRESHOLDS)

    return card_agg


def main():
    parser = argparse.ArgumentParser(
        description="Score new consumer cards using trained models"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to new consumer transactions parquet file"
    )
    parser.add_argument(
        "--output", type=str, default="output/new_card_scores.csv",
        help="Path to write scored output CSV"
    )
    parser.add_argument(
        "--model-dir", type=str, default="output/models",
        help="Directory containing saved model .pkl files"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print("=" * 60)
    print("Loading trained models...")
    print("=" * 60)
    models = load_models(model_dir)
    print(f"Models loaded from {model_dir}")

    # Load new transaction data
    if args.input is None:
        print("\nNo --input specified. Provide a parquet file of new transactions.")
        print("Example: python predict.py --input new_consumer_cards.parquet")
        return

    input_path = Path(args.input)
    print(f"\nLoading transactions from {input_path}...")
    txns = pl.read_parquet(input_path)

    # Add a dummy label column (required by feature engineering)
    if "label" not in txns.columns:
        txns = txns.with_columns(pl.lit(0).alias("label"))

    # Join merchant reference if available
    merchant_path = config.DATA_DIR / "merchants_reference.parquet"
    if merchant_path.exists():
        merchants_ref = pl.read_parquet(merchant_path)
        txns = txns.join(
            merchants_ref.select(["merchant_id", "merchant_country", "recurring_capable"]),
            on="merchant_id",
            how="left",
        )

    print(f"Transactions loaded: {txns.shape[0]:,} rows")

    # Score
    scored = score_new_data(txns, models, config.FEATURES, config.ENSEMBLE_WEIGHTS)

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_cols = [c for c in config.OUTPUT_COLUMNS if c in scored.columns]
    scored[out_cols].sort_values("score_calibrated", ascending=False).to_csv(
        output_path, index=False
    )

    print(f"\nScored output saved to {output_path}")
    print(f"Total cards scored: {len(scored):,}")
    print_segment_summary(scored, config.SEGMENT_ACTIONS)


if __name__ == "__main__":
    main()
