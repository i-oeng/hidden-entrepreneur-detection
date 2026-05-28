# predict.py - Score NEW consumer cards using trained models
#
# Use:
#   python predict.py
#   python predict.py --input new_cards.parquet --output scored_output.csv
#
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
from pipeline.scoring       import predict_component_scores
from pipeline.explainability import add_reason_codes


def load_models(model_dir: Path) -> dict:
    """Load all saved model artifacts."""
    models = {}
    for name in [
        "lgb_model", "catboost_model", "iso_model",
        "iso_ref", "ensemble_weights",
    ]:
        path = model_dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Run main.py first to train and save models."
            )
        with open(path, "rb") as f:
            models[name] = pickle.load(f)

    calibrator_path = model_dir / "score_calibrator.pkl"
    if not calibrator_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {calibrator_path}\n"
            f"Run main.py first to train and save models."
        )
    with open(calibrator_path, "rb") as f:
        models["score_calibrator"] = pickle.load(f)
    return models


def score_new_data(
    txns: pl.DataFrame,
    models: dict,
    features: list[str],
) -> pd.DataFrame:
    """Run the full scoring pipeline on new transaction data."""
    # Apply MCC semantic tagging (identical to training)
    txns = apply_mcc_tags(txns, config.B2B_MCCS, config.MIXED_MCCS)

    # Build card-level features (identical to training)
    card_agg = build_features(txns, features)

    X = card_agg[features].values

    lgb_scores, catboost_scores, iso_scores, ensemble = predict_component_scores(
        X,
        models["lgb_model"],
        models["catboost_model"],
        models["iso_model"],
        models["iso_ref"],
        models["ensemble_weights"],
    )

    # Calibrate
    calibrated = models["score_calibrator"].predict(ensemble)

    # Attach scores
    card_agg["score_lgb"]        = lgb_scores
    card_agg["score_catboost"]   = catboost_scores
    card_agg["score_iso"]        = iso_scores
    card_agg["score_ensemble"]   = ensemble
    card_agg["score_auc_optimized"] = ensemble
    card_agg["score_calibrated"] = calibrated

    # Assign segments
    card_agg = assign_segments(card_agg, config.SEGMENT_THRESHOLDS)
    card_agg = add_reason_codes(
        models["lgb_model"], card_agg, features, top_k=config.N_REASON_CODES
    )

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
    scored = score_new_data(txns, models, config.FEATURES)

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_cols = [c for c in config.OUTPUT_COLUMNS if c in scored.columns]
    rank_col = "score_auc_optimized" if "score_auc_optimized" in scored.columns else "score_ensemble"
    scored[out_cols].sort_values(rank_col, ascending=False).to_csv(
        output_path, index=False
    )

    print(f"\nScored output saved to {output_path}")
    print(f"Total cards scored: {len(scored):,}")
    print_segment_summary(scored, config.SEGMENT_ACTIONS)


if __name__ == "__main__":
    main()
