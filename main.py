# main.py - Hidden Commercial Activity Detection Pipeline
# Mastercard x AIESEC Hackathon | May 2026
#
# INSTALL DEPENDENCIES:
#   pip install polars lightgbm catboost shap optuna scikit-learn pyarrow
#               matplotlib seaborn scipy imbalanced-learn
#
# DATA FILES EXPECTED (project root, same directory as this file):
#   business_cards_MDQ.parquet
#   consumer_cards_MDQ.parquet
#   merchants_reference.parquet
#
# OUTPUT FILES PRODUCED (./output/):
#   hidden_entrepreneur_scores.csv  - scored consumer cards with segments
#   shap_summary.png                - SHAP feature explanation plot
#   feature_importance.png          - LightGBM feature importance
#   pr_curve.png                    - Precision-Recall curve
#
# Each pipeline step lives in its own module under pipeline/.
# All tuneable constants and paths are in config.py.
#
# LEAKAGE PREVENTION SUMMARY:
#   1. biz_df split into biz_train/biz_test BEFORE PU bagging
#   2. PU bags see only biz_train (never biz_test)
#   3. reliable_negs split into train/test — model trains on neg_train only
#   4. Validation uses biz_test + neg_test — both completely unseen
#   5. card_tier_enc / bank_name_enc removed from FEATURES (metadata leakage)

import warnings
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
from sklearn.model_selection import train_test_split

import config
from pipeline.ingest             import load_data
from pipeline.mcc_tags           import apply_mcc_tags
from pipeline.features           import build_features
from pipeline.pu_learning        import run_pu_bagging
from pipeline.reliable_negatives import extract_reliable_negatives
from pipeline.tuning             import tune_and_train
from pipeline.validation         import validate
from pipeline.scoring            import score_consumers
from pipeline.segmentation       import assign_segments, print_segment_summary
from pipeline.explainability     import run_shap, plot_feature_importance
from pipeline.export             import export_results


if __name__ == "__main__":

    # Section 1 - Load data
    all_txns = load_data(config.DATA_DIR)

    # Section 2 - MCC semantic tagging
    all_txns = apply_mcc_tags(all_txns, config.B2B_MCCS, config.MIXED_MCCS)

    # Section 3 - Feature engineering
    card_agg = build_features(all_txns, config.FEATURES)

    biz_df  = card_agg[card_agg["label"] == 1].copy().reset_index(drop=True)
    cons_df = card_agg[card_agg["label"] == 0].copy().reset_index(drop=True)

    # LEAKAGE PREVENTION: split business cards BEFORE PU bagging.
    # biz_test_df is locked away until validation.
    print(f"\nSplitting business cards: 80% train / 20% test")
    biz_train_idx, biz_test_idx = train_test_split(
        np.arange(len(biz_df)), test_size=0.2, random_state=config.SEED
    )
    biz_train_df = biz_df.iloc[biz_train_idx].reset_index(drop=True)
    biz_test_df  = biz_df.iloc[biz_test_idx].reset_index(drop=True)
    X_pos        = biz_train_df[config.FEATURES].values
    print(f"  biz_train: {len(biz_train_df):,}  biz_test: {len(biz_test_df):,}")

    # Section 4 - PU Bagging (only biz_train_df used as positives)
    pu_scores = run_pu_bagging(
        biz_train_df, cons_df, config.FEATURES,
        config.N_BAGS, config.BAG_RATIO, config.SEED,
    )

    # Section 5 - Reliable Negative extraction
    # Splits reliable negs into train/test internally.
    # Only reliable_negs_train enters the model training set.
    (reliable_negs_train, reliable_negs_test,
     train_df, X_train, y_train, pos_weight) = extract_reliable_negatives(
        biz_train_df, cons_df, pu_scores, config.FEATURES,
        config.RELIABLE_NEG_QUANTILE, config.SEED,
    )

    # Section 6 - Hyperparameter tuning + model training
    lgb_model, catboost_model, iso_model, best_params = tune_and_train(
        X_train, y_train, X_pos, pos_weight,
        seed            = config.SEED,
        n_trials        = config.N_OPTUNA_TRIALS,
        n_cv_folds      = config.N_CV_FOLDS,
        catboost_params = config.CATBOOST_PARAMS,
        iso_params      = config.ISO_FOREST_PARAMS,
    )

    # Section 7 - Validation
    # Uses ONLY held-out data:
    # Positives: biz_test_df (never seen during PU bagging or training)
    # Negatives: reliable_negs_test (never seen during training)
    val_results = validate(
        biz_test_df, cons_df,
        lgb_model, catboost_model, iso_model,
        X_pos, config.FEATURES, config.OUT_DIR,
        seed             = config.SEED,
        ensemble_weights = config.ENSEMBLE_WEIGHTS,
    )

    # Section 8 - Score all consumer cards + calibration
    # Calibration uses reliable_negs_train (training data) - NOT test.
    cons_df, iso_calib = score_consumers(
        cons_df, lgb_model, catboost_model, iso_model,
        iso_ref       = val_results["iso_ref"],
        X_pos         = X_pos,
        reliable_negs = reliable_negs_train,
        features      = config.FEATURES,
        ensemble_weights = config.ENSEMBLE_WEIGHTS,
    )

    # Section 8b - Save trained models for later use with predict.py
    from pipeline.save_models import save_models
    save_models(
        lgb_model, catboost_model, iso_model,
        iso_ref   = val_results["iso_ref"],
        iso_calib = iso_calib,
        out_dir   = config.OUT_DIR,
    )

    # Section 9 - Business segmentation
    cons_df = assign_segments(cons_df, config.SEGMENT_THRESHOLDS)
    print_segment_summary(cons_df, config.SEGMENT_ACTIONS)

    # Section 10 - SHAP explainability (model audit + presentation)
    run_shap(lgb_model, cons_df, config.FEATURES, config.OUT_DIR)
    plot_feature_importance(lgb_model, config.FEATURES, config.OUT_DIR)

    # Section 11 - Export results
    export_results(cons_df, config.OUT_DIR, config.OUTPUT_COLUMNS)

