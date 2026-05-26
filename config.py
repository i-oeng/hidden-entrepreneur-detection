# config.py - Global configuration for the Hidden Entrepreneur Detection pipeline
#
# All tuneable knobs, magic numbers, and paths live here.

import random
from pathlib import Path

import numpy as np

# Section Paths

# Root of the raw data files (parquet files sit directly in the project root)
DATA_DIR = Path(".")

# All output artefacts (plots, CSV) are written here
OUT_DIR = Path("./output")
OUT_DIR.mkdir(exist_ok=True)

# Section Reproducibility

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Section PU Bagging

N_BAGS    = 50    # Number of bags in the PU ensemble
BAG_RATIO = 1.0   # #negatives per bag = BAG_RATIO × #positives

# Section Reliable Negative Extraction

# Bottom N-th percentile of PU scores → used as "hard" clean negatives
RELIABLE_NEG_QUANTILE = 0.20

# Section Optuna / Model Tuning

N_OPTUNA_TRIALS = 40
N_CV_FOLDS      = 5

# CatBoost fixed hyperparameters (not tuned by Optuna)
CATBOOST_PARAMS = dict(
    iterations    = 600,
    learning_rate = 0.03,
    depth         = 7,
    eval_metric   = "AUC",
    random_seed   = SEED,
    verbose       = 0,
)

# Isolation Forest fixed hyperparameters
ISO_FOREST_PARAMS = dict(
    n_estimators  = 300,
    contamination = 0.05,
    random_state  = SEED,
    n_jobs        = -1,
)

# Section Ensemble weights

ENSEMBLE_WEIGHTS = dict(lgb=0.50, catboost=0.35, iso=0.15)

# Section MCC Semantic Sets
# ISO 18245 codes that signal B2B / commercial activity

B2B_MCCS: set[int] = {
    # Wholesale & industrial equipment
    5040, 5041, 5042, 5043, 5044, 5045, 5046, 5047, 5048, 5049,
    5065,   # Electrical parts & equipment
    5085,   # Industrial & personal service paper
    5099,   # Durable goods
    5094,   # Jewelry, watches, precious stones
    # Raw materials & office
    5111, 5112, 5113, 5122,   # Stationery, office, drugs/sundries
    5190, 5191, 5192, 5193, 5194, 5198, 5199,
    # Construction
    5211, 5251, 5261,
    # Professional & business services
    7311, 7312, 7319,   # Advertising
    7372, 7374, 7379,   # Computer services
    7380, 7381, 7382, 7389,   # Business services
    # Accounting, legal, medical (self-employed often bill here)
    8000, 8011, 8021, 8031, 8041, 8042, 8049,
    8050, 8062, 8099,
    # Freight & logistics
    4214, 4215, 4722, 4731,
    # Wholesale food & beverage
    5141, 5142, 5143, 5144, 5145, 5146, 5147, 5148, 5149,
}

# MCCs used by both consumers AND entrepreneurs (ambiguous but weighted)
MIXED_MCCS: set[int] = {5411, 5912, 5999, 5300, 5311, 5331, 5200}

# Section Feature list
# Canonical order used everywhere downstream

FEATURES: list[str] = [
    # Volume
    "tx_count", "total_spend", "mean_tx", "std_tx", "tx_amount_cv",
    "p95_tx", "max_tx", "median_tx",
    # Merchant diversity
    "unique_merchants", "unique_mccs", "mcc_entropy", "merchant_concentration",
    "spend_per_merchant",
    # MCC semantics
    "b2b_ratio", "mixed_mcc_ratio",
    # Temporal
    "business_hours_ratio", "weekend_ratio", "active_days", "active_months",
    "tx_per_active_day", "monthly_spend_cv", "gap_mean", "gap_std",
    # Channel
    "offline_ratio", "tokenized_ratio", "recurring_ratio", "round_large_ratio",
    # Geography
    "unique_countries", "foreign_ratio", "unique_merchant_countries",
    # Metadata - EXCLUDED from training features.
    # card_tier_enc and bank_name_enc are kept in the DataFrame for reference
    # but NOT used as model inputs. In synthetic data these are perfectly
    # correlated with the label; in real data they are a leakage risk.
    # Re-add only after confirming via SHAP they do not dominate.
    # "card_tier_enc", "bank_name_enc",
]

# Section Business Segmentation

SEGMENT_THRESHOLDS = [
    (0.85, "A - High Confidence Entrepreneur"),
    (0.60, "B - Likely Self-Employed"),
    (0.40, "C - Borderline"),
    (0.00, "D - True Consumer"),
]

SEGMENT_ACTIONS: dict[str, str] = {
    "A - High Confidence Entrepreneur": (
        "Immediate outreach: business card conversion, merchant acquiring offer"
    ),
    "B - Likely Self-Employed": (
        "Soft cross-sell: introduce SME credit, salary project features"
    ),
    "C - Borderline": (
        "Monitor for 1 more month; re-score with updated transaction window"
    ),
    "D - True Consumer": (
        "Standard consumer engagement; no action needed"
    ),
}

# Section Output columns

OUTPUT_COLUMNS: list[str] = [
    "card_number",
    "score_lgb", "score_catboost", "score_iso",
    "score_ensemble", "score_calibrated",
    "segment",
    # Key behavioural features for interpretability
    "tx_count", "total_spend", "b2b_ratio",
    "mcc_entropy", "merchant_concentration",
    "business_hours_ratio", "offline_ratio",
    "unique_merchants", "unique_countries",
]
