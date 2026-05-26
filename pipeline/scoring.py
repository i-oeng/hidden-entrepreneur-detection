# scoring.py - Score All Consumer Cards + Calibration
# Calibrate model scores to true probabilities.

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def score_consumers(
    cons_df: pd.DataFrame,
    lgb_model,
    catboost_model,
    iso_model,
    iso_ref: np.ndarray,
    X_pos: np.ndarray,
    reliable_negs: pd.DataFrame,
    features: list[str],
    ensemble_weights: dict,
) -> tuple[pd.DataFrame, object]:
    """Score consumer cards and apply isotonic calibration to probabilities."""
    print("\n" + "=" * 60)
    print("SECTION 8: Scoring All Consumer Cards + Calibration")
    print("=" * 60)

    X_cons_all = cons_df[features].values

    lgb_cons      = lgb_model.predict_proba(X_cons_all)[:, 1]
    catboost_cons = catboost_model.predict_proba(X_cons_all)[:, 1]

    # IsolationForest: higher decision_function -> more normal among positives
    # We flip and normalise so that "more business-like" = higher score
    iso_cons_raw = iso_model.decision_function(X_cons_all)
    iso_cons     = 1.0 - (iso_cons_raw - iso_ref.min()) / (iso_ref.max() - iso_ref.min() + 1e-9)

    ensemble_cons = (
        ensemble_weights["lgb"]      * lgb_cons +
        ensemble_weights["catboost"] * catboost_cons +
        ensemble_weights["iso"]      * iso_cons
    )

    # Isotonic calibration - fit on positives + reliable negatives.
    # We use IsotonicRegression directly (sklearn >= 1.6 removed cv='prefit').
    # This is identical to what CalibratedClassifierCV(cv='prefit', method='isotonic')
    # was doing internally: fit a monotone mapping from raw scores -> probabilities.
    calib_X = np.vstack([X_pos, reliable_negs[features].values])
    calib_y = np.hstack([np.ones(len(X_pos)), np.zeros(len(reliable_negs))])

    calib_raw     = lgb_model.predict_proba(calib_X)[:, 1]
    iso_calib     = IsotonicRegression(out_of_bounds="clip")
    iso_calib.fit(calib_raw, calib_y)
    calibrated_cons = iso_calib.predict(lgb_cons)

    cons_df = cons_df.copy()
    cons_df["score_lgb"]        = lgb_cons
    cons_df["score_catboost"]   = catboost_cons
    cons_df["score_iso"]        = iso_cons
    cons_df["score_ensemble"]   = ensemble_cons
    cons_df["score_calibrated"] = calibrated_cons

    print("All consumer cards scored and calibrated.")
    return cons_df, iso_calib

