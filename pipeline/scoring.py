# scoring.py - Score All Consumer Cards
# Apply the held-out ensemble calibrator learned during validation.

import numpy as np
import pandas as pd


def score_isolation_forest(
    iso_model,
    X: np.ndarray,
    iso_ref: np.ndarray,
) -> np.ndarray:
    """Map IsolationForest decision scores to a clipped business-likeness score."""
    raw = iso_model.decision_function(X)
    lo, hi = iso_ref.min(), iso_ref.max()
    scores = 1.0 - (raw - lo) / (hi - lo + 1e-9)
    return np.clip(scores, 0.0, 1.0)


def blend_scores(
    lgb_scores: np.ndarray,
    catboost_scores: np.ndarray,
    iso_scores: np.ndarray,
    ensemble_weights: dict,
) -> np.ndarray:
    """Weighted ensemble with normalized weights and bounded output."""
    weight_sum = sum(ensemble_weights.values())
    ensemble = (
        ensemble_weights["lgb"]      * lgb_scores +
        ensemble_weights["catboost"] * catboost_scores +
        ensemble_weights["iso"]      * iso_scores
    ) / weight_sum
    return np.clip(ensemble, 0.0, 1.0)


def predict_component_scores(
    X: np.ndarray,
    lgb_model,
    catboost_model,
    iso_model,
    iso_ref: np.ndarray,
    ensemble_weights: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score a feature matrix with every model and the blended ensemble."""
    lgb_scores = lgb_model.predict_proba(X)[:, 1]
    catboost_scores = catboost_model.predict_proba(X)[:, 1]
    iso_scores = score_isolation_forest(iso_model, X, iso_ref)
    ensemble_scores = blend_scores(
        lgb_scores, catboost_scores, iso_scores, ensemble_weights
    )
    return lgb_scores, catboost_scores, iso_scores, ensemble_scores


def score_consumers(
    cons_df: pd.DataFrame,
    lgb_model,
    catboost_model,
    iso_model,
    iso_ref: np.ndarray,
    score_calibrator,
    features: list[str],
    ensemble_weights: dict,
) -> pd.DataFrame:
    """Score consumer cards and apply held-out ensemble calibration."""
    print("\n" + "=" * 60)
    print("SECTION 8: Scoring All Consumer Cards + Calibration")
    print("=" * 60)

    X_cons_all = cons_df[features].values

    lgb_cons, catboost_cons, iso_cons, ensemble_cons = predict_component_scores(
        X_cons_all, lgb_model, catboost_model, iso_model, iso_ref, ensemble_weights
    )

    calibrated_cons = score_calibrator.predict(ensemble_cons)

    cons_df = cons_df.copy()
    cons_df["score_lgb"]        = lgb_cons
    cons_df["score_catboost"]   = catboost_cons
    cons_df["score_iso"]        = iso_cons
    cons_df["score_ensemble"]   = ensemble_cons
    cons_df["score_auc_optimized"] = ensemble_cons
    cons_df["score_calibrated"] = calibrated_cons

    print("All consumer cards scored and calibrated.")
    return cons_df
