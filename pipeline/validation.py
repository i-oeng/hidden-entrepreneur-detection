# validation.py - Held-Out Validation
# Validate on true business cards vs held-out reliable negatives.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix,
    classification_report, f1_score,
)



def _iso_score(iso_model, X: np.ndarray, iso_ref_scores: np.ndarray) -> np.ndarray:
    """Convert IsolationForest decision function to [0, 1] scale."""
    raw = iso_model.decision_function(X)
    lo, hi = iso_ref_scores.min(), iso_ref_scores.max()
    return 1.0 - (raw - lo) / (hi - lo + 1e-9)


def validate(
    biz_df: pd.DataFrame,
    cons_df: pd.DataFrame,
    lgb_model,
    catboost_model,
    iso_model,
    X_pos: np.ndarray,
    features: list[str],
    out_dir: Path,
    seed: int,
    ensemble_weights: dict,
) -> dict:
    """Run held-out validation and save the PR curve."""
    print("\n" + "=" * 60)
    print("SECTION 7: Validation on Held-Out Business Cards vs Random Consumers")
    print("=" * 60)
    print(f"  Test positives (biz_test):      {len(biz_df):,}")
    print(f"  Test negatives (random cons):   {len(biz_df):,}")

    # biz_df here is already the held-out test split (20% of business cards),
    # split before PU bagging in main.py - so these cards were never seen
    # during bag training, reliable-negative extraction, or model fitting.
    X_biz_test = biz_df[features].values

    # To get a realistic evaluation, we test against a random sample of
    # the general consumer population, NOT just the 'reliable negatives'.
    # This reflects the true distribution, including borderline cases.
    cons_test = cons_df.sample(n=len(biz_df), random_state=seed)
    X_neg_test = cons_test[features].values

    n_pos = len(X_biz_test)
    n_neg = len(X_neg_test)

    X_eval = np.vstack([X_biz_test, X_neg_test])
    y_eval = np.hstack([np.ones(n_pos), np.zeros(n_neg)])

    # Reference IsolationForest scores from training positives
    iso_ref = iso_model.decision_function(X_pos)

    # Score with each model
    lgb_eval      = lgb_model.predict_proba(X_eval)[:, 1]
    catboost_eval = catboost_model.predict_proba(X_eval)[:, 1]
    iso_eval      = _iso_score(iso_model, X_eval, iso_ref)
    ensemble_eval = (
        ensemble_weights["lgb"]      * lgb_eval +
        ensemble_weights["catboost"] * catboost_eval +
        ensemble_weights["iso"]      * iso_eval
    )

    results = {
        "LightGBM"        : lgb_eval,
        "CatBoost"        : catboost_eval,
        "Isolation Forest": iso_eval,
        "Ensemble"        : ensemble_eval,
    }

    print(f"\n{'Model':<20} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print("-" * 42)
    for name, scores in results.items():
        roc = roc_auc_score(y_eval, scores)
        pr  = average_precision_score(y_eval, scores)
        print(f"{name:<20} {roc:>10.4f} {pr:>10.4f}")

    # Find optimal threshold by maximising F1
    best_f1, best_thresh = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        preds = (ensemble_eval >= t).astype(int)
        f1 = f1_score(y_eval, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"\nOptimal threshold : {best_thresh:.2f}  (F1 = {best_f1:.4f})")
    print("\nConfusion Matrix (Ensemble @ optimal threshold):")
    cm = confusion_matrix(y_eval, (ensemble_eval >= best_thresh).astype(int))
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_eval, (ensemble_eval >= best_thresh).astype(int),
        target_names=["Consumer", "Business"], zero_division=0,
    ))

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_eval, ensemble_eval)
    pr_auc = average_precision_score(y_eval, ensemble_eval)

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, color="#CC0000", lw=2,
             label=f"Ensemble (PR-AUC={pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Hidden Entrepreneur Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"best_thresh": best_thresh, "best_f1": best_f1, "iso_ref": iso_ref}
