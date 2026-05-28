# validation.py - Held-Out Validation
# Validate on true business cards vs held-out reliable negatives.

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, fbeta_score,
)
from sklearn.model_selection import train_test_split

from pipeline.scoring import predict_component_scores


def _score_summary(group: str, score_name: str, values: np.ndarray) -> dict:
    """Return compact distribution stats for a model score."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {
            "group": group,
            "score": score_name,
            "cards": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "group": group,
        "score": score_name,
        "cards": len(values),
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "p01": np.quantile(values, 0.01),
        "p05": np.quantile(values, 0.05),
        "p25": np.quantile(values, 0.25),
        "p50": np.quantile(values, 0.50),
        "p75": np.quantile(values, 0.75),
        "p95": np.quantile(values, 0.95),
        "p99": np.quantile(values, 0.99),
        "max": values.max(),
    }


def _save_feature_separation_report(
    biz_eval: pd.DataFrame,
    neg_eval: pd.DataFrame,
    y_eval: np.ndarray,
    features: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    """Audit whether individual features almost solve the validation split."""
    rows = []
    for feature in features:
        scores = pd.concat([biz_eval[feature], neg_eval[feature]]).to_numpy(dtype=float)
        try:
            auc = roc_auc_score(y_eval, scores)
        except ValueError:
            auc = 0.5

        rows.append({
            "feature": feature,
            "roc_auc": auc,
            "separation_auc": max(auc, 1.0 - auc),
            "direction": "higher_for_business" if auc >= 0.5 else "lower_for_business",
            "business_mean": biz_eval[feature].mean(),
            "negative_mean": neg_eval[feature].mean(),
        })

    report = (
        pd.DataFrame(rows)
        .sort_values("separation_auc", ascending=False)
        .reset_index(drop=True)
    )
    path = out_dir / "feature_separation_report.csv"
    report.to_csv(path, index=False)

    print(f"\nFeature separation report saved to {path}")
    print("Top single-feature separators:")
    print(report.head(10).to_string(index=False))
    return report


def _blend_from_components(
    lgb_scores: np.ndarray,
    catboost_scores: np.ndarray,
    iso_scores: np.ndarray,
    weights: dict,
) -> np.ndarray:
    """Blend already-computed model scores with bounded output."""
    weight_sum = sum(weights.values())
    blended = (
        weights["lgb"] * lgb_scores +
        weights["catboost"] * catboost_scores +
        weights["iso"] * iso_scores
    ) / weight_sum
    return np.clip(blended, 0.0, 1.0)


def _tune_ensemble_weights(
    lgb_scores: np.ndarray,
    catboost_scores: np.ndarray,
    iso_scores: np.ndarray,
    y_true: np.ndarray,
    step: float = 0.05,
) -> tuple[dict, float]:
    """Grid-search convex ensemble weights for ROC-AUC on calibration data."""
    grid_size = int(round(1.0 / step))
    best_weights = {"lgb": 1.0, "catboost": 0.0, "iso": 0.0}
    best_roc_auc = -np.inf

    for lgb_i in range(grid_size + 1):
        for cat_i in range(grid_size - lgb_i + 1):
            iso_i = grid_size - lgb_i - cat_i
            weights = {
                "lgb": lgb_i / grid_size,
                "catboost": cat_i / grid_size,
                "iso": iso_i / grid_size,
            }
            scores = _blend_from_components(
                lgb_scores, catboost_scores, iso_scores, weights
            )
            roc_auc = roc_auc_score(y_true, scores)
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_weights = weights

    return best_weights, best_roc_auc


def validate(
    biz_df: pd.DataFrame,
    reliable_negs_test: pd.DataFrame,
    cons_df: pd.DataFrame,
    pu_scores: pd.Series,
    lgb_model,
    catboost_model,
    iso_model,
    X_pos: np.ndarray,
    features: list[str],
    out_dir: Path,
    seed: int,
    ensemble_weights: dict,
) -> dict:
    """Run held-out validation, fit ensemble calibration, and save the PR curve."""
    print("\n" + "=" * 60)
    print("SECTION 7: Held-Out Validation + Ensemble Calibration")
    print("=" * 60)
    print(f"  Test positives (biz_test):          {len(biz_df):,}")
    print(f"  Test negatives (reliable_neg_test): {len(reliable_negs_test):,}")

    biz_calib, biz_eval = train_test_split(
        biz_df, test_size=0.5, random_state=seed
    )
    neg_calib, neg_eval = train_test_split(
        reliable_negs_test, test_size=0.5, random_state=seed
    )

    print(f"  Calibration positives/negatives:    {len(biz_calib):,} / {len(neg_calib):,}")
    print(f"  Evaluation positives/negatives:     {len(biz_eval):,} / {len(neg_eval):,}")

    # Reference IsolationForest scores from training positives
    iso_ref = iso_model.decision_function(X_pos)

    X_calib = np.vstack([biz_calib[features].values, neg_calib[features].values])
    y_calib = np.hstack([np.ones(len(biz_calib)), np.zeros(len(neg_calib))])
    lgb_calib, catboost_calib, iso_calib, _ = predict_component_scores(
        X_calib, lgb_model, catboost_model, iso_model, iso_ref, ensemble_weights
    )
    tuned_weights, tuned_roc_auc = _tune_ensemble_weights(
        lgb_calib, catboost_calib, iso_calib, y_calib
    )

    print(
        "\nTuned ensemble weights "
        f"(calibration ROC-AUC={tuned_roc_auc:.4f}): "
        f"lgb={tuned_weights['lgb']:.2f}, "
        f"catboost={tuned_weights['catboost']:.2f}, "
        f"iso={tuned_weights['iso']:.2f}"
    )

    ensemble_calib = _blend_from_components(
        lgb_calib, catboost_calib, iso_calib, tuned_weights
    )

    score_calibrator = IsotonicRegression(out_of_bounds="clip")
    score_calibrator.fit(ensemble_calib, y_calib)

    X_eval = np.vstack([biz_eval[features].values, neg_eval[features].values])
    y_eval = np.hstack([np.ones(len(biz_eval)), np.zeros(len(neg_eval))])

    lgb_eval, catboost_eval, iso_eval, default_ensemble_eval = predict_component_scores(
        X_eval, lgb_model, catboost_model, iso_model, iso_ref, ensemble_weights
    )
    ensemble_eval = _blend_from_components(
        lgb_eval, catboost_eval, iso_eval, tuned_weights
    )
    calibrated_eval = score_calibrator.predict(ensemble_eval)

    results = {
        "LightGBM"        : lgb_eval,
        "CatBoost"        : catboost_eval,
        "Isolation Forest": iso_eval,
        "Default Ensemble": default_ensemble_eval,
        "Tuned Ensemble"  : ensemble_eval,
        "Calibrated Ens." : calibrated_eval,
    }
    metric_rows = []

    print(f"\n{'Model':<20} {'ROC-AUC*':>10} {'PR-AUC':>10}")
    print("-" * 42)
    for name, scores in results.items():
        roc = roc_auc_score(y_eval, scores)
        pr  = average_precision_score(y_eval, scores)
        metric_rows.append({"model": name, "roc_auc": roc, "pr_auc": pr})
        print(f"{name:<20} {roc:>10.4f} {pr:>10.4f}")

    feature_report = _save_feature_separation_report(
        biz_eval, neg_eval, y_eval, features, out_dir
    )
    best_model_auc = max(row["roc_auc"] for row in metric_rows)
    best_feature_auc = feature_report["separation_auc"].max()
    if best_model_auc >= 0.995 or best_feature_auc >= 0.995:
        print(
            "\nValidation caution: near-perfect metrics usually mean the synthetic "
            "data or selected reliable negatives are very separable. Treat this "
            "as a ranking benchmark, not as proof of real-world 100% accuracy."
        )

    print("\nSimple behavioural baselines:")
    for name in ["b2b_ratio", "total_spend", "tx_count", "unique_merchants"]:
        scores = pd.concat([biz_eval[name], neg_eval[name]]).to_numpy(dtype=float)
        roc = roc_auc_score(y_eval, scores)
        pr = average_precision_score(y_eval, scores)
        metric_rows.append({"model": f"Baseline: {name}", "roc_auc": roc, "pr_auc": pr})
        print(f"{name:<20} {roc:>10.4f} {pr:>10.4f}")

    # Find optimal threshold by maximising F0.5 (prioritizes precision)
    best_fbeta, best_thresh = 0.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        preds = (calibrated_eval >= t).astype(int)
        fbeta = fbeta_score(y_eval, preds, beta=0.5, zero_division=0)
        if fbeta > best_fbeta:
            best_fbeta, best_thresh = fbeta, t

    print(f"\nOptimal threshold : {best_thresh:.2f}  (F0.5 = {best_fbeta:.4f})")
    print("\nConfusion Matrix (Calibrated Ensemble @ optimal threshold):")
    cm = confusion_matrix(y_eval, (calibrated_eval >= best_thresh).astype(int))
    print(cm)
    print("\nClassification Report:")
    print(classification_report(
        y_eval, (calibrated_eval >= best_thresh).astype(int),
        target_names=["Consumer", "Business"], zero_division=0,
    ))

    cons_with_pu = cons_df.copy()
    cons_with_pu["pu_score"] = np.asarray(pu_scores)
    X_cons = cons_with_pu[features].values
    _, _, _, ensemble_cons = predict_component_scores(
        X_cons, lgb_model, catboost_model, iso_model, iso_ref, tuned_weights
    )
    cons_with_pu["score_auc_optimized"] = ensemble_cons
    cons_with_pu["score_calibrated"] = score_calibrator.predict(ensemble_cons)

    distribution_rows = []
    eval_groups = {
        "eval_known_business": y_eval == 1,
        "eval_reliable_negative": y_eval == 0,
    }
    for group_name, mask in eval_groups.items():
        distribution_rows.append(
            _score_summary(group_name, "score_auc_optimized", ensemble_eval[mask])
        )
        distribution_rows.append(
            _score_summary(group_name, "score_calibrated", calibrated_eval[mask])
        )

    consumer_groups = {
        "all_unlabeled_consumers": np.ones(len(cons_with_pu), dtype=bool),
        "low_pu_reliable_negative_pool": (
            cons_with_pu["pu_score"] <= cons_with_pu["pu_score"].quantile(0.20)
        ),
        "high_pu_hard_consumers_90": (
            cons_with_pu["pu_score"] >= cons_with_pu["pu_score"].quantile(0.90)
        ),
        "high_pu_hard_consumers_95": (
            cons_with_pu["pu_score"] >= cons_with_pu["pu_score"].quantile(0.95)
        ),
    }
    for group_name, mask in consumer_groups.items():
        group = cons_with_pu.loc[mask]
        distribution_rows.append(
            _score_summary(group_name, "pu_score", group["pu_score"].to_numpy())
        )
        distribution_rows.append(
            _score_summary(
                group_name,
                "score_auc_optimized",
                group["score_auc_optimized"].to_numpy(),
            )
        )
        distribution_rows.append(
            _score_summary(
                group_name,
                "score_calibrated",
                group["score_calibrated"].to_numpy(),
            )
        )

    distribution_path = out_dir / "score_distribution_report.csv"
    pd.DataFrame(distribution_rows).to_csv(distribution_path, index=False)
    print(f"\nScore distribution report saved to {distribution_path}")

    stress_rows = []
    for q in [0.80, 0.90, 0.95]:
        pu_cutoff = cons_with_pu["pu_score"].quantile(q)
        hard_cons = cons_with_pu[cons_with_pu["pu_score"] >= pu_cutoff]
        if hard_cons.empty:
            continue

        calibrated_hard = hard_cons["score_calibrated"].to_numpy()

        stress_rows.append({
            "pu_quantile_floor": q,
            "cards": len(hard_cons),
            "pu_score_cutoff": pu_cutoff,
            "avg_calibrated_score": calibrated_hard.mean(),
            "median_calibrated_score": np.median(calibrated_hard),
            "share_above_validation_threshold": (calibrated_hard >= best_thresh).mean(),
            "share_segment_a_threshold": (calibrated_hard >= 0.85).mean(),
            "avg_tx_count": hard_cons["tx_count"].mean(),
            "avg_total_spend": hard_cons["total_spend"].mean(),
            "avg_b2b_ratio": hard_cons["b2b_ratio"].mean(),
            "avg_unique_merchants": hard_cons["unique_merchants"].mean(),
        })

    if stress_rows:
        stress_path = out_dir / "hard_consumer_stress_report.csv"
        pd.DataFrame(stress_rows).to_csv(stress_path, index=False)
        print(f"\nHard-consumer stress report saved to {stress_path}")

    # ROC curve - primary jury-facing metric
    fpr, tpr, _ = roc_curve(y_eval, ensemble_eval)
    roc_auc = roc_auc_score(y_eval, ensemble_eval)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color="#CC0000", lw=2,
             label=f"Tuned Ensemble (ROC-AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#888888", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hidden Entrepreneur Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Precision-Recall curve - secondary imbalance diagnostic
    precision, recall, _ = precision_recall_curve(y_eval, calibrated_eval)
    pr_auc = average_precision_score(y_eval, calibrated_eval)

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

    metrics_path = out_dir / "validation_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print(f"\nValidation metrics saved to {metrics_path}")

    return {
        "best_thresh": best_thresh,
        "best_fbeta": best_fbeta,
        "iso_ref": iso_ref,
        "score_calibrator": score_calibrator,
        "ensemble_weights": tuned_weights,
    }
