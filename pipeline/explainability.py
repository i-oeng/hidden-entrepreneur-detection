# MODEL AUDIT: Ensure model learns behavioural signals, not metadata.

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def run_shap(
    lgb_model,
    cons_df: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    top_n: int = 500,
) -> None:
    """Compute SHAP values on top-N scored cards and save summary plot."""
    print("\n" + "=" * 60)
    print("SECTION 10: SHAP Explainability")
    print("=" * 60)

    top_idx = cons_df["score_ensemble"].nlargest(top_n).index
    X_shap  = cons_df.loc[top_idx, features].values

    explainer = shap.TreeExplainer(lgb_model)
    shap_vals = explainer.shap_values(X_shap)

    # Handle both 2D and 3D SHAP output formats
    if isinstance(shap_vals, list):
        shap_matrix = shap_vals[1]          # class=1 for binary classifiers
    elif shap_vals.ndim == 3:
        shap_matrix = shap_vals[:, :, 1]
    else:
        shap_matrix = shap_vals

    plt.figure(figsize=(12, 9))
    shap.summary_plot(
        shap_matrix, X_shap, feature_names=features,
        show=False, max_display=20,
    )
    plt.title(
        f"SHAP Feature Contributions - Top {top_n} Predicted Hidden Entrepreneurs",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP summary plot saved.")


def plot_feature_importance(
    lgb_model,
    features: list[str],
    out_dir: Path,
    top_n: int = 20,
) -> None:
    """Save LightGBM feature importance bar chart."""
    fi = pd.DataFrame({
        "feature"   : features,
        "importance": lgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi.head(top_n), x="importance", y="feature", palette="Reds_r")
    plt.title(f"Top {top_n} Feature Importances - LightGBM", fontsize=13)
    plt.xlabel("Importance (# splits)")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Feature importance plot saved.")
