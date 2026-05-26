# reliable_negatives.py - Reliable Negative Extraction
# Extract reliable negatives after PU scoring to avoid contamination.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def extract_reliable_negatives(
    biz_df: pd.DataFrame,
    cons_df: pd.DataFrame,
    pu_scores: pd.Series,
    features: list[str],
    quantile: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, float]:
    """Identify clean negatives and build the supervised training set."""
    cons_df = cons_df.copy()
    cons_df["pu_score"] = pu_scores

    neg_threshold = cons_df["pu_score"].quantile(quantile)
    reliable_negs = cons_df[cons_df["pu_score"] <= neg_threshold].copy()

    print(f"\nReliable Negatives (bottom {int(quantile * 100)}%): {len(reliable_negs):,} cards")

    # SPLIT reliable negatives into train (80%) and test (20%).
    # Only train portion enters the training set.
    # Test portion is reserved exclusively for validation in Section 7.
    reliable_negs_train, reliable_negs_test = train_test_split(
        reliable_negs, test_size=0.2, random_state=seed,
    )
    reliable_negs_train = reliable_negs_train.reset_index(drop=True)
    reliable_negs_test  = reliable_negs_test.reset_index(drop=True)

    print(f"  Reliable neg train : {len(reliable_negs_train):,} cards (for model training)")
    print(f"  Reliable neg test  : {len(reliable_negs_test):,} cards (held out for validation)")

    # Build clean training set
    train_df = pd.concat([
        biz_df.assign(label=1),
        reliable_negs_train.assign(label=0),
    ], ignore_index=True)

    X_train    = train_df[features].values
    y_train    = train_df["label"].values
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(
        f"Clean training set: {len(train_df):,} total | "
        f"{int(y_train.sum()):,} positives | pos_weight={pos_weight:.1f}"
    )

    return reliable_negs_train, reliable_negs_test, train_df, X_train, y_train, pos_weight
