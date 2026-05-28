# WHY NOT NAIVE BINARY CLASSIFICATION:
#   Labeling all consumer cards as "0" is wrong. Some are hidden entrepreneurs.
#   Training on contaminated negatives teaches the model a blurry boundary.
#
# WHY BAGGING PU (Mordelet & Vert, 2014) WORKS:
#   Each bag randomly samples |P| negatives from the unlabeled set U.
#   Hidden entrepreneurs occasionally land in the training negatives for a
#   given bag - but averaged across N_BAGS bags with different random draws,
#   their contamination effect cancels out.
#   True negatives appear consistently every bag → consistently low scores.
#   Hidden positives appear inconsistently → their averaged scores are higher.
#
# RESULT:
#   Ensemble of bag scores gives a PU-corrected probability surface that
#   can be thresholded to extract reliable negatives for the final model.
#
# CONTRACT:
#   run_pu_bagging(biz_df, cons_df, features, n_bags, bag_ratio, seed)
#       -> pd.Series  (PU scores indexed to cons_df)

import numpy as np
import pandas as pd
import lightgbm as lgb


def run_pu_bagging(
    biz_df: pd.DataFrame,
    cons_df: pd.DataFrame,
    features: list[str],
    n_bags: int,
    bag_ratio: float,
    seed: int,
) -> pd.Series:
    """
    Run PU Bagging and return out-of-bag scores for every consumer card.

    Parameters
    biz_df    : card-level DataFrame for known business cards (label=1)
    cons_df   : card-level DataFrame for consumer cards (unlabeled)
    features  : canonical feature column list (config.FEATURES)
    n_bags    : number of bags (default 50)
    bag_ratio : #negatives per bag = bag_ratio × #positives (default 1.0)
    seed      : base random seed; each bag uses seed + bag_index

    Returns
    pd.Series
        PU score for each consumer card (higher = more business-like).
        Indexed identically to cons_df.
    """
    print("\n" + "=" * 60)
    print(f"SECTION 4: PU Learning - Bagging Method ({n_bags} bags)")
    print("=" * 60)

    X_pos       = biz_df[features].values
    X_unlabeled = cons_df[features].values

    pu_scores     = np.zeros(len(cons_df))
    pu_bag_counts = np.zeros(len(cons_df))

    for bag in range(n_bags):
        n_neg   = int(len(X_pos) * bag_ratio)
        neg_idx = np.random.choice(len(X_unlabeled), size=n_neg, replace=True)

        # Out-of-bag mask: consumer cards NOT drawn for this bag
        oob_mask           = np.ones(len(X_unlabeled), dtype=bool)
        oob_mask[neg_idx]  = False

        X_bag = np.vstack([X_pos, X_unlabeled[neg_idx]])
        y_bag = np.hstack([np.ones(len(X_pos)), np.zeros(n_neg)])

        bag_clf = lgb.LGBMClassifier(
            n_estimators     = 200,
            learning_rate    = 0.05,
            num_leaves       = 31,
            min_child_samples= 20,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            n_jobs           = -1,
            random_state     = seed + bag,
            verbose          = -1,
        )
        bag_clf.fit(X_bag, y_bag)

        oob_preds                = bag_clf.predict_proba(X_unlabeled[oob_mask])[:, 1]
        pu_scores[oob_mask]     += oob_preds
        pu_bag_counts[oob_mask] += 1

        if (bag + 1) % 10 == 0:
            print(f"  Bag {bag + 1}/{n_bags} done")

    # Normalise - some cards may have been OOB fewer than n_bags times
    valid              = pu_bag_counts > 0
    pu_scores[valid]  /= pu_bag_counts[valid]

    pu_series = pd.Series(pu_scores, index=cons_df.index, name="pu_score")

    print(f"\nPU bagging complete.")
    print(f"  Consumer cards scoring > 0.70 : {(pu_series > 0.70).sum():,}")
    print(f"  Consumer cards scoring < 0.20 : {(pu_series < 0.20).sum():,}")

    return pu_series
