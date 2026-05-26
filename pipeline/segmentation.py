# segmentation.py - Business Segmentation
# Assign business segments based on calibrated scores.

import pandas as pd


def _assign_segment(score: float, thresholds: list[tuple[float, str]]) -> str:
    """Map a single calibrated score to its segment label."""
    for min_score, label in thresholds:
        if score >= min_score:
            return label
    return thresholds[-1][1]   # fallback to the lowest tier


def assign_segments(
    cons_df: pd.DataFrame,
    thresholds: list[tuple[float, str]],
) -> pd.DataFrame:
    """Attach a `segment` column to cons_df based on `score_calibrated`."""
    print("\n" + "=" * 60)
    print("SECTION 9: Business Segmentation")
    print("=" * 60)

    cons_df = cons_df.copy()
    cons_df["segment"] = cons_df["score_calibrated"].apply(
        lambda s: _assign_segment(s, thresholds)
    )
    return cons_df


def print_segment_summary(
    cons_df: pd.DataFrame,
    segment_actions: dict[str, str],
) -> None:
    """Print a per-segment summary table and recommended product actions."""
    seg_summary = (
        cons_df.groupby("segment")
        .agg(
            count           = ("card_number",      "count"),
            avg_score       = ("score_calibrated", "mean"),
            avg_tx_count    = ("tx_count",          "mean"),
            avg_total_spend = ("total_spend",       "mean"),
            avg_b2b_ratio   = ("b2b_ratio",         "mean"),
            avg_entropy     = ("mcc_entropy",       "mean"),
        )
        .sort_index()
    )

    print("\nSegment Summary:")
    print(seg_summary.to_string())
    print("\nRecommended Actions:")
    for seg, action in segment_actions.items():
        n = seg_summary.loc[seg, "count"] if seg in seg_summary.index else 0
        print(f"  [{n:>6,}] {seg}: {action}")
