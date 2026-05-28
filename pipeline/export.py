#
# CONTRACT:
#   export_results(cons_df, out_dir, output_columns) -> None

from pathlib import Path

import pandas as pd


def export_results(
    cons_df: pd.DataFrame,
    out_dir: Path,
    output_columns: list[str],
) -> None:
    """
    Write the scored, segmented consumer card DataFrame to CSV and print
    the final pipeline summary.

    Parameters
    cons_df        : fully scored and segmented consumer DataFrame
    out_dir        : directory to write `hidden_entrepreneur_scores.csv`
    output_columns : ordered list of columns to include in the output file
    """
    print("\n" + "=" * 60)
    print("SECTION 11: Exporting Results")
    print("=" * 60)

    rank_col = "score_auc_optimized" if "score_auc_optimized" in cons_df.columns else "score_ensemble"
    results_df = (
        cons_df[output_columns]
        .sort_values(rank_col, ascending=False)
    )
    out_path = out_dir / "hidden_entrepreneur_scores.csv"
    results_df.to_csv(out_path, index=False)

    print(f"Scores saved to {out_path}")
    print(f"\nFinal summary:")
    print(f"  Total consumer cards scored   : {len(cons_df):,}")
    print(f"  Segment A (high confidence)   : {(cons_df['segment'] == 'A - High Confidence Entrepreneur').sum():,}")
    print(f"  Segment B (likely)            : {(cons_df['segment'] == 'B - Likely Self-Employed').sum():,}")
    print(f"  Segment C (borderline)        : {(cons_df['segment'] == 'C - Borderline').sum():,}")
    print(f"  Segment D (true consumer)     : {(cons_df['segment'] == 'D - True Consumer').sum():,}")
    print("\nPipeline complete. [OK]")
