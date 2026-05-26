"""Quick investigation: what makes business and consumer cards perfectly separable?"""
import polars as pl
import numpy as np
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

biz = pl.read_parquet("business_cards_MDQ.parquet")
con = pl.read_parquet("consumer_cards_MDQ.parquet")

amt = "transaction_amount_kzt"

print("=== KEY DISTRIBUTIONS ===")
print(f"Business mean amount:   {biz[amt].mean():.0f}  median: {biz[amt].median():.0f}")
print(f"Consumer mean amount:   {con[amt].mean():.0f}  median: {con[amt].median():.0f}")
print(f"Business MCCs: {len(set(biz['mcc'].unique().to_list()))}  Consumer MCCs: {len(set(con['mcc'].unique().to_list()))}")

B2B_MCCS = {
    5040,5041,5042,5043,5044,5045,5046,5047,5048,5049,5065,5085,5099,5094,
    5111,5112,5113,5122,5190,5191,5192,5193,5194,5198,5199,5211,5251,5261,
    7311,7312,7319,7372,7374,7379,7380,7381,7382,7389,
    8000,8011,8021,8031,8041,8042,8049,8050,8062,8099,
    4214,4215,4722,4731,5141,5142,5143,5144,5145,5146,5147,5148,5149,
}

print("\n=== B2B MCC RATIO PER CARD ===")
for label, df in [("Business", biz), ("Consumer", con)]:
    b2b = (
        df.with_columns(
            pl.col("mcc").cast(pl.Int64, strict=False).is_in(list(B2B_MCCS)).alias("is_b2b")
        )
        .group_by("card_number")
        .agg(pl.col("is_b2b").mean().alias("b2b_ratio"))
    )
    vals = b2b["b2b_ratio"].to_numpy()
    print(f"\n{label}:")
    print(f"  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}  "
          f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")
    print(f"  Cards with b2b_ratio == 0: {(vals == 0).sum()} / {len(vals)} "
          f"({100*(vals==0).sum()/len(vals):.1f}%)")

print("\n=== SINGLE-FEATURE SEPARABILITY TEST ===")
print("Can ANY single feature alone perfectly separate the classes?\n")

all_txns = pl.concat([
    biz.with_columns(pl.lit(1).alias("label")),
    con.with_columns(pl.lit(0).alias("label")),
], how="diagonal")

card_level = (
    all_txns.group_by("card_number")
    .agg([
        pl.col("label").first(),
        pl.col("mcc").cast(pl.Int64, strict=False).is_in(list(B2B_MCCS)).mean().alias("b2b_ratio"),
        pl.col("merchant_id").n_unique().alias("unique_merchants"),
        pl.col("mcc").n_unique().alias("unique_mccs"),
        pl.col(amt).mean().alias("mean_tx"),
        pl.col(amt).sum().alias("total_spend"),
        pl.len().alias("tx_count"),
    ])
)

pdf = card_level.to_pandas()
from sklearn.metrics import roc_auc_score

for feat in ["b2b_ratio", "unique_merchants", "unique_mccs", "mean_tx", "total_spend", "tx_count"]:
    auc = roc_auc_score(pdf["label"], pdf[feat])
    biz_vals = pdf[pdf["label"]==1][feat]
    con_vals = pdf[pdf["label"]==0][feat]
    biz_min, biz_max = biz_vals.min(), biz_vals.max()
    con_min, con_max = con_vals.min(), con_vals.max()
    overlap = (biz_min < con_max) and (con_min < biz_max)
    print(f"  {feat:25s}  AUC={auc:.4f}  biz=[{biz_min:.1f}, {biz_max:.1f}]  "
          f"con=[{con_min:.1f}, {con_max:.1f}]  overlap={'YES' if overlap else 'NO'}")

# The critical test: can total_spend alone separate perfectly?
print("\n=== TOTAL_SPEND DISTRIBUTION OVERLAP ===")
biz_spend = pdf[pdf["label"]==1]["total_spend"]
con_spend = pdf[pdf["label"]==0]["total_spend"]
print(f"Business total_spend:  min={biz_spend.min():.0f}  p5={biz_spend.quantile(0.05):.0f}  "
      f"p25={biz_spend.quantile(0.25):.0f}  median={biz_spend.median():.0f}  p95={biz_spend.quantile(0.95):.0f}")
print(f"Consumer total_spend:  min={con_spend.min():.0f}  p5={con_spend.quantile(0.05):.0f}  "
      f"p25={con_spend.quantile(0.25):.0f}  median={con_spend.median():.0f}  p95={con_spend.quantile(0.95):.0f}")

# How many consumer cards have total_spend > min business card?
biz_min_spend = biz_spend.min()
n_overlap = (con_spend >= biz_min_spend).sum()
print(f"\nConsumer cards with total_spend >= business minimum ({biz_min_spend:.0f}): "
      f"{n_overlap} / {len(con_spend)} ({100*n_overlap/len(con_spend):.1f}%)")

# mean_tx overlap
biz_mtx = pdf[pdf["label"]==1]["mean_tx"]
con_mtx = pdf[pdf["label"]==0]["mean_tx"]
print(f"\nBusiness mean_tx: p5={biz_mtx.quantile(0.05):.0f}  median={biz_mtx.median():.0f}  p95={biz_mtx.quantile(0.95):.0f}")
print(f"Consumer mean_tx: p5={con_mtx.quantile(0.05):.0f}  median={con_mtx.median():.0f}  p95={con_mtx.quantile(0.95):.0f}")
