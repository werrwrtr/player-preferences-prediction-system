# ==========================================================
# ALS RECOMMENDER (IMPROVED GRID SEARCH) - Steam dataset
#
# Implements:
# - user filtering: keep users with >= 5 interactions
# - item filtering: keep games with >= 20 users
# - no min-max normalization of confidence
# - stronger implicit feedback confidence:
#       log1p(playtime) * (1 + combined_rating_norm)
# - grid search over rank / regParam / alpha
# - leave-one-out test split by user
# - internal validation split for model selection
# - Precision@10 / Recall@10 / NDCG@10
# - saves model, metadata, mappings and top-10 recommendations
# ==========================================================

import os
import glob
import sys
import traceback
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.storagelevel import StorageLevel


# ==========================================================
# DEBUG / ERROR LOGGING
# ==========================================================

def print_full_error(exc_type, exc_value, exc_tb):
    print("\n" + "=" * 100)
    print("UNCAUGHT EXCEPTION")
    print("=" * 100)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print("=" * 100 + "\n")

sys.excepthook = print_full_error


# ==========================================================
# CONFIG
# ==========================================================

BASE_PATH = r"D:\steam_recommender_system\3. data_analysis\db"

MODEL_PATH = "als_model_improved"
RECS_PATH = "als_top10_improved.parquet"
MAPPINGS_PATH = "als_mappings_improved.pkl"
META_PATH = "als_meta_improved.pkl"

TOP_K = 10

# Filtering suggestions from analysis
MIN_USER_INTERACTIONS = 5
MIN_ITEM_USERS = 20

# Confidence / ALS tuning
# stronger confidence usually works better for implicit Steam data
ALPHA_GRID = [40.0, 60.0]
RANK_GRID = [64, 128]
REG_GRID = [0.05, 0.10]
MAX_ITER_GRID = [15]

# Validation split within train_ratings for model selection
VAL_RATIO = 0.30
MIN_INTERACTIONS_FOR_VAL = 5

# Leave-one-out test split by user
MAX_USERS_DEBUG = None  # e.g. 2000


# ==========================================================
# SPARK SESSION
# ==========================================================

spark = (
    SparkSession.builder
    .appName("ALS_Steam_Recommender_Improved")
    .config("spark.driver.memory", "16g")
    .config("spark.executor.memory", "16g")
    .config("spark.driver.maxResultSize", "4g")
    .config("spark.sql.shuffle.partitions", "400")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ==========================================================
# HELPERS
# ==========================================================

def latest_dataset_folder(base_path: str) -> str:
    folders = glob.glob(os.path.join(base_path, "final_dataset_*"))
    if not folders:
        raise FileNotFoundError(f"No final_dataset_* folders found in {base_path}")
    return max(folders, key=os.path.getmtime)


def normalize_rating_col(df: DataFrame, col_name: str, output_col: str) -> DataFrame:
    """
    Heuristic normalization to roughly 0..1:
    - if value > 1, assumes 0..100 scale and divides by 100
    - otherwise keeps the value
    """
    return df.withColumn(
        output_col,
        F.when(F.col(col_name).isNull(), F.lit(0.0))
         .when(F.col(col_name) > 1.0, F.col(col_name) / F.lit(100.0))
         .otherwise(F.col(col_name).cast("double"))
    )


def precision_at_k(recs, rel, k=10):
    if not recs or not rel:
        return 0.0
    recs = recs[:k]
    return len(set(recs) & set(rel)) / k


def recall_at_k(recs, rel, k=10):
    if not recs or not rel:
        return 0.0
    recs = recs[:k]
    return len(set(recs) & set(rel)) / len(rel)


def ndcg_at_k(recs, rel, k=10):
    if not recs or not rel:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(recs[:k]):
        if item in rel:
            dcg += 1.0 / np.log2(i + 2)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return 0.0 if idcg == 0 else dcg / idcg


def make_implicit_strength(df: DataFrame) -> DataFrame:
    """
    Implicit feedback strength without min-max normalization.
    Suggestion tested:
        log1p(playtime) * (1 + combined_rating_norm)
    """
    return df.withColumn(
        "rating",
        F.log1p(F.col("game_playtime_forever").cast("double")) * (F.lit(1.0) + F.col("combined_rating_norm"))
    ).withColumn(
        "rating",
        F.when(F.col("rating").isNull(), F.lit(1e-4))
         .otherwise(F.greatest(F.col("rating"), F.lit(1e-4)))
    )


def save_id_mappings(user_indexer_model, item_indexer_model, path: str):
    user_labels = user_indexer_model.labels
    item_labels = item_indexer_model.labels

    user_to_idx = {label: int(i) for i, label in enumerate(user_labels)}
    idx_to_user = {int(i): label for i, label in enumerate(user_labels)}

    item_to_idx = {label: int(i) for i, label in enumerate(item_labels)}
    idx_to_item = {int(i): label for i, label in enumerate(item_labels)}

    payload = {
        "user_to_idx": user_to_idx,
        "idx_to_user": idx_to_user,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
    }

    joblib.dump(payload, path, compress=3)
    return payload


def apply_count_filters(df: DataFrame) -> DataFrame:
    """
    Keep only users with >= MIN_USER_INTERACTIONS and items with >= MIN_ITEM_USERS.
    """
    print("\nApplying interaction count filters...")

    user_counts = df.groupBy("user_steamid").count().withColumnRenamed("count", "user_cnt")
    item_counts = df.groupBy("game_appid").count().withColumnRenamed("count", "item_cnt")

    df = (
        df.join(user_counts, on="user_steamid", how="inner")
          .join(item_counts, on="game_appid", how="inner")
          .filter(
              (F.col("user_cnt") >= F.lit(MIN_USER_INTERACTIONS)) &
              (F.col("item_cnt") >= F.lit(MIN_ITEM_USERS))
          )
          .drop("user_cnt", "item_cnt")
    )
    return df


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

latest_folder = latest_dataset_folder(BASE_PATH)
print(f"Loading: {latest_folder}")

df = spark.read.parquet(latest_folder)
print("Rows:", df.count())

needed_cols = [
    "user_steamid",
    "game_appid",
    "game_playtime_forever",
    "combined_rating",
    "steam_rating",
]

existing_cols = [c for c in needed_cols if c in df.columns]
df = df.select(*existing_cols)

df = df.filter(
    F.col("user_steamid").isNotNull() &
    F.col("game_appid").isNotNull()
)

df = df.withColumn("user_steamid_str", F.col("user_steamid").cast("string"))
df = df.withColumn("game_appid_str", F.col("game_appid").cast("string"))

df = normalize_rating_col(df, "combined_rating", "combined_rating_norm")
df = normalize_rating_col(df, "steam_rating", "steam_rating_norm")

print("After cleaning rows:", df.count())

df = apply_count_filters(df)
print("After user/item filters:", df.count())


# ==========================================================
# TRAIN / TEST SPLIT (leave-one-out)
# ==========================================================

print("\nCreating leave-one-out split...")

window = Window.partitionBy("user_steamid_str").orderBy(
    F.desc("game_playtime_forever"),
    F.asc("game_appid_str")
)

df = df.withColumn("rn", F.row_number().over(window))

test_df = df.filter(F.col("rn") == 1).drop("rn")
train_df = df.filter(F.col("rn") > 1).drop("rn")

if MAX_USERS_DEBUG is not None:
    keep_users = [
        r["user_steamid_str"]
        for r in train_df.select("user_steamid_str").distinct().limit(MAX_USERS_DEBUG).collect()
    ]
    train_df = train_df.filter(F.col("user_steamid_str").isin(keep_users))
    test_df = test_df.filter(F.col("user_steamid_str").isin(keep_users))

print("Train:", train_df.count())
print("Test:", test_df.count())


# ==========================================================
# INDEX USERS / ITEMS FOR ALS
# ==========================================================

print("\nEncoding users and items to integer ids...")

user_indexer = StringIndexer(
    inputCol="user_steamid_str",
    outputCol="user_idx_raw",
    handleInvalid="skip"
)

item_indexer = StringIndexer(
    inputCol="game_appid_str",
    outputCol="item_idx_raw",
    handleInvalid="skip"
)

user_indexer_model = user_indexer.fit(train_df)
train_df = user_indexer_model.transform(train_df)
test_df = user_indexer_model.transform(test_df)

item_indexer_model = item_indexer.fit(train_df)
train_df = item_indexer_model.transform(train_df)
test_df = item_indexer_model.transform(test_df)

train_df = train_df.withColumn("user", F.col("user_idx_raw").cast("int"))
train_df = train_df.withColumn("item", F.col("item_idx_raw").cast("int"))
test_df = test_df.withColumn("user", F.col("user_idx_raw").cast("int"))
test_df = test_df.withColumn("item", F.col("item_idx_raw").cast("int"))

train_df = train_df.filter(F.col("user").isNotNull() & F.col("item").isNotNull())
test_df = test_df.filter(F.col("user").isNotNull() & F.col("item").isNotNull())

print("Indexed train rows:", train_df.count())
print("Indexed test rows:", test_df.count())

print("\nSchema check:")
train_df.select("user", "item").printSchema()
train_df.select(
    F.max("user").alias("max_user"),
    F.max("item").alias("max_item")
).show()


# ==========================================================
# BUILD IMPLICIT RATING
# ==========================================================

print("\nBuilding implicit strength...")

train_ratings = (
    make_implicit_strength(train_df)
    .select(
        F.col("user").cast("int"),
        F.col("item").cast("int"),
        F.col("rating").cast("double")
    )
    .persist(StorageLevel.MEMORY_AND_DISK)
)

test_ratings = (
    test_df.select(
        F.col("user").cast("int"),
        F.col("item").cast("int")
    )
    .persist(StorageLevel.MEMORY_AND_DISK)
)

print("Distinct train users:", train_ratings.select("user").distinct().count())
print("Distinct train items:", train_ratings.select("item").distinct().count())


# ==========================================================
# ALS GRID SEARCH WITH VALIDATION SPLIT
# ==========================================================

print("\nCreating ALS validation split...")

als_window = Window.partitionBy("user").orderBy(F.rand(42))

als_split = (
    train_ratings
    .withColumn("rn", F.row_number().over(als_window))
    .withColumn("cnt", F.count("*").over(Window.partitionBy("user")))
    .withColumn(
        "val_cut",
        F.when(
            F.col("cnt") >= F.lit(MIN_INTERACTIONS_FOR_VAL),
            F.greatest(F.lit(1), F.ceil(F.col("cnt") * F.lit(VAL_RATIO))).cast("int")
        ).otherwise(F.lit(0))
    )
    .withColumn("is_val", F.col("rn") <= F.col("val_cut"))
)

als_train = als_split.filter(~F.col("is_val")).select("user", "item", "rating").persist(StorageLevel.MEMORY_AND_DISK)
als_val = als_split.filter(F.col("is_val")).select("user", "item", "rating").persist(StorageLevel.MEMORY_AND_DISK)

print("ALS train rows:", als_train.count())
print("ALS val rows:", als_val.count())

val_users_df = als_val.select("user").distinct()
val_truth_df = als_val.groupBy("user").agg(F.collect_set("item").alias("true_items"))
val_truth = {int(r["user"]): set(r["true_items"]) for r in val_truth_df.collect()}


def eval_model_on_subset(model, users_df, truth_map, top_k=10):
    rec_df = model.recommendForUserSubset(users_df, top_k)
    user_rec_map = {}
    for row in rec_df.toLocalIterator():
        user = int(row["user"])
        rec_items = [int(x["item"]) for x in row["recommendations"]]
        user_rec_map[user] = rec_items

    precisions, recalls, ndcgs = [], [], []
    for user, rel in truth_map.items():
        recs = user_rec_map.get(user, [])
        if not recs:
            continue
        precisions.append(precision_at_k(recs, rel, top_k))
        recalls.append(recall_at_k(recs, rel, top_k))
        ndcgs.append(ndcg_at_k(recs, rel, top_k))

    mean_p = float(np.mean(precisions)) if precisions else 0.0
    mean_r = float(np.mean(recalls)) if recalls else 0.0
    mean_n = float(np.mean(ndcgs)) if ndcgs else 0.0
    return mean_p, mean_r, mean_n


print("\nStarting grid search...")

best_model = None
best_params = None
best_score = -1.0
best_metrics = None

for alpha in ALPHA_GRID:
    for rank in RANK_GRID:
        for reg in REG_GRID:
            for max_iter in MAX_ITER_GRID:
                print(f"\nTrying ALS: alpha={alpha}, rank={rank}, regParam={reg}, maxIter={max_iter}")

                als = ALS(
                    userCol="user",
                    itemCol="item",
                    ratingCol="rating",
                    implicitPrefs=True,
                    nonnegative=True,
                    coldStartStrategy="drop",
                    rank=rank,
                    regParam=reg,
                    maxIter=max_iter,
                    alpha=alpha,
                    seed=42
                )

                try:
                    model = als.fit(als_train)
                    mean_p, mean_r, mean_n = eval_model_on_subset(model, val_users_df, val_truth, TOP_K)

                    print(
                        f"Validation metrics -> "
                        f"Precision@10={mean_p:.4f}, "
                        f"Recall@10={mean_r:.4f}, "
                        f"NDCG@10={mean_n:.4f}"
                    )

                    if mean_n > best_score:
                        best_score = mean_n
                        best_model = model
                        best_params = {
                            "alpha": alpha,
                            "rank": rank,
                            "regParam": reg,
                            "maxIter": max_iter,
                        }
                        best_metrics = {
                            "precision_at_10": mean_p,
                            "recall_at_10": mean_r,
                            "ndcg_at_10": mean_n,
                        }

                except Exception as e:
                    print("\n" + "=" * 100)
                    print("GRID SEARCH CONFIG FAILED")
                    print("=" * 100)
                    print(f"alpha={alpha}, rank={rank}, regParam={reg}, maxIter={max_iter}")
                    print(type(e).__name__, str(e))
                    traceback.print_exc()
                    print("=" * 100 + "\n")
                    continue

print("\nBest ALS params:")
print(best_params)
print("Best validation metrics:")
print(best_metrics)

if best_params is None:
    raise RuntimeError("No ALS configuration finished successfully.")


# ==========================================================
# FINAL TRAINING ON FULL TRAIN DATA WITH BEST PARAMS
# ==========================================================

print("\nRefitting final ALS model on full train_ratings...")

final_als = ALS(
    userCol="user",
    itemCol="item",
    ratingCol="rating",
    implicitPrefs=True,
    nonnegative=True,
    coldStartStrategy="drop",
    rank=best_params["rank"],
    regParam=best_params["regParam"],
    maxIter=best_params["maxIter"],
    alpha=best_params["alpha"],
    seed=42
)

final_model = final_als.fit(train_ratings)
print("Final ALS model trained.")


# ==========================================================
# TEST RECOMMENDATIONS
# ==========================================================

print("\nGenerating test recommendations...")

test_users_df = test_ratings.select("user").distinct()
test_truth_df = test_ratings.groupBy("user").agg(F.collect_set("item").alias("true_items"))
test_truth = {int(r["user"]): set(r["true_items"]) for r in test_truth_df.collect()}

test_recs_df = final_model.recommendForUserSubset(test_users_df, TOP_K)

recs_rows = []
user_rec_map = {}

for row in tqdm(test_recs_df.toLocalIterator(), desc="Collecting test recs"):
    user = int(row["user"])
    rec_items = [int(x["item"]) for x in row["recommendations"]]
    rec_scores = [float(x["rating"]) for x in row["recommendations"]]

    user_rec_map[user] = rec_items

    for rank, (item, score) in enumerate(zip(rec_items, rec_scores), start=1):
        recs_rows.append({
            "user_idx": user,
            "rank": rank,
            "item_idx": item,
            "score": score
        })

recs_df = pd.DataFrame(recs_rows)


# ==========================================================
# MAP BACK TO ORIGINAL IDS
# ==========================================================

print("\nSaving mappings...")

mappings = save_id_mappings(user_indexer_model, item_indexer_model, MAPPINGS_PATH)
idx_to_user = mappings["idx_to_user"]
idx_to_item = mappings["idx_to_item"]

recs_df["user_steamid"] = recs_df["user_idx"].map(idx_to_user)
recs_df["game_appid"] = recs_df["item_idx"].map(idx_to_item)

recs_df = recs_df[
    [
        "user_steamid",
        "user_idx",
        "rank",
        "game_appid",
        "item_idx",
        "score",
    ]
]

recs_df.to_parquet(RECS_PATH, index=False)
print(f"Recommendations saved: {len(recs_df):,} -> {RECS_PATH}")


# ==========================================================
# FINAL TEST EVALUATION
# ==========================================================

print("\nEvaluating on held-out test split...")

precisions, recalls, ndcgs = [], [], []

evaluated_users = 0

for i, (user, recs) in enumerate(tqdm(user_rec_map.items(), desc="Evaluating users", unit="users")):
    rel = test_truth.get(user, set())
    if not rel:
        continue

    evaluated_users += 1
    precisions.append(precision_at_k(recs, rel, TOP_K))
    recalls.append(recall_at_k(recs, rel, TOP_K))
    ndcgs.append(ndcg_at_k(recs, rel, TOP_K))

    if (i + 1) % 1000 == 0:
        print("\n" + "=" * 50)
        print(f"Users processed: {i + 1:,}")
        print(f"Current Precision@10: {np.mean(precisions):.4f}")
        print(f"Current Recall@10: {np.mean(recalls):.4f}")
        print(f"Current NDCG@10: {np.mean(ndcgs):.4f}")
        print("=" * 50 + "\n")

print("\n" + "=" * 50)
print("FINAL TEST METRICS")
print("=" * 50)
print(f"Precision@10 = {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
print(f"Recall@10    = {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
print(f"NDCG@10      = {np.mean(ndcgs):.4f} (±{np.std(ndcgs):.4f})")
print(f"Total users evaluated: {evaluated_users:,}")


# ==========================================================
# SAVE FINAL MODEL + META
# ==========================================================

print("\nSaving final ALS model...")

final_model.write().overwrite().save(MODEL_PATH)

meta = {
    "model_path": MODEL_PATH,
    "recs_path": RECS_PATH,
    "mappings_path": MAPPINGS_PATH,
    "top_k": TOP_K,
    "best_params": best_params,
    "best_validation_metrics": best_metrics,
    "final_test_metrics": {
        "precision_at_10": float(np.mean(precisions)),
        "recall_at_10": float(np.mean(recalls)),
        "ndcg_at_10": float(np.mean(ndcgs)),
    },
    "filters": {
        "min_user_interactions": MIN_USER_INTERACTIONS,
        "min_item_users": MIN_ITEM_USERS,
    },
}

joblib.dump(meta, META_PATH, compress=3)

print(f"Meta saved: {META_PATH}")
print(f"ALS model saved: {MODEL_PATH}")
print(f"Mappings saved: {MAPPINGS_PATH}")
print("DONE!")


if __name__ == "__main__":
    try:
        pass
    finally:
        try:
            train_ratings.unpersist()
            test_ratings.unpersist()
            als_train.unpersist()
            als_val.unpersist()
        except Exception:
            pass
        try:
            spark.stop()
        except Exception:
            pass
