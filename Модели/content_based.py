# ==========================================================
# CONTENT-BASED RECOMMENDER (FIXED / MEMORY-SAFE VERSION)
# Steam dataset
#
# Main fixes:
# - split before fitting feature pipeline (no leakage)
# - fit feature pipeline on train only
# - use FeatureHasher for developer/publisher to avoid huge OHE vectors
# - keep only item table with vectors in Pandas; interactions table stays slim
# - weak-interaction filtering on train only
# - leave-one-out evaluation
# - Precision@10 / Recall@10 / NDCG@10
# - separate saving of Spark pipeline and Python model
# - cold-start recommendation function for Streamlit
# ==========================================================

import os
import glob
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import FeatureHasher, VectorAssembler
from pyspark.ml import Pipeline

from sklearn.preprocessing import normalize


# ==========================================================
# CONFIG
# ==========================================================

BASE_PATH = r"D:\steam_recommender_system\3. data_analysis\db"

TOP_K = 10
MIN_PLAYTIME_MINUTES = 180

MODEL_PATH = "content_based_model_v4.pkl"
RECS_PATH = "content_based_top10_v4.parquet"
PIPELINE_PATH = "content_pipeline_v4"

SAVE_INTERMEDIATE_PARQUET = False
INTERMEDIATE_PATH = "content_based_train_debug.parquet"


# ==========================================================
# SPARK SESSION
# ==========================================================

spark = (
    SparkSession.builder
    .appName("ContentBasedFixed")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)


# ==========================================================
# HELPERS
# ==========================================================

def latest_dataset_folder(base_path: str) -> str:
    folders = glob.glob(os.path.join(base_path, "final_dataset_*"))
    if not folders:
        raise FileNotFoundError(f"No final_dataset_* folders found in {base_path}")
    return max(folders, key=os.path.getmtime)


def clean_string_col(df, col_name: str, replacement: str = "unknown"):
    return df.withColumn(
        col_name,
        F.when(
            (F.col(col_name).isNull()) | (F.trim(F.col(col_name)) == ""),
            F.lit(replacement),
        ).otherwise(F.col(col_name)),
    )


def normalize_rating_col(df, col_name: str, output_col: str):
    """
    Heuristic normalization to roughly 0..1:
    - if values look like percentages, divide by 100
    - otherwise keep as is
    """
    return df.withColumn(
        output_col,
        F.when(F.col(col_name).isNull(), F.lit(0.0))
        .when(F.col(col_name) > 1.0, F.col(col_name) / F.lit(100.0))
        .otherwise(F.col(col_name).cast("double")),
    )


def safe_to_array(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if hasattr(x, "toArray"):
        return x.toArray().astype(np.float32)
    if isinstance(x, dict):
        vec = np.zeros(x["size"], dtype=np.float32)
        for idx, val in zip(x["indices"], x["values"]):
            vec[idx] = val
        return vec
    raise ValueError(f"Unsupported vector type: {type(x)}")


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


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    latest_folder = latest_dataset_folder(BASE_PATH)
    print(f"Loading: {latest_folder}")

    df = spark.read.parquet(latest_folder)
    print("Rows:", df.count())

    # ------------------------------------------------------
    # CLEAN DATA + BASIC NORMALIZATION
    # ------------------------------------------------------
    print("\nCleaning data and building structured features...")

    df = clean_string_col(df, "game_detail_developer", "unknown")
    df = clean_string_col(df, "game_detail_publisher", "unknown")

    df = normalize_rating_col(df, "combined_rating", "combined_rating_norm")
    df = normalize_rating_col(df, "steam_rating", "steam_rating_norm")

    df = df.withColumn(
        "game_detail_release_date",
        F.when(
            F.col("game_detail_release_date").isNull()
            | (F.trim(F.col("game_detail_release_date")) == ""),
            F.lit("2000-01-01"),
        ).otherwise(F.col("game_detail_release_date")),
    )

    df = df.withColumn(
        "release_year_raw",
        F.year(F.to_date(F.col("game_detail_release_date")))
    )

    df = df.withColumn(
        "release_year_raw",
        F.when(F.col("release_year_raw").isNull(), F.lit(2000)).otherwise(F.col("release_year_raw"))
    )

    # Keep year in a small range so it does not dominate the vector
    df = df.withColumn(
        "release_year_norm",
        ((F.col("release_year_raw") - F.lit(2000)) / F.lit(30.0)).cast("double")
    )

    df = df.withColumn(
        "release_year_norm",
        F.when(F.col("release_year_norm").isNull(), F.lit(0.0))
         .when(F.col("release_year_norm") < 0.0, F.lit(0.0))
         .when(F.col("release_year_norm") > 2.0, F.lit(2.0))
         .otherwise(F.col("release_year_norm"))
    )

    # ------------------------------------------------------
    # TRAIN / TEST SPLIT (leave-one-out)
    # ------------------------------------------------------
    print("\nCreating train/test split...")

    window = Window.partitionBy("user_steamid").orderBy(
        F.desc("game_playtime_forever"),
        F.asc("game_appid")
    )

    df = df.withColumn("rn", F.row_number().over(window))

    test_df = df.filter(F.col("rn") == 1).drop("rn")
    train_df = df.filter(F.col("rn") > 1).drop("rn")

    print("Train:", train_df.count())
    print("Test:", test_df.count())

    if SAVE_INTERMEDIATE_PARQUET:
        print(f"Saving intermediate train parquet to {INTERMEDIATE_PATH}")
        train_df.write.mode("overwrite").parquet(INTERMEDIATE_PATH)

    # ------------------------------------------------------
    # FEATURE PIPELINE
    # IMPORTANT: fit on train only (no leakage)
    # We use FeatureHasher to keep the vector compact.
    # ------------------------------------------------------
    print("\nBuilding feature pipeline (fit on train only)...")

    hasher = FeatureHasher(
        inputCols=["game_detail_developer", "game_detail_publisher"],
        outputCol="cat_hash_vec",
        numFeatures=32
    )

    assembler = VectorAssembler(
        inputCols=[
            "features",               # existing 100-dim sparse game features
            "cat_hash_vec",           # compact hashed categorical features
            "combined_rating_norm",   # numeric
            "steam_rating_norm",      # numeric
            "release_year_norm",      # numeric
        ],
        outputCol="features_v4",
    )

    pipeline = Pipeline(stages=[hasher, assembler])

    pipeline_model = pipeline.fit(train_df)

    # We only need the transformed unique games table for embeddings.
    # Interactions can stay slim.
    print("Applying feature pipeline to unique games table...")

    games_source = (
        df.select(
            "game_appid",
            "game_name",
            "game_img_url",
            "features",
            "game_detail_developer",
            "game_detail_publisher",
            "combined_rating_norm",
            "steam_rating_norm",
            "release_year_norm",
        )
        .dropDuplicates(["game_appid"])
    )

    games_transformed = pipeline_model.transform(games_source)

    print("Feature pipeline applied successfully!")

    # ------------------------------------------------------
    # CONVERT INTERACTIONS TO PANDAS (NO LARGE VECTOR COLUMNS!)
    # ------------------------------------------------------
    print("\nConverting needed interaction columns to Pandas...")

    train_pdf = train_df.select(
        "user_steamid",
        "game_appid",
        "game_playtime_forever",
        "combined_rating_norm",
    ).toPandas()

    test_pdf = test_df.select(
        "user_steamid",
        "game_appid",
    ).toPandas()

    print(f"Train interactions loaded: {len(train_pdf):,}")
    print(f"Test interactions loaded: {len(test_pdf):,}")

    # ------------------------------------------------------
    # FILTER WEAK INTERACTIONS (train only)
    # ------------------------------------------------------
    print(f"\nBefore filtering train interactions: {len(train_pdf):,}")
    train_pdf = train_pdf[train_pdf["game_playtime_forever"] >= MIN_PLAYTIME_MINUTES].copy()
    print(f"After filtering (>= {MIN_PLAYTIME_MINUTES} min): {len(train_pdf):,}")

    if len(train_pdf) == 0:
        raise RuntimeError("All training interactions were filtered out. Reduce MIN_PLAYTIME_MINUTES.")

    # ------------------------------------------------------
    # BUILD GAMES TABLE + EMBEDDINGS
    # ------------------------------------------------------
    print("\nConverting games table to Pandas...")

    games_pdf = games_transformed.select(
        "game_appid",
        "game_name",
        "game_img_url",
        "features_v4",
    ).toPandas()

    print(f"Games: {len(games_pdf):,}")

    print("\nChecking features type...")
    print(type(games_pdf["features_v4"].iloc[0]))
    print(games_pdf["features_v4"].iloc[0])

    print("\nBuilding game embeddings...")

    try:
        game_embeddings = np.vstack(games_pdf["features_v4"].apply(lambda x: x.toArray()).values)
        print("Used toArray() method")
    except Exception:
        game_embeddings = np.vstack(games_pdf["features_v4"].apply(safe_to_array).values)
        print("Used safe_to_array() method")

    game_embeddings = game_embeddings.astype(np.float32)
    game_embeddings_norm = normalize(game_embeddings, norm="l2")

    print("Embedding shape:", game_embeddings.shape)

    game_appids = games_pdf["game_appid"].to_numpy()
    game_names = games_pdf["game_name"].fillna("").to_numpy()
    game_urls = games_pdf["game_img_url"].fillna("").to_numpy()

    appid_to_idx = {appid: idx for idx, appid in enumerate(game_appids)}

    # ------------------------------------------------------
    # USER PROFILE
    # ------------------------------------------------------
    def build_profile(user_df: pd.DataFrame):
        vectors = []
        weights = []

        for row in user_df.itertuples(index=False):
            appid = row.game_appid
            idx = appid_to_idx.get(appid)
            if idx is None:
                continue

            vectors.append(game_embeddings[idx])

            playtime = float(getattr(row, "game_playtime_forever", 0.0))
            rating = float(getattr(row, "combined_rating_norm", 1.0))

            weight = np.log1p(max(playtime, 0.0)) * max(rating, 0.1)
            if weight <= 0:
                weight = 1.0

            weights.append(weight)

        if not vectors:
            return None

        vectors = np.asarray(vectors, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        profile = np.average(vectors, axis=0, weights=weights)
        return profile.astype(np.float32)

    # ------------------------------------------------------
    # RECOMMENDER
    # ------------------------------------------------------
    def recommend_user(user_id, user_df: pd.DataFrame, top_k=10):
        profile = build_profile(user_df)
        if profile is None:
            return []

        profile_norm = normalize(profile.reshape(1, -1), norm="l2")
        scores = (profile_norm @ game_embeddings_norm.T).ravel()

        played_appids = set(user_df["game_appid"].tolist())
        played_indices = [appid_to_idx[a] for a in played_appids if a in appid_to_idx]

        if played_indices:
            scores = scores.copy()
            scores[played_indices] = -np.inf

        k = min(top_k, len(scores))
        if k <= 0:
            return []

        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        recommendations = []
        for idx in top_idx:
            if not np.isfinite(scores[idx]):
                continue

            recommendations.append(
                {
                    "user_steamid": user_id,
                    "game_appid": int(game_appids[idx]),
                    "game_name": game_names[idx],
                    "game_img_url": game_urls[idx],
                    "score": float(scores[idx]),
                }
            )

            if len(recommendations) >= top_k:
                break

        return recommendations

    def recommend_new_user(selected_appids, top_k=10):
        vectors = []
        for appid in selected_appids:
            idx = appid_to_idx.get(appid)
            if idx is not None:
                vectors.append(game_embeddings[idx])

        if not vectors:
            return pd.DataFrame(columns=["game_appid", "game_name", "game_img_url", "score"])

        profile = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
        profile_norm = normalize(profile.reshape(1, -1), norm="l2")
        scores = (profile_norm @ game_embeddings_norm.T).ravel()

        selected_set = set(selected_appids)
        selected_indices = [appid_to_idx[a] for a in selected_set if a in appid_to_idx]

        if selected_indices:
            scores = scores.copy()
            scores[selected_indices] = -np.inf

        k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        rows = []
        for idx in top_idx:
            if not np.isfinite(scores[idx]):
                continue

            rows.append(
                {
                    "game_appid": int(game_appids[idx]),
                    "game_name": game_names[idx],
                    "game_img_url": game_urls[idx],
                    "score": float(scores[idx]),
                }
            )

            if len(rows) >= top_k:
                break

        return pd.DataFrame(rows)

    # ------------------------------------------------------
    # EVALUATION SETUP
    # ------------------------------------------------------
    print("\nPreparing user groups...")

    user_groups = {
        uid: g
        for uid, g in train_pdf.groupby("user_steamid", sort=False)
    }

    test_groups = {
        uid: set(g["game_appid"].tolist())
        for uid, g in test_pdf.groupby("user_steamid", sort=False)
    }

    # Evaluate only users that still have training history after filtering
    eval_user_ids = [uid for uid in user_groups.keys() if uid in test_groups]

    print(f"Total users to process: {len(eval_user_ids):,}")

    precisions = []
    recalls = []
    ndcgs = []
    all_results = []

    # ------------------------------------------------------
    # EVALUATION LOOP
    # ------------------------------------------------------
    for i, user_id in enumerate(
        tqdm(eval_user_ids, total=len(eval_user_ids), desc="Evaluating users", unit="users")
    ):
        user_df = user_groups[user_id]
        recs = recommend_user(user_id, user_df, TOP_K)

        if not recs:
            continue

        rec_ids = [r["game_appid"] for r in recs]
        relevant = test_groups.get(user_id, set())

        precisions.append(precision_at_k(rec_ids, relevant, TOP_K))
        recalls.append(recall_at_k(rec_ids, relevant, TOP_K))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, TOP_K))

        for rank, rec in enumerate(recs, start=1):
            all_results.append(
                {
                    "user_steamid": user_id,
                    "rank": rank,
                    "game_appid": rec["game_appid"],
                    "game_name": rec["game_name"],
                    "game_img_url": rec["game_img_url"],
                    "score": rec["score"],
                }
            )

        if (i + 1) % 1000 == 0:
            print("\n" + "=" * 50)
            print(f"Users processed: {i + 1:,}")
            print(f"Current Precision@10: {np.mean(precisions):.4f}")
            print(f"Current Recall@10: {np.mean(recalls):.4f}")
            print(f"Current NDCG@10: {np.mean(ndcgs):.4f}")
            print("=" * 50 + "\n")

    # ------------------------------------------------------
    # FINAL METRICS
    # ------------------------------------------------------
    print("\n" + "=" * 50)
    print("FINAL METRICS")
    print("=" * 50)

    if precisions:
        print(f"Precision@10 = {np.mean(precisions):.4f} (±{np.std(precisions):.4f})")
        print(f"Recall@10    = {np.mean(recalls):.4f} (±{np.std(recalls):.4f})")
        print(f"NDCG@10      = {np.mean(ndcgs):.4f} (±{np.std(ndcgs):.4f})")
    else:
        print("No users were evaluated.")

    print(f"\nTotal users evaluated: {len(precisions):,}")

    # ------------------------------------------------------
    # SAVE RECOMMENDATIONS
    # ------------------------------------------------------
    print("\nSaving recommendations...")

    recs_df = pd.DataFrame(all_results)
    recs_df.to_parquet(RECS_PATH, index=False)

    print(f"Recommendations saved: {len(recs_df):,}")
    print(f"Saved to: {RECS_PATH}")

    # ------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------
    print("\nSaving model...")

    try:
        pipeline_model.write().overwrite().save(PIPELINE_PATH)
        print(f"Spark pipeline saved to: {PIPELINE_PATH}")
    except Exception as e:
        print(f"Warning: failed to save Spark pipeline: {e}")

    model = {
        "games": games_pdf[["game_appid", "game_name", "game_img_url"]].copy(),
        "game_embeddings": game_embeddings,
        "game_embeddings_norm": game_embeddings_norm,
        "appid_to_idx": appid_to_idx,
        "feature_dim": int(game_embeddings.shape[1]),
        "top_k": TOP_K,
        "min_playtime_minutes": MIN_PLAYTIME_MINUTES,
        "pipeline_path": PIPELINE_PATH,
    }

    joblib.dump(model, MODEL_PATH, compress=3)

    print(f"Model saved: {MODEL_PATH}")
    print("=" * 50)
    print("DONE!")
    print("=" * 50)

    # ------------------------------------------------------
    # STREAMLIT HELPERS (available if imported)
    # ------------------------------------------------------
    return model, recommend_new_user


if __name__ == "__main__":
    main()
