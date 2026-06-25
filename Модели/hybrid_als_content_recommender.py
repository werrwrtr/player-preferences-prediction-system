

import os
import sys
import glob
import traceback
import joblib
import numpy as np
import pandas as pd

# IMPORTANT: set Python executable before pyspark import on Windows
PYTHON_EXE = sys.executable
os.environ["PYSPARK_PYTHON"] = PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXE

from tqdm import tqdm
from sklearn.preprocessing import normalize
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALSModel


# ==========================================================
# CONFIG
# ==========================================================

BASE_PATH = r"D:\steam_recommender_system\3. data_analysis\db"

CONTENT_MODEL_CANDIDATES = [
    "content_based_model_v4.pkl",
    "content_based_model_v3.pkl",
    "content_based_model_v2.pkl",
]

ALS_MODEL_CANDIDATES = [
    "als_model_fixed",
    "als_model_improved",
    "als_model_v2",
]

ALS_MAPPINGS_CANDIDATES = [
    "als_mappings_fixed.pkl",
    "als_mappings_improved.pkl",
    "als_mappings_v2.pkl",
]

OUT_RECS_PATH = "hybrid_top10_701515.parquet"
OUT_META_PATH = "hybrid_meta_701515.pkl"

TOP_K = 10
ALS_CANDIDATES_K = 100
CONTENT_CANDIDATES_K = 100

# Blend weights
W_ALS = 0.60
W_CONTENT = 0.30
W_POP = 0.10

# Content profile weighting
MIN_PLAYTIME_FOR_CONTENT = 60

# 70/15/15 split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MIN_USER_INTERACTIONS = 5
MAX_USERS_DEBUG = None  # e.g. 2000


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
# HELPERS
# ==========================================================

def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def latest_dataset_folder(base_path: str) -> str:
    folders = glob.glob(os.path.join(base_path, "final_dataset_*"))
    if not folders:
        raise FileNotFoundError(f"No final_dataset_* folders found in {base_path}")
    return max(folders, key=os.path.getmtime)


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


def normalize_minmax(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    vmin = np.min(values)
    vmax = np.max(values)
    denom = max(float(vmax - vmin), 1e-9)
    return (values - vmin) / denom


def compute_content_profile(user_hist_df, appid_to_idx, game_embeddings):
    vectors = []
    weights = []

    for row in user_hist_df.itertuples(index=False):
        appid = int(row.game_appid)
        idx = appid_to_idx.get(appid)
        if idx is None:
            continue

        playtime = float(getattr(row, "game_playtime_forever", 0.0))
        if playtime < MIN_PLAYTIME_FOR_CONTENT:
            continue

        rating = float(getattr(row, "combined_rating", 1.0)) if hasattr(row, "combined_rating") else 1.0
        weight = np.log1p(max(playtime, 0.0)) * max(rating, 0.1)
        if weight <= 0:
            weight = 1.0

        vectors.append(game_embeddings[idx])
        weights.append(weight)

    if len(vectors) == 0:
        return None

    vectors = np.asarray(vectors, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    if weights.sum() <= 0:
        weights = np.ones(len(vectors), dtype=np.float32)

    profile = np.average(vectors, axis=0, weights=weights)
    return profile.astype(np.float32)


def recommend_popular(popular_df, played_appids, top_k=10):
    recs = []
    for row in popular_df.itertuples(index=False):
        appid = int(row.game_appid)
        if appid in played_appids:
            continue
        recs.append((appid, float(row.popularity)))
        if len(recs) >= top_k:
            break
    return recs


def split_user_history_701515(user_pdf: pd.DataFrame):
    """
    Deterministic per-user split by playtime desc.
    Returns train/val/test dataframes for a single user.
    """
    user_pdf = user_pdf.sort_values(
        ["game_playtime_forever", "game_appid"],
        ascending=[False, True]
    ).reset_index(drop=True)

    n = len(user_pdf)
    if n < MIN_USER_INTERACTIONS:
        return None, None, None

    train_end = int(np.floor(n * TRAIN_RATIO))
    val_end = int(np.floor(n * (TRAIN_RATIO + VAL_RATIO)))

    if train_end <= 0:
        train_end = 1
    if val_end <= train_end and n >= 3:
        val_end = min(train_end + 1, n - 1)

    train_part = user_pdf.iloc[:train_end].copy()
    val_part = user_pdf.iloc[train_end:val_end].copy()
    test_part = user_pdf.iloc[val_end:].copy()

    if len(val_part) == 0 and n >= 2:
        val_part = user_pdf.iloc[train_end:train_end+1].copy()
        test_part = user_pdf.iloc[train_end+1:].copy()

    if len(test_part) == 0 and n >= 2:
        test_part = user_pdf.iloc[-1:].copy()

    return train_part, val_part, test_part


def get_user_idx(user_to_idx, user_id):
    """
    Robust lookup for string/int key variants.
    """
    if user_id in user_to_idx:
        return user_to_idx[user_id]
    try:
        if str(user_id) in user_to_idx:
            return user_to_idx[str(user_id)]
    except Exception:
        pass
    try:
        if int(user_id) in user_to_idx:
            return user_to_idx[int(user_id)]
    except Exception:
        pass
    return None


# ==========================================================
# LOAD MODELS
# ==========================================================

content_model_path = find_existing_path(CONTENT_MODEL_CANDIDATES)
als_model_path = find_existing_path(ALS_MODEL_CANDIDATES)
als_mappings_path = find_existing_path(ALS_MAPPINGS_CANDIDATES)

print("=" * 80)
print("LOADING MODELS")
print("=" * 80)
print("Content model:", content_model_path)
print("ALS model:", als_model_path)
print("ALS mappings:", als_mappings_path)

content_model = joblib.load(content_model_path)
als_mappings = joblib.load(als_mappings_path)

content_games = content_model["games"].copy()
game_embeddings = content_model["game_embeddings"].astype(np.float32)
game_embeddings_norm = content_model.get("game_embeddings_norm")
if game_embeddings_norm is None:
    game_embeddings_norm = normalize(game_embeddings, norm="l2")

appid_to_idx = content_model.get("appid_to_idx")
if appid_to_idx is None:
    appid_to_idx = {int(appid): idx for idx, appid in enumerate(content_games["game_appid"].tolist())}

game_appids = content_games["game_appid"].astype(int).to_numpy()

user_to_idx = als_mappings["user_to_idx"]
idx_to_user = als_mappings["idx_to_user"]
idx_to_item = als_mappings["idx_to_item"]


# ==========================================================
# SPARK SESSION
# ==========================================================

spark = (
    SparkSession.builder
    .appName("HybridALSFixed")
    .config("spark.pyspark.python", PYTHON_EXE)
    .config("spark.pyspark.driver.python", PYTHON_EXE)
    .config("spark.executorEnv.PYSPARK_PYTHON", PYTHON_EXE)
    .config("spark.executorEnv.PYSPARK_DRIVER_PYTHON", PYTHON_EXE)
    .config("spark.driver.memory", "16g")
    .config("spark.executor.memory", "16g")
    .config("spark.driver.maxResultSize", "4g")
    .config("spark.sql.shuffle.partitions", "400")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ==========================================================
# LOAD ALS MODEL
# ==========================================================

print("\nLoading ALS model...")
als_model = ALSModel.load(als_model_path)
print("ALS model loaded.")


# ==========================================================
# LOAD DATA
# ==========================================================

print("\n" + "=" * 80)
print("LOADING DATASET")
print("=" * 80)

latest_folder = latest_dataset_folder(BASE_PATH)
print("Loading:", latest_folder)

df = spark.read.parquet(latest_folder)

needed_cols = [
    "user_steamid",
    "game_appid",
    "game_playtime_forever",
    "combined_rating",
]
existing_cols = [c for c in needed_cols if c in df.columns]
df = df.select(*existing_cols)

df = df.filter(
    F.col("user_steamid").isNotNull() &
    F.col("game_appid").isNotNull()
)

# Remove sparse users before split
user_counts = df.groupBy("user_steamid").count().withColumnRenamed("count", "user_cnt")
df = (
    df.join(user_counts, on="user_steamid", how="inner")
      .filter(F.col("user_cnt") >= F.lit(MIN_USER_INTERACTIONS))
      .drop("user_cnt")
)

print("Rows after user filter:", df.count())

# Keep only columns needed for per-user split
pdf = df.select(
    "user_steamid",
    "game_appid",
    "game_playtime_forever",
    "combined_rating",
).toPandas()

train_parts = []
val_parts = []
test_parts = []

print("\nBuilding 70/15/15 split per user...")
for uid, group in tqdm(pdf.groupby("user_steamid", sort=False), desc="Splitting users"):
    tr, va, te = split_user_history_701515(group)
    if tr is None:
        continue
    train_parts.append(tr)
    if len(va) > 0:
        val_parts.append(va)
    if len(te) > 0:
        test_parts.append(te)

train_pdf = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=pdf.columns)
val_pdf = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=pdf.columns)
test_pdf = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=pdf.columns)

print("Train rows:", len(train_pdf))
print("Val rows:", len(val_pdf))
print("Test rows:", len(test_pdf))

# Small pandas tables for histories/truth/fallbacks
train_histories = {
    uid: g.reset_index(drop=True)
    for uid, g in train_pdf.groupby("user_steamid", sort=False)
}

val_truth = {
    uid: set(g["game_appid"].tolist())
    for uid, g in val_pdf.groupby("user_steamid", sort=False)
}

test_truth = {
    uid: set(g["game_appid"].tolist())
    for uid, g in test_pdf.groupby("user_steamid", sort=False)
}

eval_users_val = [uid for uid in train_histories.keys() if uid in val_truth]
eval_users_test = [uid for uid in train_histories.keys() if uid in test_truth]

print("Users with val:", len(eval_users_val))
print("Users with test:", len(eval_users_test))

# Spark dataframes only for popularity and ALS candidate prep
train_sdf = spark.createDataFrame(train_pdf)
val_sdf = spark.createDataFrame(val_pdf) if len(val_pdf) > 0 else spark.createDataFrame(pd.DataFrame(columns=train_pdf.columns))
test_sdf = spark.createDataFrame(test_pdf) if len(test_pdf) > 0 else spark.createDataFrame(pd.DataFrame(columns=train_pdf.columns))


# ==========================================================
# POPULARITY FALLBACK
# ==========================================================

popular_pdf = (
    train_sdf.groupBy("game_appid")
    .agg(
        F.countDistinct("user_steamid").alias("users_count"),
        F.sum(F.log1p(F.col("game_playtime_forever").cast("double"))).alias("playtime_sum")
    )
    .withColumn(
        "popularity_raw",
        F.log1p(F.col("users_count").cast("double")) + F.log1p(F.col("playtime_sum").cast("double"))
    )
    .select("game_appid", "popularity_raw")
    .toPandas()
)

if len(popular_pdf) > 0:
    popular_pdf["popularity"] = normalize_minmax(popular_pdf["popularity_raw"].values)
    popular_pdf = popular_pdf.sort_values("popularity", ascending=False).reset_index(drop=True)
else:
    popular_pdf = pd.DataFrame(columns=["game_appid", "popularity_raw", "popularity"])


# ==========================================================
# PRECOMPUTE ALS CANDIDATES
# ==========================================================

print("\nPrecomputing ALS top candidates for all known ALS users...")
als_recs_df = als_model.recommendForAllUsers(ALS_CANDIDATES_K)

als_candidates = {}
for row in tqdm(als_recs_df.toLocalIterator(), desc="Collecting ALS candidates"):
    user_idx = int(row["user"])
    recs = []
    for x in row["recommendations"]:
        item_idx = int(x["item"])
        score = float(x["rating"])
        recs.append((item_idx, score))
    als_candidates[user_idx] = recs

print("ALS candidates ready:", len(als_candidates))


# ==========================================================
# RECOMMENDATION FUNCTIONS
# ==========================================================

def recommend_content_only(user_id, user_hist_df, top_k=10):
    played = set(user_hist_df["game_appid"].astype(int).tolist())
    profile = compute_content_profile(user_hist_df, appid_to_idx, game_embeddings)

    if profile is None:
        return recommend_popular(popular_pdf, played, top_k)

    profile_norm = profile / (np.linalg.norm(profile) + 1e-9)
    scores = (profile_norm.reshape(1, -1) @ game_embeddings_norm.T).ravel()

    played_idx = [appid_to_idx[a] for a in played if a in appid_to_idx]
    if played_idx:
        scores = scores.copy()
        scores[played_idx] = -np.inf

    k = min(top_k, len(scores))
    if k <= 0:
        return []

    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    out = []
    for idx in top_idx:
        if not np.isfinite(scores[idx]):
            continue
        out.append((int(game_appids[idx]), float(scores[idx])))
        if len(out) >= top_k:
            break
    return out


def recommend_als_only(user_id, user_hist_df=None, top_k=10):
    user_idx = get_user_idx(user_to_idx, user_id)
    if user_idx is None:
        return []

    recs = als_candidates.get(int(user_idx), [])
    out = []
    for item_idx, score in recs:
        appid = idx_to_item.get(int(item_idx))
        if appid is None:
            continue
        out.append((int(appid), float(score)))
        if len(out) >= top_k:
            break
    return out


def recommend_hybrid(user_id, user_hist_df, top_k=10):
    played = set(user_hist_df["game_appid"].astype(int).tolist())

    profile = compute_content_profile(user_hist_df, appid_to_idx, game_embeddings)
    if profile is None:
        return recommend_popular(popular_pdf, played, top_k)

    profile_norm = profile / (np.linalg.norm(profile) + 1e-9)
    content_scores = (profile_norm.reshape(1, -1) @ game_embeddings_norm.T).ravel()

    user_idx = get_user_idx(user_to_idx, user_id)
    als_recs = als_candidates.get(int(user_idx), []) if user_idx is not None else []

    # Candidate union
    k = min(CONTENT_CANDIDATES_K, len(content_scores))
    if k <= 0:
        return recommend_popular(popular_pdf, played, top_k)

    content_top_idx = np.argpartition(content_scores, -k)[-k:]
    content_top_idx = content_top_idx[np.argsort(content_scores[content_top_idx])[::-1]]

    candidate_appids = set(int(game_appids[i]) for i in content_top_idx)
    for item_idx, _ in als_recs:
        appid = idx_to_item.get(int(item_idx))
        if appid is not None:
            candidate_appids.add(int(appid))

    candidate_appids = [a for a in candidate_appids if a not in played and a in appid_to_idx]

    if not candidate_appids:
        return recommend_popular(popular_pdf, played, top_k)

    cb_raw = np.array([float(content_scores[appid_to_idx[a]]) for a in candidate_appids], dtype=np.float32)

    als_map = {}
    for item_idx, score in als_recs:
        appid = idx_to_item.get(int(item_idx))
        if appid is not None:
            als_map[int(appid)] = float(score)

    als_raw = np.array([float(als_map.get(a, 0.0)) for a in candidate_appids], dtype=np.float32)

    pop_map = {int(r.game_appid): float(r.popularity) for r in popular_pdf.itertuples(index=False)} if len(popular_pdf) > 0 else {}
    pop_raw = np.array([float(pop_map.get(a, 0.0)) for a in candidate_appids], dtype=np.float32)

    cb_norm = normalize_minmax(cb_raw)
    als_norm = normalize_minmax(als_raw)
    pop_norm = normalize_minmax(pop_raw)

    final_scores = (W_ALS * als_norm) + (W_CONTENT * cb_norm) + (W_POP * pop_norm)

    order = np.argsort(final_scores)[::-1]
    out = []
    for pos in order:
        appid = int(candidate_appids[pos])
        out.append((appid, float(final_scores[pos])))
        if len(out) >= top_k:
            break

    return out


# ==========================================================
# EVALUATION
# ==========================================================

def eval_recommender(recommender_fn, label, users, truth_map, pass_user_id=True):
    precisions, recalls, ndcgs = [], [], []
    rec_rows = []

    print(f"\nEvaluating: {label}")

    for i, uid in enumerate(tqdm(users, desc=f"{label} users", unit="users")):
        hist = train_histories[uid]
        rel = truth_map.get(uid, set())
        if not rel:
            continue

        recs = recommender_fn(uid, hist, TOP_K) if pass_user_id else recommender_fn(uid, hist, TOP_K)
        rec_ids = [appid for appid, _ in recs]

        precisions.append(precision_at_k(rec_ids, rel, TOP_K))
        recalls.append(recall_at_k(rec_ids, rel, TOP_K))
        ndcgs.append(ndcg_at_k(rec_ids, rel, TOP_K))

        for rank, (appid, score) in enumerate(recs, start=1):
            rec_rows.append({
                "user_steamid": uid,
                "rank": rank,
                "game_appid": appid,
                "score": score,
                "model": label
            })

        if (i + 1) % 1000 == 0:
            print("\n" + "=" * 50)
            print(f"{label} | Users processed: {i + 1:,}")
            print(f"Current Precision@10: {np.mean(precisions):.4f}")
            print(f"Current Recall@10: {np.mean(recalls):.4f}")
            print(f"Current NDCG@10: {np.mean(ndcgs):.4f}")
            print("=" * 50 + "\n")

    metrics = {
        "precision_at_10": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "evaluated_users": int(len(precisions)),
    }

    print("\n" + "=" * 50)
    print(f"FINAL METRICS: {label}")
    print("=" * 50)
    print(f"Precision@10 = {metrics['precision_at_10']:.4f}")
    print(f"Recall@10    = {metrics['recall_at_10']:.4f}")
    print(f"NDCG@10      = {metrics['ndcg_at_10']:.4f}")
    print(f"Total users evaluated: {metrics['evaluated_users']:,}")

    return metrics, pd.DataFrame(rec_rows)


content_val_metrics, content_val_recs = eval_recommender(
    recommend_content_only,
    "content_val",
    eval_users_val,
    val_truth,
    pass_user_id=True
)

als_val_metrics, als_val_recs = eval_recommender(
    recommend_als_only,
    "als_val",
    eval_users_val,
    val_truth,
    pass_user_id=True
)

hybrid_val_metrics, hybrid_val_recs = eval_recommender(
    recommend_hybrid,
    "hybrid_val",
    eval_users_val,
    val_truth,
    pass_user_id=True
)

content_test_metrics, content_test_recs = eval_recommender(
    recommend_content_only,
    "content_test",
    eval_users_test,
    test_truth,
    pass_user_id=True
)

als_test_metrics, als_test_recs = eval_recommender(
    recommend_als_only,
    "als_test",
    eval_users_test,
    test_truth,
    pass_user_id=True
)

hybrid_test_metrics, hybrid_test_recs = eval_recommender(
    recommend_hybrid,
    "hybrid_test",
    eval_users_test,
    test_truth,
    pass_user_id=True
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

print("\nSaving outputs...")

hybrid_test_recs.to_parquet(OUT_RECS_PATH, index=False)

meta = {
    "content_model_path": content_model_path,
    "als_model_path": als_model_path,
    "als_mappings_path": als_mappings_path,
    "split": {
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "min_user_interactions": MIN_USER_INTERACTIONS,
    },
    "weights": {
        "w_als": W_ALS,
        "w_content": W_CONTENT,
        "w_pop": W_POP,
    },
    "candidate_sizes": {
        "als_candidates_k": ALS_CANDIDATES_K,
        "content_candidates_k": CONTENT_CANDIDATES_K,
    },
    "filters": {
        "min_playtime_for_content": MIN_PLAYTIME_FOR_CONTENT,
    },
    "metrics": {
        "content_val": content_val_metrics,
        "als_val": als_val_metrics,
        "hybrid_val": hybrid_val_metrics,
        "content_test": content_test_metrics,
        "als_test": als_test_metrics,
        "hybrid_test": hybrid_test_metrics,
    },
}

joblib.dump(meta, OUT_META_PATH, compress=3)

print(f"Hybrid recommendations saved: {len(hybrid_test_recs):,} -> {OUT_RECS_PATH}")
print(f"Meta saved: {OUT_META_PATH}")

print("\nSUMMARY")
print("Content VAL :", content_val_metrics)
print("ALS VAL     :", als_val_metrics)
print("Hybrid VAL  :", hybrid_val_metrics)
print("Content TEST:", content_test_metrics)
print("ALS TEST    :", als_test_metrics)
print("Hybrid TEST :", hybrid_test_metrics)

print("\nDONE!")


if __name__ == "__main__":
    try:
        pass
    finally:
        try:
            spark.stop()
        except Exception:
            pass
