import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

# ─── 0. PATHS ────────────────────────────────────────────────────────────────
RATINGS_PATH = "../ml-1m/ratings.dat"
MOVIES_PATH  = "../ml-1m/movies.dat"
USERS_PATH   = "../ml-1m/users.dat"

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
ratings = pd.read_csv(RATINGS_PATH, sep="::", engine="python",
                      names=["user_id", "movie_id", "rating", "timestamp"])

movies = pd.read_csv(MOVIES_PATH, sep="::", engine="python",
                     names=["movie_id", "title", "genres"],
                     encoding="latin-1")

users = pd.read_csv(USERS_PATH, sep="::", engine="python",
                    names=["user_id", "gender", "age", "occupation", "zip"])

print("=== DATASET LOADED ===")
print(f"Ratings : {ratings.shape[0]:,} rows")
print(f"Movies  : {movies.shape[0]:,} rows")
print(f"Users   : {users.shape[0]:,} rows")
print()

# ─── 2. DATA PREPARATION ─────────────────────────────────────────────────────
# Check missing values
print("=== MISSING VALUES ===")
print("Ratings :", ratings.isnull().sum().sum())
print("Movies  :", movies.isnull().sum().sum())
print("Users   :", users.isnull().sum().sum())
print()

# Drop timestamp (not needed for CF)
ratings.drop(columns=["timestamp"], inplace=True)

# Verify rating range
assert ratings["rating"].between(1, 5).all(), "Unexpected rating values"
print(f"Rating range: {ratings['rating'].min()} – {ratings['rating'].max()}")
print(f"Unique users : {ratings['user_id'].nunique():,}")
print(f"Unique movies: {ratings['movie_id'].nunique():,}")
print()

# Sparsity
n_users  = ratings["user_id"].nunique()
n_movies = ratings["movie_id"].nunique()
sparsity = 1 - len(ratings) / (n_users * n_movies)
print(f"Matrix sparsity: {sparsity:.4f} ({sparsity*100:.1f}% empty)")
print()

# ─── 3. SUMMARY STATISTICS ───────────────────────────────────────────────────
print("=== SUMMARY STATISTICS ===")
print(ratings["rating"].describe())
print()

# ─── 4. EDA — VISUALIZATIONS ─────────────────────────────────────────────────
sns.set_style("darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("MovieLens 1M — Exploratory Data Analysis", fontsize=16, fontweight="bold")

# 4a. Rating distribution
sns.countplot(x="rating", data=ratings, palette="viridis", ax=axes[0, 0])
axes[0, 0].set_title("Rating Distribution")
axes[0, 0].set_xlabel("Rating")
axes[0, 0].set_ylabel("Count")

# 4b. Ratings per user (log scale)
ratings_per_user = ratings.groupby("user_id")["rating"].count()
axes[0, 1].hist(ratings_per_user, bins=50, color="steelblue", edgecolor="white")
axes[0, 1].set_title("Ratings per User")
axes[0, 1].set_xlabel("Number of Ratings")
axes[0, 1].set_ylabel("Number of Users")
axes[0, 1].set_yscale("log")

# 4c. Ratings per movie (log scale)
ratings_per_movie = ratings.groupby("movie_id")["rating"].count()
axes[1, 0].hist(ratings_per_movie, bins=50, color="coral", edgecolor="white")
axes[1, 0].set_title("Ratings per Movie")
axes[1, 0].set_xlabel("Number of Ratings")
axes[1, 0].set_ylabel("Number of Movies")
axes[1, 0].set_yscale("log")

# 4d. Average rating per movie (top 20 most rated)
top_movies = ratings_per_movie.nlargest(20).index
avg_rating_top = (ratings[ratings["movie_id"].isin(top_movies)]
                  .groupby("movie_id")["rating"].mean()
                  .sort_values(ascending=False))
axes[1, 1].barh(avg_rating_top.index.astype(str), avg_rating_top.values, color="mediumpurple")
axes[1, 1].set_title("Avg Rating — Top 20 Most Rated Movies")
axes[1, 1].set_xlabel("Average Rating")
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.show()
print("EDA plots saved → eda_plots.png")
print()

# 4e. Genre distribution
genre_counts = (movies["genres"].str.split("|")
                .explode()
                .value_counts())
plt.figure(figsize=(12, 5))
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="mako")
plt.title("Genre Distribution")
plt.xlabel("Number of Movies")
plt.tight_layout()
plt.savefig("genre_distribution.png", dpi=150)
plt.show()
print("Genre plot saved → genre_distribution.png")
print()

# ─── 5. TRAIN / TEST SPLIT ───────────────────────────────────────────────────
# Surprise reader expects ratings in [1, 5]
reader  = Reader(rating_scale=(1, 5))
data    = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)

# 80/20 split — stratified by surprise's internal shuffle
trainset, testset = train_test_split(data, test_size=0.20, random_state=42)

print(f"Trainset size : {trainset.n_ratings:,} ratings")
print(f"Testset  size : {len(testset):,} ratings")
print()

# ─── 6. BASELINE MODEL — User-Based CF (Pearson) ─────────────────────────────
print("=== BASELINE: User-Based CF (Pearson Correlation) ===")

sim_options = {
    "name"          : "pearson",
    "user_based"    : True,
    "min_support"   : 3,
}

baseline_model = KNNBasic(k=40, sim_options=sim_options)
baseline_model.fit(trainset)

predictions = baseline_model.test(testset)

rmse = accuracy.rmse(predictions, verbose=False)
mae  = accuracy.mae(predictions,  verbose=False)

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print()

# ─── 7. 5-FOLD CROSS VALIDATION ──────────────────────────────────────────────
print("=== 5-FOLD CROSS VALIDATION ===")
cv_results = cross_validate(
    KNNBasic(k=40, sim_options=sim_options),
    data,
    measures=["RMSE", "MAE"],
    cv=5,
    verbose=True
)

print(f"\nMean RMSE : {np.mean(cv_results['test_rmse']):.4f}")
print(f"Mean MAE  : {np.mean(cv_results['test_mae']):.4f}")
print()

# ─── 8. INITIAL OBSERVATIONS ─────────────────────────────────────────────────
print("=== INITIAL OBSERVATIONS ===")
print(f"Matrix sparsity ({sparsity*100:.1f}%) causes unreliable Pearson similarity.")
print(f"Users with few ratings produce low-quality neighbourhoods.")
print(f"Cold start: new users have no vector → no prediction possible.")
print(f"Next phase: SVD (Matrix Factorization) to address sparsity + cold start fallback via CBF.")