"""
Phase 4 - SVD-Based Collaborative Filtering Recommender System
Dataset: ratings.csv (100,836 ratings | 610 users | 9,724 movies)
Filter: Movies rated >= 100 times (retains 20,188 ratings across 138 movies)
Method: TruncatedSVD (n_components=20) via scikit-learn
RMSE: ~1.04
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from math import sqrt

# ------------------------------------------------------------------
# Step 1: Load and filter dataset
# ------------------------------------------------------------------
ratings_df = pd.read_csv('data/ratings.csv')
print("Raw dataset shape:", ratings_df.shape)
print(ratings_df.head())
print(ratings_df.info())

# Filter: keep only movies rated >= 100 times
movie_counts     = ratings_df.groupby('movieId').size()
movies_qualified = movie_counts[movie_counts >= 100].index
ratings_filtered = ratings_df[ratings_df['movieId'].isin(movies_qualified)].copy()

print(f"\nAfter filtering (>=100 ratings per movie):")
print(f"  Ratings: {len(ratings_filtered):,}")
print(f"  Unique users:  {ratings_filtered['userId'].nunique()}")
print(f"  Unique movies: {ratings_filtered['movieId'].nunique()}")

# ------------------------------------------------------------------
# Step 2: Build utility matrix (users x movies)
# ------------------------------------------------------------------
utility_matrix = ratings_filtered.pivot_table(
    values='rating',
    index='userId',
    columns='movieId',
    fill_value=0,
)

print(f"\nUtility matrix shape: {utility_matrix.shape}")

# ------------------------------------------------------------------
# Step 3: Sparsity analysis
# ------------------------------------------------------------------
total_entries   = utility_matrix.size
filled_entries  = ratings_filtered.shape[0]
missing_entries = total_entries - filled_entries
missing_pct     = (missing_entries / total_entries) * 100

print(f"\nUtility Matrix Sparsity:")
print(f"  Total entries:   {total_entries:,}")
print(f"  Filled entries:  {filled_entries:,}")
print(f"  Missing entries: {missing_entries:,}")
print(f"  Missing %:       {missing_pct:.2f}%")

# ------------------------------------------------------------------
# Step 4: TruncatedSVD collaborative filtering
# ------------------------------------------------------------------
ratings_sparse = csr_matrix(utility_matrix.values)

svd = TruncatedSVD(n_components=20, random_state=42)
matrix_svd = svd.fit_transform(ratings_sparse)

# Reconstruct the full matrix
matrix_reconstructed = np.dot(matrix_svd, svd.components_)
print(f"\nExplained variance ratio (sum, 20 components): "
      f"{svd.explained_variance_ratio_.sum():.4f}")

# ------------------------------------------------------------------
# Step 5: RMSE calculation
# ------------------------------------------------------------------
ratings_diff = ratings_sparse - matrix_reconstructed
rmse = sqrt(np.mean(np.power(ratings_diff, 2)))

print(f"\nSVD Collaborative Filtering RMSE: {rmse:.4f}")
print("Interpretation: On a 0.5-5.0 rating scale, an RMSE of ~1.04 represents")
print("reasonable accuracy for a dimensionality-reduced implicit-feedback model.")

# ------------------------------------------------------------------
# Step 6: Top-5 recommendations for a sample user
# ------------------------------------------------------------------
# Find a user who has rated at least one movie in the filtered set
sample_user_id = ratings_filtered['userId'].sample(1, random_state=7).iloc[0]
print(f"\nGenerating recommendations for userId: {sample_user_id}")

user_idx = utility_matrix.index.get_loc(sample_user_id)
user_pred = matrix_reconstructed[user_idx]

predicted_df = pd.DataFrame(
    user_pred,
    index=utility_matrix.columns,
    columns=['predicted_rating'],
)

# Exclude movies already rated by this user
already_rated = ratings_filtered[ratings_filtered['userId'] == sample_user_id]['movieId']
predicted_df  = predicted_df.loc[~predicted_df.index.isin(already_rated)]

top5 = predicted_df.sort_values(by='predicted_rating', ascending=False).head(5)
print(f"\nTop 5 movie recommendations for userId {sample_user_id}:")
print(top5.to_string())

# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------
print("\n--- Model Summary ---")
print(f"Dataset:         ratings.csv")
print(f"Filter applied:  movies rated >= 100 times")
print(f"Ratings retained:{len(ratings_filtered):,} of {len(ratings_df):,}")
print(f"Matrix shape:    {utility_matrix.shape[0]} users x {utility_matrix.shape[1]} movies")
print(f"Sparsity:        {missing_pct:.2f}% missing")
print(f"SVD components:  20")
print(f"RMSE:            {rmse:.4f}")
print(f"Sample user:     {sample_user_id}")
print(f"Top 5 recs:      {top5.index.tolist()}")
