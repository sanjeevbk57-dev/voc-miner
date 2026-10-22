"""Step 3 of the guide: turn complaint reviews into vectors and group them by problem.

Run from the voc-miner folder:  python src/cluster_reviews.py
Input:  data/reviews.csv
Output: data/reviews_clustered.csv

Version 2. First attempt clustered all reviews directly on 384-dim embeddings:
94% noise and only 2 clusters, both generic praise. Fixes:
  - keep only complaints (rating <= 3), since the question is what frustrates users
  - reduce embeddings to 10 dims with UMAP before HDBSCAN
"""
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import HDBSCAN

MAX_RATING = 3          # treat 1-3 star reviews as complaints
MIN_CLUSTER_SIZE = 15   # smaller = more, narrower groups; larger = fewer, broader groups

df = pd.read_csv("data/reviews.csv")
df = df[df.rating <= MAX_RATING].reset_index(drop=True)
print(f"Complaint reviews (rating <= {MAX_RATING}): {len(df)}")
print(df.groupby("app").size().to_string())

# 1. Embed: each review becomes a 384-number vector.
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(df.text.tolist(), normalize_embeddings=True,
                   show_progress_bar=True, batch_size=64)

# 2. Squash to 10 dims so density-based clustering can find groups.
reduced = UMAP(n_neighbors=15, n_components=10, min_dist=0.0,
               metric="cosine", random_state=42).fit_transform(emb)

# 3. Cluster. -1 means "noise" (didn't fit any group).
df["cluster"] = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                        min_samples=5).fit_predict(reduced)

# 4. Report
n_clusters = df.cluster.max() + 1
noise = (df.cluster == -1).mean()
print(f"\nFound {n_clusters} clusters, noise rate {noise:.0%}")

sizes = (df[df.cluster >= 0].groupby("cluster")
         .agg(reviews=("text", "size"), avg_rating=("rating", "mean"))
         .sort_values("reviews", ascending=False))

print("\nLargest clusters, with 3 sample reviews each:")
for cid, row in sizes.head(12).iterrows():
    print(f"\n--- Cluster {cid}: {int(row.reviews)} reviews, avg rating {row.avg_rating:.1f}")
    for t in df[df.cluster == cid].text.sample(3, random_state=1):
        print("   •", t[:140].replace("\n", " "))

df.to_csv("data/reviews_clustered.csv", index=False)
print("\nSaved to data/reviews_clustered.csv")
