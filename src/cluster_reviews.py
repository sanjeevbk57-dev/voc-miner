"""Step 3 of the guide: turn reviews into vectors and group them by problem.

Run from the voc-miner folder:  python src/cluster_reviews.py
Input:  data/reviews.csv
Output: data/reviews_clustered.csv
"""
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

MIN_CLUSTER_SIZE = 25   # smaller = more, narrower groups; larger = fewer, broader groups

df = pd.read_csv("data/reviews.csv")
print(f"Loaded {len(df)} reviews")

# 1. Embed: each review becomes a vector; similar complaints land close together.
#    The model downloads once (~90 MB) the first time you run this.
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(df.text.tolist(), normalize_embeddings=True,
                   show_progress_bar=True, batch_size=64)

# 2. Cluster: HDBSCAN finds the groups itself. -1 means "noise" (didn't fit any group).
df["cluster"] = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE).fit_predict(emb)

# 3. Report
n_clusters = df.cluster.max() + 1
noise = (df.cluster == -1).mean()
print(f"\nFound {n_clusters} clusters, noise rate {noise:.0%}")

sizes = (df[df.cluster >= 0].groupby("cluster")
         .agg(reviews=("text", "size"), avg_rating=("rating", "mean"))
         .sort_values("reviews", ascending=False))

print("\nLargest clusters, with 3 sample reviews each:")
for cid, row in sizes.head(10).iterrows():
    print(f"\n--- Cluster {cid}: {row.reviews} reviews, avg rating {row.avg_rating:.1f}")
    for t in df[df.cluster == cid].text.sample(3, random_state=1):
        print("   •", t[:140].replace("\n", " "))

df.to_csv("data/reviews_clustered.csv", index=False)
print("\nSaved to data/reviews_clustered.csv")
