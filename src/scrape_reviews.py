"""Step 2: collect recent Play Store reviews for Zepto, Blinkit and Swiggy Instamart.

Run from the voc-miner folder:  python src/scrape_reviews.py
Output: data/reviews.csv
"""
from google_play_scraper import reviews, Sort
import pandas as pd

# Package id = the text after "id=" in the app's Play Store link.
# Instamart has no standalone Android app, so its reviews live inside the
# main Swiggy app. We pull more Swiggy reviews and keep only Instamart ones.
APPS = {
    "Zepto":     {"pkg": "com.zeptoconsumerapp",    "count": 4000, "keywords": None},
    "Blinkit":   {"pkg": "com.grofers.customerapp", "count": 4000, "keywords": None},
    "Instamart": {"pkg": "in.swiggy.android",       "count": 40000,
                  "keywords": r"insta ?mart|grocer|vegetable|fruit|milk|dairy|atta|"
                              r"\begg|bread|onion|tomato|expired|rotten"},
}

rows = []
for name, cfg in APPS.items():
    try:
        data, _ = reviews(cfg["pkg"], lang="en", country="in",
                          sort=Sort.NEWEST, count=cfg["count"])
    except Exception as e:
        print(f"{name}: FAILED ({e}) - check the package id")
        continue
    app_df = pd.DataFrame([{
        "app": name,
        "text": r["content"],
        "rating": r["score"],
        "date": r["at"],
        "thumbs_up": r["thumbsUpCount"],
    } for r in data])
    pulled = len(app_df)
    if cfg["keywords"]:
        app_df = app_df[app_df.text.fillna("").str.contains(cfg["keywords"], case=False)]
        print(f"{name}: {pulled} Swiggy reviews pulled, {len(app_df)} are about Instamart")
    else:
        print(f"{name}: {pulled} reviews")
    rows.append(app_df)

df = pd.concat(rows, ignore_index=True)
before = len(df)
df = df[df.text.fillna("").str.len() > 15].drop_duplicates("text")
print(f"\nKept {len(df)} of {before} reviews after removing short and duplicate ones")
print(df.groupby("app").size().to_string())
print(f"\nDate range: {df.date.min():%d %b %Y} to {df.date.max():%d %b %Y}")

df.to_csv("data/reviews.csv", index=False)
print("Saved to data/reviews.csv")
