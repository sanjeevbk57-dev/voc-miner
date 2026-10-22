# Quick-Commerce Voice-of-Customer Miner

**Problem:** Reading thousands of app reviews to find out what users actually hate is slow, so product decisions get made on a handful of anecdotes. This project automates it.

**Question:** What frustrates quick-commerce app users the most, and where is each app (Zepto, Blinkit, Swiggy Instamart) weakest?

## How it works
- Collects recent Google Play Store reviews for each app (public data only)
- Keeps the complaints (1–3 star reviews), embeds them, reduces dimensions with UMAP, and groups them by problem with HDBSCAN
- Uses Claude to name each problem group and pull verbatim quotes, verifies every quote against the raw data, then ranks problems by size × pain and compares apps

```
Play Store reviews → filter complaints → embeddings (MiniLM) → UMAP → HDBSCAN clusters
                  → Claude names clusters + quotes → quote check → ranking + app comparison
```

## Data
Google Play Store reviews (India, English), newest first, collected on 22 Oct 2026.

| App | Reviews pulled | Kept after cleaning |
|---|---|---|
| Zepto | 4,000 | 2,219 |
| Blinkit | 4,000 | 1,706 |
| Swiggy Instamart | 40,000 Swiggy reviews → 717 about Instamart | 713 |
| **Total** | | **4,638** |

Cleaning removed reviews of 15 characters or fewer (mostly "good", "nice") and exact duplicates.

**Instamart caveat:** Instamart has no standalone Android app, so its reviews come from the main Swiggy app, filtered by keywords (instamart, grocery, vegetables, milk, eggs, expired, etc.). This misses Instamart reviews that don't use those words, may include some food-delivery reviews, and covers a longer time window than the other two apps.

## Results
_To be filled in after the eval:_

| Metric | Score |
|---|---|
| Theme match rate (100 hand-labelled reviews) | |
| Quote accuracy (quotes found in raw data) | |
| Noise rate (reviews left unclustered) | |

**Top problems:** _ranked table goes here_

**Competitor gap:** _headline insight goes here_

## Where it fails
1. **First clustering attempt: 94% noise, only 2 clusters, both generic praise.** Positive reviews ("very good service", "nice app") dominated, and HDBSCAN couldn't find dense groups in raw 384-dimension embeddings. Fix: filtered to 1–3 star reviews (the question is about frustrations) and reduced embeddings to 10 dimensions with UMAP before clustering.
2. **Instamart sample was too small at first.** 15,000 Swiggy reviews produced only 184 Instamart ones. Fix: pulled 40,000 and broadened the keyword list, giving 713.
3. _To add: a real case where clusters or labels were wrong._

## What I would do next
- _To fill in_
- _To fill in_

## Run it
```
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=your-key' > .env
python src/scrape_reviews.py
python src/cluster_reviews.py
```
