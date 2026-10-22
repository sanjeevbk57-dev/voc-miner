# Quick-Commerce Voice-of-Customer Miner

**Question:** What frustrates quick-commerce app users the most, and where is each app (Zepto, Blinkit, Swiggy Instamart) weakest?

## How it works
- Collects recent Play Store reviews for each app (public data only)
- Groups reviews by problem using embeddings + HDBSCAN clustering
- Uses Claude to name each problem group and pull verbatim quotes, then ranks problems by size and pain

## Data
Google Play Store reviews (India, English), newest first, collected on: _fill in date_

| App | Reviews collected |
|---|---|
| Zepto | |
| Blinkit | |
| Swiggy Instamart | |

Note: Instamart has no standalone Android app, so its reviews come from the main Swiggy app, filtered by keywords (instamart, grocery, vegetables, milk, etc.). This misses Instamart reviews that don't use those words and may include a few food-delivery reviews.

## Results
_To fill in: theme match rate, quote accuracy, noise rate_

## Where it fails
_To fill in: 3 real examples_

## What I would do next
_To fill in_

## Run it
```
pip install -r requirements.txt
python src/scrape_reviews.py
python src/cluster_reviews.py
```
