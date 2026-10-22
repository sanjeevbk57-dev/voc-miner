# Quick-Commerce Voice-of-Customer Miner

**Problem:** Product teams make decisions from a handful of anecdotal reviews because reading thousands is too slow. This agent reads them all and tells you which problems users hate most, and where each competitor is weak, with real quotes and measured accuracy.

**Question:** What frustrates quick-commerce users the most, and where is Zepto weaker than Blinkit?

## Headline findings (Google Play, 1–21 Oct 2026)

| | Zepto | Blinkit | Gap |
|---|---|---|---|
| Missing or wrong items (% of all reviews) | 8.2% | 3.0% | **2.7x** |
| Expired, rotten or poor quality | 12.1% | 5.8% | 2.1x |
| Support also failed (% of complaints) | 48% | 27% | 1.8x |
| Complaint rate (1–3★ reviews) | 64% | 37% | 1.7x* |

**Zepto's weakness is fulfilment accuracy plus recovery:** users get missing, wrong or expired items more often, and when they complain, support fails them about half the time. A simple keyword cross-check puts the missing-items gap at about 5x (6.8% vs 1.4%), and the eval shows Claude *undercounts* missing items, so 2.7x is the conservative figure.

Blinkit's complaints lean relatively more towards fees (10.6% of complaints vs 5.4%) and delivery-partner behaviour (7.1% vs 2.6%).

\*Partly a review-prompt effect: Blinkit got ~5x more reviews in the same window, likely from prompting satisfied users.

**If I were the PM:** fix recovery first (instant auto-refund or replacement for reported missing items), because it's faster to ship than warehouse picking accuracy and directly addresses the 48% support-failure rate.

## Demo
_60-second video: add link_ · Run locally: `streamlit run src/app.py`

## How it works
```
Play Store reviews ─► complaints only (1–3★) ─► embeddings + UMAP + HDBSCAN ─► Claude names clusters
                                                                                       │
                                            discover themes ◄──────────────────────────┘
                                                   │
                         fixed 13-theme taxonomy (src/themes.py, my decision)
                                                   │
           Claude labels each complaint: root-cause theme + "support also failed" flag
                                                   │
                       ranking, Zepto vs Blinkit comparison, Streamlit demo
```
- **Clustering discovers** what people complain about; **review-level classification measures** it. (Cluster-level labels hid root causes. See the failure log, #10.)
- Every quote Claude returns is string-matched against the raw reviews, and made-up quotes are dropped.
- Tools: Python, sentence-transformers, UMAP, HDBSCAN, Claude (Anthropic API), pandas, Streamlit.

## Data
Public Google Play reviews (India, English locale; many are Hindi, Hinglish or other languages), 1–21 Oct 2026.

| App | Reviews in window | Complaints sampled for labelling |
|---|---|---|
| Zepto | 2,219 | 800 |
| Blinkit | 11,542 | 800 |
| Swiggy Instamart | (excluded) | 120 |

**Instamart is excluded from all claims.** It has no standalone Android app, so its reviews came from the Swiggy app via a keyword filter that biased the sample towards expired and rotten products, and only 120 complaints fell in the window.

## Eval
100 complaint reviews (50 Zepto, 50 Blinkit), hand-labelled blind (app and Claude's label hidden), using a labelling app (`src/label_app.py`).

| Metric | Score |
|---|---|
| Theme match rate (13 themes) | **81%** |
| "Missing or wrong items": precision / recall | 91% / 71% |
| Quote accuracy (cluster quotes found word-for-word) | 100% (90/90) |
| "Support failed" agreement | 63% before a written definition; see failure log #12 |

With 100 labels, each percentage has a margin of roughly ±8 points.

## Where it fails
Full log with 13 entries: [FAILURE_LOG.md](FAILURE_LOG.md). The three that taught me most:
1. **Cluster-level themes hid the root cause.** Missing items showed 0.3% for both apps while a keyword search showed 6.8% vs 1.4%, because "item missing + refund refused" reviews were clustered under support. Fix: label each review individually.
2. **My own fix biased the data.** The keywords I added to find more Instamart reviews ("expired", "rotten") then made Instamart look 6x worse on quality. Fix: excluded Instamart from claims.
3. **The low eval score was partly my labels.** Support-flag agreement was 63% because I'd labelled without a written definition; Claude was right on most disputed cases. Lesson: read the disagreements before "fixing" the model.

## What I would do next
- Run a blind re-label of all 100 eval reviews for the support flag, and grow the eval set to 300 for tighter error bars
- Swap in a multilingual embedding model so Hindi and Marathi reviews cluster by topic, not language
- Track the themes weekly to catch spikes (e.g. a missing-items jump after a warehouse change)

## Run it
```
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=your-key' > .env

python src/scrape_reviews.py      # collect reviews (data/reviews.csv)
python src/cluster_reviews.py     # discover themes
python src/name_clusters.py       # Claude names clusters, quotes verified
python src/classify_reviews.py    # label each complaint by root cause (under $1)
python src/score_eval.py          # score against hand labels
streamlit run src/app.py          # demo page
```