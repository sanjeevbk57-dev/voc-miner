# Failure Log

Everything that broke while building this, what caused it, and what I changed. Ordered as it happened.

---

### 1. Too few Instamart reviews
**What happened:** Pulling 15,000 Swiggy reviews and filtering for Instamart gave only 184. That's too few to compare against Zepto and Blinkit.
**Why:** Instamart has no standalone Android app, so its reviews are buried inside the main Swiggy app, which is mostly about food delivery.
**Fix:** Pulled 40,000 Swiggy reviews and broadened the keywords ("insta mart", eggs, bread, expired, rotten...), which gave 713.
**Lesson:** Check where the data actually lives before choosing competitors. (This fix later caused failure #6.)

### 2. First clustering run: 94% noise, 2 clusters, both praise
**What happened:** HDBSCAN on all reviews found only 2 clusters ("very good service", "nice app") and left 94% of reviews unassigned.
**Why:** Positive one-liners dominated, and HDBSCAN can't find dense groups in raw 384-dimension embeddings, where every review looks equally far from every other.
**Fix:** Kept only 1–3 star reviews (the question is about frustrations) and reduced embeddings to 10 dimensions with UMAP before clustering. Result: 30 clusters, 13% noise.
**Lesson:** Filter the data to the question being asked before modelling it.

### 3. Large clusters mixed several problems
**What happened:** Claude flagged 8 of 30 clusters as mixed. The biggest ones combined missing items, refunds and damaged products.
**Why:** Angry reviews list several problems at once ("4 items missing, support refused a refund"), so they sit between topics.
**Fix:** Led to the redesign in #10.

### 4. One cluster grouped reviews by language, not topic
**What happened:** A 232-review cluster collected Hindi, Hinglish and Tamil reviews about unrelated problems.
**Why:** The embedding model (all-MiniLM-L6-v2) is English-only, so non-English text clusters by script and vocabulary rather than meaning.
**Fix:** Review-level classification with Claude (#10), which reads these languages. A multilingual embedding model would be the next improvement.

### 5. Claude ignored the "Off-topic" instruction
**What happened:** I told Claude to mark restaurant food-delivery clusters as "Off-topic". It put "Bad food, no refund" under expired products instead.
**Why:** The cluster was labelled with a plausible grocery-sounding problem, and the instruction competed with "group by root problem".
**Fix:** Made Off-topic an explicit theme in the fixed list and labelled reviews individually. 10% of sampled Instamart complaints were then correctly marked off-topic.
**Lesson:** Verify that instructions were followed instead of assuming it.

### 6. Instamart numbers were an artifact of the keyword filter
**What happened:** Instamart showed 15% expired products, 11% rotten produce and an 88% complaint rate, far worse than the others.
**Why:** The keyword filter from #1 specifically selected reviews containing "expired" and "rotten", so we measured our own filter. After matching time windows (#8), only 120 Instamart complaints remained.
**Fix:** Excluded Instamart from all comparison claims. It's reported as a data limitation.
**Lesson:** A fix to one problem can quietly bias the result. Check how each group was sampled before comparing groups.

### 7. Clustering overstated the headline gap
**What happened:** Cluster themes said Zepto had 12x more missing-item complaints than Blinkit.
**Why:** A large mixed cluster (#3) was labelled "missing items" and pulled in other Zepto complaints.
**Fix:** Cross-checked with a simple keyword count, which gave about 4x. I report the conservative number and always cross-check headline claims with a second method.

### 8. Apps were compared over different time windows
**What happened:** Blinkit's 4,000 reviews covered 3 days (18–21 Oct), while Zepto's covered 3 weeks (1–21 Oct).
**Why:** Blinkit gets far more reviews per day, so "newest 4,000" means a much shorter period.
**Fix:** Pulled 30,000 Blinkit reviews and cut every app to the same window, 1–21 Oct 2026.
**Lesson:** Equal sample sizes don't make a fair comparison. Equal conditions do.

### 9. Themes changed on every run
**What happened:** When Claude invented themes, a re-run gave a different list. "Missing or wrong items" disappeared, and "Support and refunds" swelled to 43%.
**Why:** Open-ended grouping isn't stable, and "support" is a plausible label for almost any angry review.
**Fix:** I defined a fixed list of 13 themes (`src/themes.py`) based on what clustering discovered. Claude only assigns reviews to it. This also made the eval possible.
**Lesson:** The taxonomy is a product decision. The model should apply it, not invent it.

### 10. Cluster-level labels hid root causes
**What happened:** Even with fixed themes, "Missing or wrong items" showed 0.3% for both apps, while a keyword search showed Zepto 6.8% vs Blinkit 1.4%.
**Why:** Each cluster gets one theme, so a cluster of "item missing + support refused" reviews went entirely to support.
**Fix:** Redesigned the pipeline. Clustering now *discovers* themes, and Claude *measures* them by labelling each review with a root cause plus a separate "support failed" flag. Missing items then showed Zepto 8.2% vs Blinkit 3.0%, in line with the keyword check.
**Lesson:** Use clustering to discover and classification to measure.

### 11. Claude still under-labels missing items
**What happened:** In the eval, Claude found 10 of the 14 missing-item reviews I labelled (71% recall). Two went to "Support and refunds".
**Why:** The same pull toward support as #9 and #10, now much smaller.
**Impact:** Claude undercounts missing items, so the true Zepto gap is likely bigger than reported. The headline is conservative.

### 12. "Support failed" flag is unreliable
**What happened:** Claude agreed with my support-failed labels only 63% of the time.
**Why:** _To fill in after checking whether Claude over- or under-flags._
**Impact:** The "support fails in 48% of Zepto complaints vs 27% of Blinkit's" finding is not used as a headline claim.

### 13. Complaint rate is partly a review-prompt effect
**What happened:** Blinkit's complaint rate (37%) is much lower than Zepto's (64%), but Blinkit also got 11,542 reviews in three weeks against Zepto's 2,219.
**Why:** Blinkit likely prompts satisfied users to rate the app more often, which dilutes its complaint rate. Play Store reviews aren't a random sample of customers.
**Impact:** I compare *what* users complain about more confidently than *how much* they complain.

### Eval disagreements (from `evals/disagreements.csv`)
_Pick 3 real cases where my label and Claude's differed and explain each in one line._
1.
2.
3.

### Engineering issues
- **Python 3.15 had no PyTorch build.** Installed Python 3.12 alongside it.
- **The API rejected forced tool calls** (`tool_choice: tool`) for this model. Switched to offering the tool, with a JSON fallback parser.
- **HDBSCAN cluster labels changed between runs** once the data changed, so all downstream steps re-run from scratch.
