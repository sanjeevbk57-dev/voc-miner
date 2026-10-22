"""Measure themes per review (not per cluster).

Why: one review often has a root problem ("4 items missing") AND a support failure
("refund refused"). Cluster-level labels put the whole cluster under support and hid
the root cause. Here Claude labels each complaint review with:
  - theme: the root problem, from the fixed list in themes.py
  - support_failed: whether the review also says support/refund/replacement failed

Run from the voc-miner folder:  python src/classify_reviews.py
Input:  data/reviews.csv
Output: data/review_labels.csv, data/theme_share.csv, results/competitor_gap.png
Safe to re-run: finished batches are cached in data/review_labels_cache.jsonl
"""
import json
import os
import re
import pandas as pd
import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from themes import THEMES

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
PER_APP = 800      # complaint reviews sampled per app
BATCH = 40         # reviews per API call
CACHE = "data/review_labels_cache.jsonl"

# ---------- 1. Sample complaint reviews ----------
reviews = pd.read_csv("data/reviews.csv")
reviews["rid"] = range(len(reviews))
totals = reviews.groupby("app").size()
complaints = reviews[reviews.rating <= 3]
complaint_counts = complaints.groupby("app").size()
sample = pd.concat([g.sample(min(PER_APP, len(g)), random_state=7)
                    for _, g in complaints.groupby("app")])
print("Sampled complaint reviews:")
print(sample.groupby("app").size().to_string())

# ---------- 2. Classify in batches (cached) ----------
done = {}
if os.path.exists(CACHE):
    with open(CACHE) as f:
        for line in f:
            row = json.loads(line)
            done[row["rid"]] = row

TOOL = {
    "name": "record_labels",
    "description": "Record a label for every review in the batch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "theme": {"type": "string", "enum": list(THEMES)},
                    "support_failed": {"type": "boolean"},
                },
                "required": ["id", "theme", "support_failed"]}},
        },
        "required": ["labels"],
    },
}
theme_list = "\n".join(f"- {name}: {desc}" for name, desc in THEMES.items())


def classify(batch):
    reviews_txt = "\n".join(f"[{r.rid}] {str(r.text)[:600]}" for r in batch.itertuples())
    msg = client.messages.create(
        model=MODEL, max_tokens=4000, tools=[TOOL],
        messages=[{"role": "user", "content":
            "Label each complaint review of an Indian quick-commerce app "
            "(Zepto, Blinkit, Swiggy Instamart).\n\n"
            "theme = the ROOT problem, meaning what actually went wrong, from this list:\n"
            f"{theme_list}\n\n"
            "If a review names a root problem (missing item, expired product, non-delivery, "
            "fees...) use that theme even if it also complains about support. "
            "If it has several root problems, pick the main one.\n"
            "support_failed = true if the review says customer support, a refund, a return "
            "or a replacement failed or was refused; otherwise false.\n"
            "Reviews may be in Hindi, Hinglish or other languages; label them anyway.\n\n"
            f"{reviews_txt}\n\n"
            "Respond only by calling the record_labels tool, with one label per review id."}])
    out = next((b.input for b in msg.content if b.type == "tool_use"), None)
    if out is None:
        text = "".join(b.text for b in msg.content if b.type == "text")
        out = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
    return out["labels"]


todo = sample[~sample.rid.isin(done)]
batches = [todo.iloc[i:i + BATCH] for i in range(0, len(todo), BATCH)]
print(f"\n{len(done)} already labelled, {len(todo)} to go in {len(batches)} batches")

with open(CACHE, "a") as f:
    for i, batch in enumerate(batches, 1):
        try:
            labels = classify(batch)
        except Exception as e:
            print(f"Batch {i}: FAILED ({e}) - re-run the script to retry")
            continue
        ids = set(batch.rid)
        for lab in labels:
            if lab["id"] in ids:
                row = {"rid": lab["id"], "theme": lab["theme"],
                       "support_failed": bool(lab["support_failed"])}
                done[row["rid"]] = row
                f.write(json.dumps(row) + "\n")
        print(f"Batch {i}/{len(batches)} done")

# ---------- 3. Results ----------
labels = pd.DataFrame(done.values())
labelled = sample.merge(labels, on="rid", how="inner")
labelled.loc[~labelled.theme.isin(THEMES), "theme"] = "Generic complaint"
unlabelled = len(sample) - len(labelled)
if unlabelled:
    print(f"\nNote: {unlabelled} sampled reviews got no label (re-run to retry)")
labelled.to_csv("data/review_labels.csv", index=False)

off = (labelled.theme == "Off-topic").groupby(labelled.app).mean() * 100
labelled = labelled[labelled.theme != "Off-topic"]

per_app_n = labelled.groupby("app").size()
share_of_complaints = (labelled.groupby(["theme", "app"]).size().unstack(fill_value=0)
                       .div(per_app_n, axis=1) * 100)
complaint_rate = complaint_counts / totals
share_of_all = share_of_complaints.mul(complaint_rate, axis=1)   # estimated % of ALL reviews
order = labelled.theme.value_counts().index
share_of_complaints = share_of_complaints.loc[order]
share_of_all = share_of_all.loc[order]
share_of_all.round(2).to_csv("data/theme_share.csv")

print("\nComplaint rate (1-3 star share of all reviews):")
print((complaint_rate * 100).round(0).astype(int).astype(str).add("%").to_string())

print("\nTHEME SHARE: % of each app's COMPLAINTS")
print(share_of_complaints.round(1).to_string())

print("\nTHEME SHARE: estimated % of ALL each app's reviews")
print(share_of_all.round(1).to_string())

print("\nSupport also failed (% of complaints):")
print((labelled.groupby("app").support_failed.mean() * 100).round(0).to_string())

print("\nOff-topic (food delivery) share dropped:")
print(off.round(1).to_string())

# ---------- 4. Chart: Zepto vs Blinkit (same collection method, same window) ----------
os.makedirs("results", exist_ok=True)
cols = [c for c in ["Blinkit", "Zepto"] if c in share_of_all.columns]
ax = share_of_all[cols].iloc[::-1].plot(kind="barh", figsize=(10, 7), width=0.8)
ax.set_xlabel("Estimated % of all the app's reviews (1-21 Oct 2026)")
ax.set_ylabel("")
ax.set_title("Zepto vs Blinkit: what users complain about")
plt.tight_layout()
plt.savefig("results/competitor_gap.png", dpi=150)
print("\nSaved data/review_labels.csv, data/theme_share.csv, results/competitor_gap.png")
