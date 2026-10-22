"""Steps 5 and 6 of the guide: merge clusters into themes, rank them, compare apps.

Run from the voc-miner folder:  python src/rank_themes.py
Inputs:  data/reviews.csv, data/reviews_clustered.csv, data/cluster_labels.csv
Outputs: data/cluster_themes.csv, data/theme_ranking.csv, data/competitor_gap.csv,
         results/competitor_gap.png
"""
import json
import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"

all_reviews = pd.read_csv("data/reviews.csv")
clustered = pd.read_csv("data/reviews_clustered.csv")
labels = pd.read_csv("data/cluster_labels.csv")

# ---------- 1. Ask Claude to merge 30 clusters into ~10 root-cause themes ----------
TOOL = {
    "name": "record_themes",
    "description": "Record how clusters are grouped into themes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "themes": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Theme name, 4 words or fewer"},
                    "cluster_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "cluster_ids"]}},
        },
        "required": ["themes"],
    },
}

# Fixed theme list (a PM decision): keeps results stable across runs and is the
# same list used to hand-label reviews for the eval.
THEMES = {
    "Missing or wrong items": "items missing from the order, or a different item sent",
    "Order not delivered or cancelled": "order never arrived, or was cancelled after payment",
    "Expired, rotten or poor quality": "expired, spoiled, rotten, fake or low-quality products",
    "Damaged or leaking packaging": "torn bags, leaks, broken seals, melted or warm items",
    "Late delivery": "slower than the promised time, wrong ETAs",
    "Fees and charges": "delivery, handling, surge, platform or small-order fees",
    "Prices and offers": "high prices, price changes, offers or cashback not working",
    "Payment problems": "failed payments, double debits, cash on delivery unavailable",
    "App, account or location": "app crashes, login, account blocked, area not serviceable, referral",
    "Delivery partner behaviour": "rude or unresponsive delivery partners",
    "Support and refunds": "ONLY when the main complaint is support itself or a refund, with no clearer root problem above",
    "Generic complaint": "vague 'worst app' reviews with no specific problem",
    "Off-topic": "restaurant food delivery, not groceries or quick commerce",
}
theme_list = "\n".join(f"- {name}: {desc}" for name, desc in THEMES.items())

table = "\n".join(f"cluster {r.cluster} | {r.reviews} reviews | {r.label} | {r.summary}"
                  for r in labels.itertuples())
prompt = (
    "Below are clusters of complaint reviews for Indian quick-commerce apps "
    "(Zepto, Blinkit, Swiggy Instamart).\n\n" + table + "\n\n"
    "Assign every cluster to exactly one of these themes, using the theme names "
    "exactly as written:\n" + theme_list + "\n\n"
    "Choose by the ROOT problem: what actually went wrong. Most complaints also say "
    "support was unhelpful or a refund was denied; that alone does not make it "
    "'Support and refunds'. If a cluster's label names a root problem (missing item, "
    "expired product, damaged item, non-delivery), use that theme. "
    "Respond only by calling the record_themes tool, with one entry per theme used."
)
msg = client.messages.create(model=MODEL, max_tokens=3000, tools=[TOOL],
                             messages=[{"role": "user", "content": prompt}])
result = next((b.input for b in msg.content if b.type == "tool_use"), None)
if result is None:
    text = "".join(b.text for b in msg.content if b.type == "text")
    result = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))

theme_of = {}
for t in result["themes"]:
    name = t["name"] if t["name"] in THEMES else "Other"
    if name == "Other":
        print(f"Note: Claude used an unknown theme '{t['name']}', put in 'Other'")
    for cid in t["cluster_ids"]:
        theme_of.setdefault(int(cid), name)
missing = set(labels.cluster) - set(theme_of)
for cid in missing:
    theme_of[cid] = "Other"
if missing:
    print(f"Note: clusters {sorted(missing)} were not assigned, put in 'Other'")

labels["theme"] = labels.cluster.map(theme_of)
labels[["cluster", "reviews", "label", "theme"]].to_csv("data/cluster_themes.csv", index=False)

print("Cluster -> theme mapping:")
for theme, grp in labels.groupby("theme"):
    print(f"  {theme}: " + ", ".join(f"{c} ({l})" for c, l in zip(grp.cluster, grp.label)))

# ---------- 2. Rank themes ----------
comp = clustered[clustered.cluster >= 0].copy()
comp["theme"] = comp.cluster.map(theme_of)
off_topic = (comp.theme == "Off-topic").sum()
comp = comp[comp.theme != "Off-topic"]
print(f"\nDropped {off_topic} off-topic (food delivery) reviews")

rank = comp.groupby("theme").agg(reviews=("text", "size"), avg_rating=("rating", "mean"))
rank["share"] = rank.reviews / len(comp)
rank["pain"] = 5 - rank.avg_rating
rank["score"] = rank.share * rank.pain
rank = rank.sort_values("score", ascending=False)

def theme_quotes(theme, n=3):
    qs = []
    for q in labels[labels.theme == theme].quotes.dropna():
        qs += [x for x in q.split(" || ") if x]
    return qs[:n]

rank["quotes"] = [" || ".join(theme_quotes(t)) for t in rank.index]
rank.insert(0, "rank", range(1, len(rank) + 1))
rank.round(3).to_csv("data/theme_ranking.csv")

print("\nRANKED PROBLEMS")
print(f"{'#':>2}  {'Theme':<32} {'Share':>6} {'Avg★':>5}")
for theme, r in rank.iterrows():
    print(f"{r['rank']:>2}  {theme:<32} {r.share:>6.0%} {r.avg_rating:>5.1f}")
    for q in theme_quotes(theme):
        print(f"      “{q[:110]}”")

# ---------- 3. Competitor gap: % of ALL each app's reviews that hit each theme ----------
totals = all_reviews.groupby("app").size()
complaint_rate = all_reviews[all_reviews.rating <= 3].groupby("app").size() / totals
print("\nComplaint rate (1-3 star share of all reviews):")
for app, v in complaint_rate.items():
    print(f"  {app}: {v:.0%}")

gap = (comp.groupby(["theme", "app"]).size().unstack(fill_value=0)
       .div(totals, axis=1) * 100).loc[rank.index]
gap.round(1).to_csv("data/competitor_gap.csv")

print("\nCOMPETITOR GAP (% of each app's reviews mentioning the problem)")
print(gap.round(1).to_string())

print("\nBiggest gaps:")
for theme, row in gap.iterrows():
    hi, lo = row.idxmax(), row.idxmin()
    if row[lo] > 0 and row[hi] / row[lo] >= 2:
        print(f"  {theme}: {hi} {row[hi]:.1f}% vs {lo} {row[lo]:.1f}% ({row[hi]/row[lo]:.1f}x)")

# ---------- 4. Chart ----------
os.makedirs("results", exist_ok=True)
ax = gap.iloc[::-1].plot(kind="barh", figsize=(10, 7), width=0.8)
ax.set_xlabel("% of the app's reviews mentioning this problem")
ax.set_ylabel("")
ax.set_title("Where each quick-commerce app is weakest")
plt.tight_layout()
plt.savefig("results/competitor_gap.png", dpi=150)
print("\nSaved data/theme_ranking.csv, data/competitor_gap.csv, results/competitor_gap.png")