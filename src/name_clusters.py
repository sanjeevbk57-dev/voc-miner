"""Step 4 of the guide: have Claude name each cluster, then verify its quotes.

Run from the voc-miner folder:  python src/name_clusters.py
Input:  data/reviews_clustered.csv
Output: data/cluster_labels.csv
Cost: roughly 30 small API calls, a few cents.
"""
import json
import re
import sys
import pandas as pd
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
SAMPLES_PER_CLUSTER = 30

# Forcing a tool call makes Claude return clean structured data instead of free text.
TOOL = {
    "name": "record_cluster",
    "description": "Record the label for one group of app reviews.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string",
                      "description": "The main problem, 4 words or fewer"},
            "summary": {"type": "string",
                        "description": "One sentence describing the complaint"},
            "quotes": {"type": "array", "items": {"type": "string"},
                       "description": "3 short quotes copied EXACTLY, character for character, from the reviews"},
            "coherent": {"type": "boolean",
                         "description": "True if most reviews share one main problem; false if it is a mix of unrelated complaints"},
        },
        "required": ["label", "summary", "quotes", "coherent"],
    },
}


def norm(s):
    """Lowercase and collapse whitespace so tiny formatting differences don't fail the check."""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def name_cluster(samples):
    reviews = "\n".join(f"{i+1}. {t}" for i, t in enumerate(samples))
    msg = client.messages.create(
        model=MODEL, max_tokens=1500,
        tools=[TOOL],  # offered, not forced: this model rejects forced tool_choice
        messages=[{"role": "user", "content":
            "These are complaint reviews of Indian quick-commerce apps "
            "(Zepto, Blinkit, Swiggy Instamart) that were grouped together.\n\n"
            f"{reviews}\n\n"
            "Name the main problem they share. Quotes must be copied exactly "
            "from the reviews above; do not fix spelling or grammar. "
            "Respond only by calling the record_cluster tool."}])
    for b in msg.content:
        if b.type == "tool_use":
            return b.input
    # Fallback: Claude answered in text instead; pull the JSON out of it.
    text = "".join(b.text for b in msg.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no tool call or JSON in response")
    return json.loads(match.group(0))


df = pd.read_csv("data/reviews_clustered.csv")
clustered = df[df.cluster >= 0]
order = clustered.cluster.value_counts().index  # biggest first

rows, returned, found = [], 0, 0
for cid in order:
    grp = clustered[clustered.cluster == cid]
    samples = grp.text.sample(min(SAMPLES_PER_CLUSTER, len(grp)), random_state=1).tolist()
    try:
        out = name_cluster(samples)
    except Exception as e:
        print(f"Cluster {cid}: FAILED ({e})")
        continue

    # Quote check: every quote must appear in this cluster's actual reviews.
    texts = [norm(t) for t in grp.text]
    good, bad = [], []
    for q in out["quotes"]:
        q_clean = q.strip(' "\'“”…')
        (good if any(norm(q_clean) in t for t in texts) else bad).append(q_clean)
    returned += len(out["quotes"])
    found += len(good)

    rows.append({
        "cluster": cid,
        "reviews": len(grp),
        "avg_rating": round(grp.rating.mean(), 2),
        "label": out["label"],
        "summary": out["summary"],
        "coherent": out["coherent"],
        "quotes": " || ".join(good),
        "fake_quotes": " || ".join(bad),
    })
    flag = "" if out["coherent"] else "  [MIXED]"
    fake = f"  ({len(bad)} quote(s) not found)" if bad else ""
    print(f"Cluster {cid:>2} | {len(grp):>4} reviews | {out['label']}{flag}{fake}")

if not rows:
    sys.exit("\nEvery cluster failed - see the errors above. Nothing saved.")

labels = pd.DataFrame(rows)
labels.to_csv("data/cluster_labels.csv", index=False)

print(f"\nQuote accuracy: {found}/{returned} = {found / max(returned, 1):.0%}")
print(f"Mixed clusters: {(~labels.coherent).sum()} of {len(labels)}")
print("Saved to data/cluster_labels.csv")