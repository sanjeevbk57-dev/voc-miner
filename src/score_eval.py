"""Score Claude's review labels against your hand labels.

Run from the voc-miner folder:  python src/score_eval.py
Inputs:  evals/hand_labels.csv, data/review_labels.csv
Output:  evals/eval_results.txt, evals/disagreements.csv
"""
import pandas as pd

hand = pd.read_csv("evals/hand_labels.csv")
model = pd.read_csv("data/review_labels.csv")[["rid", "app", "text", "theme", "support_failed"]]
df = hand.merge(model, on="rid", suffixes=("_hand", "_claude"))
n = len(df)

theme_match = (df.theme_hand == df.theme_claude).mean()
support_match = (df.support_failed_hand.astype(bool) == df.support_failed_claude.astype(bool)).mean()

# Precision / recall for the headline theme
T = "Missing or wrong items"
claude_t, hand_t = df.theme_claude == T, df.theme_hand == T
precision = (claude_t & hand_t).sum() / max(claude_t.sum(), 1)
recall = (claude_t & hand_t).sum() / max(hand_t.sum(), 1)

lines = [
    f"Eval set: {n} hand-labelled complaint reviews (Zepto + Blinkit)",
    f"Theme match rate:          {theme_match:.0%}",
    f"Support-failed agreement:  {support_match:.0%}",
    f"'{T}': precision {precision:.0%} ({claude_t.sum()} labelled by Claude), "
    f"recall {recall:.0%} ({hand_t.sum()} labelled by you)",
    "",
    "Most common disagreements (you -> Claude):",
]
dis = df[df.theme_hand != df.theme_claude]
for (h, c), k in dis.groupby(["theme_hand", "theme_claude"]).size().sort_values(ascending=False).head(6).items():
    lines.append(f"  {k} x  {h}  ->  {c}")

report = "\n".join(lines)
print(report)
with open("evals/eval_results.txt", "w") as f:
    f.write(report + "\n")

dis[["rid", "app", "text", "theme_hand", "theme_claude"]].to_csv("evals/disagreements.csv", index=False)
print(f"\nSaved {len(dis)} disagreements to evals/disagreements.csv - pick 3 for your failure log")
