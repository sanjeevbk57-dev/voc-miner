"""Re-check the support-failed flag on reviews where you and Claude disagreed.

Run from the voc-miner folder:  python src/recheck_support.py
Updates evals/hand_labels.csv in place (backup saved to evals/hand_labels_before_recheck.csv)

Definition to apply: TRUE if the review says customer support, a refund, a return
or a replacement failed, was refused, was delayed, or wasn't available. Includes
"no customer care", "they don't listen after", "return failed".
"""
import os
import shutil
import pandas as pd

HAND = "evals/hand_labels.csv"
if not os.path.exists("evals/hand_labels_before_recheck.csv"):
    shutil.copy(HAND, "evals/hand_labels_before_recheck.csv")

hand = pd.read_csv(HAND)
hand["support_failed"] = hand.support_failed.astype(bool)
model = pd.read_csv("data/review_labels.csv")[["rid", "text", "support_failed"]]
d = hand.merge(model, on="rid", suffixes=("", "_claude"))
disputed = d[d.support_failed != d.support_failed_claude.astype(bool)]

print(__doc__.split("Definition to apply:")[1])
print(f"{len(disputed)} disputed reviews. Type y (support failed), n (it didn't), or q to stop.\n")

changed = 0
for i, r in enumerate(disputed.itertuples(), 1):
    print(f"[{i}/{len(disputed)}] {r.text}\n")
    ans = ""
    while ans not in ("y", "n", "q"):
        ans = input("Did support / refund / return fail? (y/n/q): ").strip().lower()
    if ans == "q":
        break
    new = ans == "y"
    if new != r.support_failed:
        hand.loc[hand.rid == r.rid, "support_failed"] = new
        changed += 1
    print()

hand.to_csv(HAND, index=False)
print(f"Changed {changed} labels. Now run: python src/score_eval.py")
