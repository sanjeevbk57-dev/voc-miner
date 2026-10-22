"""Hand-labelling tool for the eval. Shows one review at a time, hides Claude's label.

Run from the voc-miner folder:  streamlit run src/label_app.py
Output: evals/hand_labels.csv (saved after every click, so you can stop anytime)
"""
import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from themes import THEMES

EVAL_SET = "evals/eval_set.csv"
OUT = "evals/hand_labels.csv"
os.makedirs("evals", exist_ok=True)

# Build the eval set once: 50 random Zepto + 50 random Blinkit labelled complaints.
if not os.path.exists(EVAL_SET):
    labelled = pd.read_csv("data/review_labels.csv")
    labelled = labelled[labelled.app.isin(["Zepto", "Blinkit"])]
    eval_set = pd.concat([g.sample(50, random_state=11) for _, g in labelled.groupby("app")])
    eval_set = eval_set.sample(frac=1, random_state=11)  # shuffle so apps are mixed
    eval_set[["rid", "text"]].to_csv(EVAL_SET, index=False)

eval_set = pd.read_csv(EVAL_SET)
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["rid", "theme", "support_failed"])
todo = eval_set[~eval_set.rid.isin(done.rid)]

st.title("Label reviews for the eval")
st.progress(len(done) / len(eval_set), text=f"{len(done)} of {len(eval_set)} labelled")

if todo.empty:
    st.success("All done! Now run:  python src/score_eval.py")
    st.stop()

row = todo.iloc[0]
st.markdown(f"> {row.text}")

with st.form(key=f"f{row.rid}"):
    theme = st.radio("Root problem (what actually went wrong?)", list(THEMES),
                     captions=list(THEMES.values()), index=None)
    support = st.checkbox("Says support / refund / return / replacement failed")
    if st.form_submit_button("Save and next"):
        if theme is None:
            st.warning("Pick a theme first")
        else:
            new = pd.DataFrame([{"rid": row.rid, "theme": theme, "support_failed": support}])
            pd.concat([done, new]).to_csv(OUT, index=False)
            st.rerun()
