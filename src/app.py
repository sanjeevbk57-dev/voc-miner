"""Demo page for the Voice-of-Customer Miner.

Run from the voc-miner folder:  streamlit run src/app.py
"""
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Quick-Commerce VoC Miner", layout="wide")
APPS = ["Zepto", "Blinkit"]
T_MISSING = "Missing or wrong items"


@st.cache_data
def load():
    reviews = pd.read_csv("data/reviews.csv")
    labels = pd.read_csv("data/review_labels.csv")
    labels = labels[(labels.theme != "Off-topic") & labels.app.isin(APPS)]
    labels["support_failed"] = labels.support_failed.astype(bool)
    totals = reviews.groupby("app").size()
    rate = reviews[reviews.rating <= 3].groupby("app").size() / totals
    return labels, totals, rate


labels, totals, rate = load()


def theme_table(app):
    l = labels[labels.app == app]
    t = l.groupby("theme").agg(reviews=("text", "size"), support=("support_failed", "mean"))
    t["% of complaints"] = t.reviews / len(l) * 100
    t["% of all reviews"] = t["% of complaints"] * rate[app]
    t["Support also failed"] = t.support * 100

    def examples(theme):
        texts = l[(l.theme == theme) & l.text.str.len().between(40, 220)].text
        return " | ".join(texts.sample(min(2, len(texts)), random_state=3)) if len(texts) else ""

    t["Example reviews"] = [examples(th) for th in t.index]
    t = t.sort_values("% of all reviews", ascending=False)
    t.insert(0, "Rank", range(1, len(t) + 1))
    return t.drop(columns=["support"])


tables = {a: theme_table(a) for a in APPS}
share = pd.DataFrame({a: tables[a]["% of all reviews"] for a in APPS}).fillna(0)
miss = share.loc[T_MISSING]
support = labels.groupby("app").support_failed.mean() * 100

# ---------- Header ----------
st.title("What frustrates quick-commerce users?")
st.caption(f"Google Play reviews, 1–21 Oct 2026 · Zepto {totals['Zepto']:,} and "
           f"Blinkit {totals['Blinkit']:,} reviews · complaints labelled by Claude")

st.subheader(f"Zepto users report missing or wrong items {miss['Zepto'] / miss['Blinkit']:.1f}x "
             f"as often as Blinkit users, and support also fails in "
             f"{support['Zepto']:.0f}% of Zepto complaints vs {support['Blinkit']:.0f}%.")

c1, c2, c3 = st.columns(3)
c1.metric("Complaint rate (1–3★)", f"Zepto {rate['Zepto']:.0%}", f"Blinkit {rate['Blinkit']:.0%}",
          delta_color="off")
c2.metric("Missing or wrong items (% of reviews)", f"Zepto {miss['Zepto']:.1f}%",
          f"Blinkit {miss['Blinkit']:.1f}%", delta_color="off")
c3.metric("Support also failed (% of complaints)", f"Zepto {support['Zepto']:.0f}%",
          f"Blinkit {support['Blinkit']:.0f}%", delta_color="off")

# ---------- Ranked problems ----------
st.divider()
app = st.radio("Show ranked problems for", APPS, horizontal=True)
st.dataframe(
    tables[app],
    column_config={
        "% of complaints": st.column_config.NumberColumn(format="%.1f%%"),
        "% of all reviews": st.column_config.NumberColumn(format="%.1f%%"),
        "Support also failed": st.column_config.NumberColumn(format="%.0f%%"),
        "Example reviews": st.column_config.TextColumn(width="large"),
    },
    use_container_width=True,
)

# ---------- Comparison ----------
st.divider()
st.subheader("Zepto vs Blinkit: estimated % of all reviews per problem")
st.bar_chart(share, horizontal=True, stack=False)

# ---------- Eval and limitations ----------
st.divider()
left, right = st.columns(2)
with left:
    st.subheader("How well it works")
    if os.path.exists("evals/eval_results.txt"):
        st.code(open("evals/eval_results.txt").read(), language=None)
    st.markdown("- Quotes used in cluster names: 90/90 found word-for-word in the data\n"
                "- Claude **undercounts** missing items (71% recall), so the gap above is conservative")
with right:
    st.subheader("Limitations")
    st.markdown(
        "- **Instamart excluded:** no standalone Android app; reviews came from the Swiggy app via "
        "keyword filter, which biased the sample\n"
        "- **Complaint rate is partly a review-prompt effect:** Blinkit got ~5x more reviews in the "
        "same window, likely from prompting happy users\n"
        "- Play Store reviewers are not a random sample of customers\n"
        "- Full list of what broke: FAILURE_LOG.md"
    )
