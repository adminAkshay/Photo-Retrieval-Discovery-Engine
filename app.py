import re
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Photo Retrieval Discovery Engine", layout="wide")
st.title("Photo Retrieval Discovery Engine")
st.caption("Analyze user feedback about finding photos — free, local, no API keys, no server.")


# ---------- Load data ----------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("data.csv")
    df["quote_or_summary"] = df["quote_or_summary"].fillna("").astype(str)
    df["topic"] = df["topic"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].fillna("neutral").astype(str).str.lower()
    df["workaround"] = df["workaround"].fillna("N").astype(str).str.upper()
    return df


uploaded = st.sidebar.file_uploader("Upload CSV (or use bundled data.csv)", type=["csv"])
df = load_data(uploaded)


# ---------- Layer 1: Memory cue extraction (rule-based) ----------
CUE_CATEGORIES = {
    "Person": ["dad", "mom", "mother", "father", "daughter", "son", "wife", "husband",
               "friend", "family", "baby", "child", "kid", "grandma", "grandpa",
               "jane", "john", "selfie", " me "],
    "Place": ["beach", "temple", "japan", "city", "park", "mountain", "home", "house",
              "travel", "trip", "vacation", "abroad", "outdoor", "indoor"],
    "Object": ["dog", "cat", "pet", "car", "truck", "blowtorch", "bird", "kookaburra",
               "coyote", "redeye", "red eye", "food"],
    "Event": ["birthday", "wedding", "party", "concert", "graduation", "christmas",
              "holiday", "anniversary"],
    "Time": ["2012", "2015", "2016", "2019", "2020", "years ago", "old photo",
             "old picture", "childhood", "recent", "yesterday", "last year"],
    "Visual": ["sunset", "blossom", "cherry", "yellow", "sunlight", "night",
               "screenshot", "portrait", "red"],
}


def extract_cues(text):
    t = text.lower()
    return [(cat, kw) for cat, kws in CUE_CATEGORIES.items() for kw in kws if kw in t]


# ---------- Layer 2: Retrieval problem taxonomy ----------
PROBLEM_MAP = {
    "search_failure": "Search / recall failure",
    "face_recognition": "Person / face retrieval",
    "folder_view": "Organization / browse failure",
    "ai_gemini": "AI search usability",
    "data_loss": "Missing / lost photos",
    "recovery": "Recovery attempt",
    "backup_issue": "Backup / sync issue",
    "storage_quota": "Storage / quota pressure",
    "free_up_space": "Free-up-space confusion",
    "ui_change": "UI / layout disruption",
    "privacy": "Privacy / trust",
    "workaround": "User workaround (coping)",
    "feature_request": "Feature request",
    "positive_experience": "Positive experience",
    "sharing": "Sharing",
}


def get_problems(topic_str):
    parts = [p.strip() for p in topic_str.split(",") if p.strip()]
    return [PROBLEM_MAP.get(p, p) for p in parts]


# ---------- Layer 3: Query formulation detection ----------
def detect_query_formulation(text):
    t = text.lower()
    patterns = [
        ("Quoted keyword", [r"'[^']+'", r'"[^"]+"']),
        ("Face / people query", [r"\bsearch.*(face|person|people|mom|dad|daughter|son)\b",
                                 r"\bface.*search\b"]),
        ("Date / time query", [r"\bsearch.*(date|year|month|2015|2016|2017)\b"]),
        ("Descriptive scene", [r"\bsearch.*(beach|temple|dog|cat|car|sunset|truck|bird)\b"]),
        ("Natural language", [r"\bhow (can|do) i\b", r"\bwhere (is|are)\b", r"i want"]),
    ]
    for label, pats in patterns:
        for p in pats:
            if re.search(p, t):
                return label
    return "Unspecified"


df["problems"] = df["topic"].apply(get_problems)
df["cues"] = df["quote_or_summary"].apply(extract_cues)
df["query_form"] = df["quote_or_summary"].apply(detect_query_formulation)


# ---------- KPIs ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", len(df))
c2.metric("Sources", df["source"].nunique())
c3.metric("Negative", int((df["sentiment"] == "negative").sum()))
c4.metric("Have workaround", int((df["workaround"] == "Y").sum()))


# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Problems", "Memory cues", "Query formulation", "Explore"])

with tab1:
    st.subheader("Top retrieval problems")
    prob_counts = Counter(p for ps in df["problems"] for p in ps)
    st.bar_chart(pd.DataFrame(prob_counts.most_common(15),
                              columns=["problem", "count"]).set_index("problem"))

    st.subheader("Problems by source (where does each problem live?)")
    exploded = df.explode("problems").dropna(subset=["problems"])
    pivot = exploded.groupby(["source", "problems"]).size().unstack(fill_value=0)
    st.dataframe(pivot)

with tab2:
    st.subheader("What do users actually remember about photos?")
    cue_counts = Counter(c[1] for cs in df["cues"] for c in cs)
    st.bar_chart(pd.DataFrame(cue_counts.most_common(20),
                              columns=["cue", "count"]).set_index("cue"))

    st.subheader("By cue category")
    cat_counts = Counter(c[0] for cs in df["cues"] for c in cs)
    st.bar_chart(pd.DataFrame(cat_counts.most_common(),
                              columns=["category", "count"]).set_index("category"))

with tab3:
    st.subheader("How users formulate searches when memory is incomplete")
    qf = df[df["query_form"] != "Unspecified"]["query_form"].value_counts()
    st.bar_chart(qf)

    st.subheader("Query formulation × sentiment")
    st.dataframe(pd.crosstab(df["query_form"], df["sentiment"]))

with tab4:
    st.subheader("Filter & export")
    sources = st.multiselect("Source", sorted(df["source"].unique()),
                             default=sorted(df["source"].unique()))
    sentiments = st.multiselect("Sentiment", sorted(df["sentiment"].unique()),
                                default=sorted(df["sentiment"].unique()))
    filtered = df[df["source"].isin(sources) & df["sentiment"].isin(sentiments)]

    st.write(f"**{len(filtered)}** records")
    st.dataframe(filtered[["row_id", "source", "date", "topic",
                           "sentiment", "quote_or_summary"]],
                 use_container_width=True)

    st.download_button("Download filtered CSV",
                       filtered.to_csv(index=False).encode("utf-8"),
                       "filtered.csv", "text/csv")
