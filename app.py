import re
from collections import Counter

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Photo Retrieval Discovery Engine", layout="wide")
st.title("Photo Retrieval Discovery Engine")
st.caption("Analyze user feedback about finding photos — free, local, no API keys, no server.")


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_csv("data.csv")
    except FileNotFoundError:
        return pd.DataFrame()


uploaded = st.sidebar.file_uploader("Upload CSV (or use bundled data.csv)", type=["csv"])
df = load_data(uploaded)

if df.empty:
    st.warning("⚠️ No data found. Please upload your CSV file using the sidebar on the left to begin.")
    st.stop()

df["quote_or_summary"] = df["quote_or_summary"].fillna("").astype(str)
df["topic"] = df["topic"].fillna("").astype(str)
df["sentiment"] = df["sentiment"].fillna("neutral").astype(str).str.lower()
df["workaround"] = df["workaround"].fillna("N").astype(str).str.upper()
df["source"] = df["source"].fillna("unknown").astype(str)


# ============================================================
# LAYER 1: MEMORY CUES
# ============================================================
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


# ============================================================
# LAYER 2: PROBLEM TAXONOMY
# ============================================================
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


# ============================================================
# LAYER 3: QUERY FORMULATION
# ============================================================
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


# ============================================================
# Q&A ENGINE (rule-based, no LLM, free)
# ============================================================
PHOTO_KEYWORDS = [
    "photo", "picture", "image", "pic", "album", "search", "find", "recall",
    "memory", "memories", "face", "faces", "person", "people", "backup",
    "storage", "gallery", "camera", "screenshot", "video", "media", "google photos",
    "icloud", "takeout", "retriev", "lookup", "look up", "folder", "tag", "tagging",
    "recognition", "gemini", "ai search", "old photo", "old picture", "timeline",
    "organize", "organise", "complaint", "review", "reddit", "play store",
    "app store", "feedback", "workaround", "device", "forget", "forgot",
    "remember", "query", "sentiment", "topic", "problem", "issue", "record",
    "source", "insight", "dataset", "analyze", "analyse", "trend", "pattern",
    "user", "users", "ui", "layout", "gemini", "google one", "quota", "shared",
]

SUGGESTED_QUESTIONS = [
    "What are the top retrieval problems users report?",
    "What do users remember about their photos?",
    "What do users forget about their photos?",
    "How do users formulate searches?",
    "Which source complains most about AI search?",
    "Show me quotes about face recognition",
    "How many negative records do we have?",
    "What workarounds do users mention?",
]

OFF_TOPIC_RESPONSE = """### 🤔 That's outside what I do

I'm a **photo retrieval discovery engine**. I analyze user feedback and conversations about finding, organizing, and searching photos — I don't answer general questions.

**Here are some things I *can* tell you:**

- What are the top retrieval problems users report?
- What do users remember about their photos?
- What do users forget about their photos?
- How do users formulate searches when memory is incomplete?
- Which source complains most about AI search?
- Show me quotes about face recognition
- How many negative records do we have?
- What workarounds do users mention?

Just tap one of the suggestions above or ask something like *"show me quotes about old photos"*.
"""


def is_on_topic(q):
    return any(kw in q for kw in PHOTO_KEYWORDS)


def _md_list(items, bullet="-"):
    return "\n".join(f"{bullet} {i}" for i in items)


def answer_question(question, df):
    """Returns (markdown_answer, evidence_dataframe_or_None)."""
    q = question.lower().strip()

    if not is_on_topic(q):
        return OFF_TOPIC_RESPONSE, None

    # --- Top problems ---
    if re.search(r"(top|main|biggest|common|what).*problem", q) or "pain point" in q:
        counts = Counter(p for ps in df["problems"] for p in ps)
        md = "### 🔎 Top retrieval problems users report\n\n"
        for i, (p, c) in enumerate(counts.most_common(6), 1):
            pct = 100 * c / len(df)
            md += f"{i}. **{p}** — {c} mentions ({pct:.0f}% of records)\n"
        md += f"\n_Based on {len(df)} records from {df['source'].nunique()} sources._"
        return md, None

    # --- What users remember ---
    if re.search(r"remember|recall", q) and not re.search(r"forget|forgot", q):
        cue_counts = Counter(c[1] for cs in df["cues"] for c in cs)
        cat_counts = Counter(c[0] for cs in df["cues"] for c in cs)
        md = "### 🧠 What users remember about their photos\n\n"
        md += "**By category:**\n"
        md += _md_list([f"{cat}: {c} mentions" for cat, c in cat_counts.most_common()])
        md += "\n\n**Top specific cues:**\n"
        md += _md_list([f"`{cue}` — {c}" for cue, c in cue_counts.most_common(10)])
        md += "\n\n_People remember **who** and **what** and **where** — but rarely *when*._"
        return md, None

    # --- What users forget ---
    if re.search(r"forget|forgot|can'?t remember", q):
        mask = df["quote_or_summary"].str.lower().str.contains(
            "forgot|can't remember|cannot remember|don't remember|do not remember", na=False
        )
        subset = df[mask]
        md = "### 🕳️ What users forget about their photos\n\n"
        md += f"**{len(subset)}** records explicitly mention forgetting details.\n\n"
        md += "**Most commonly forgotten:**\n"
        md += "- 📅 Dates and years\n- 📍 Locations\n- 👤 Names of people\n- 📁 Which album or folder\n- 🔢 Sequence of events\n\n"
        md += "Sample quotes below 👇"
        return md, subset

    # --- How users search ---
    if re.search(r"how.*(search|query|ask|phrase|formulate)", q) or "query formulation" in q:
        qf = df[df["query_form"] != "Unspecified"]["query_form"].value_counts()
        md = "### 🔍 How users formulate searches\n\n"
        for form, c in qf.items():
            pct = 100 * c / len(df)
            md += f"- **{form}** — {c} records ({pct:.0f}%)\n"
        md += "\n_Users mix descriptive scene queries, face queries, and natural language._"
        return md, None

    # --- Which source / by source ---
    if re.search(r"(which|what) source|by source|per source|where.*live", q):
        exploded = df.explode("problems").dropna(subset=["problems"])
        pivot = exploded.groupby(["source", "problems"]).size().unstack(fill_value=0)
        md = "### 📊 Problems by source\n\n**Top problem per source:**\n\n"
        for src in pivot.index:
            top_prob = pivot.loc[src].idxmax()
            top_count = int(pivot.loc[src].max())
            md += f"- **{src}** → {top_prob} ({top_count})\n"
        return md, None

    # --- Stats ---
    if re.search(r"how many|count|stat|overall|negative|positive|total", q):
        md = "### 📈 Dataset overview\n\n"
        md += f"- **Total records:** {len(df)}\n"
        md += f"- **Sources:** {df['source'].nunique()}\n"
        for s, c in df["sentiment"].value_counts().items():
            md += f"- **{s.title()}:** {c}\n"
        wc = int((df["workaround"] == "Y").sum())
        md += f"- **With workaround:** {wc}\n"
        return md, None

    # --- Workarounds ---
    if "workaround" in q or "work around" in q or "coping" in q:
        subset = df[df["workaround"] == "Y"]
        md = f"### 🛠️ User workarounds\n\n**{len(subset)}** records describe a workaround.\n\n"
        md += "Users typically:\n"
        md += "- Download backups via Google Takeout\n"
        md += "- Switch to Immich, Synology, or local storage\n"
        md += "- Roll back to an older app version\n"
        md += "- Double-tap the search icon to access the old search\n"
        md += "- Manually correct face groups\n\n"
        md += "Full list below 👇"
        return md, subset

    # --- Quotes / examples ---
    if re.search(r"quote|example|show me|evidence|say about", q):
        for kw in PHOTO_KEYWORDS:
            if len(kw) > 4 and kw in q:
                subset = df[df["quote_or_summary"].str.lower().str.contains(kw, na=False)]
                if not subset.empty:
                    md = f"### 💬 Quotes about `{kw}`\n\n**{len(subset)}** matching records."
                    return md, subset

    # --- Specific topic mentions ---
    topic_matches = [
        (r"face|person|people|selfie", "Person / face retrieval"),
        (r"ai search|gemini|ask photos|ai usability", "AI search usability"),
        (r"old photo|childhood|years ago|forgot when", "Missing / lost photos"),
        (r"ui|layout|update|new design|redesign", "UI / layout disruption"),
        (r"backup|sync|upload", "Backup / sync issue"),
        (r"storage|quota|space|google one", "Storage / quota pressure"),
        (r"organization|organise|organize|folder|album", "Organization / browse failure"),
        (r"search|find|retrieve|lookup|look up", "Search / recall failure"),
        (r"privacy|trust|consent", "Privacy / trust"),
    ]
    for pattern, topic in topic_matches:
        if re.search(pattern, q):
            subset = df[df["problems"].apply(lambda ps: topic in ps)]
            if not subset.empty:
                md = f"### 📌 {topic}\n\n"
                md += f"**{len(subset)}** records report this problem.\n\n"
                md += "**Breakdown by source:**\n"
                for s, c in subset["source"].value_counts().items():
                    md += f"- {s}: {c}\n"
                md += "\n**Sample quotes below 👇**"
                return md, subset

    # --- Fallback: keyword search on quotes ---
    stopwords = {"what", "how", "does", "the", "and", "for", "with", "about",
                 "from", "they", "this", "that", "are", "can", "you", "show",
                 "give", "tell", "me", "some", "any", "many"}
    words = [w for w in re.findall(r"\b[a-z]{4,}\b", q) if w not in stopwords]
    if words:
        mask = df["quote_or_summary"].str.lower().str.contains("|".join(words), na=False)
        subset = df[mask]
        if not subset.empty:
            md = f"### 🔍 Results matching: _{', '.join(words)}_\n\n**{len(subset)}** matching records."
            return md, subset

    return OFF_TOPIC_RESPONSE, None


# ============================================================
# KPIs
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", len(df))
c2.metric("Sources", df["source"].nunique())
c3.metric("Negative", int((df["sentiment"] == "negative").sum()))
c4.metric("Have workaround", int((df["workaround"] == "Y").sum()))


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Problems", "Memory cues", "Query formulation", "Explore", "💬 Ask"]
)

with tab1:
    st.subheader("Top retrieval problems")
    prob_counts = Counter(p for ps in df["problems"] for p in ps)
    prob_df = pd.DataFrame(prob_counts.most_common(15), columns=["problem", "count"])
    prob_df = prob_df[prob_df["count"] > 0].set_index("problem")
    st.bar_chart(prob_df)

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

with tab5:
    st.subheader("💬 Ask about photo retrieval")
    st.caption("Ask about what users say. Anything off-topic will be politely declined.")

    # Suggested question buttons
    st.markdown("**Try one of these:**")
    cols = st.columns(2)
    for i, sq in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(sq, key=f"sugg_{i}", use_container_width=True):
            st.session_state["pending_q"] = sq

    question = st.text_input(
        "Or type your own question:",
        value=st.session_state.get("pending_q", ""),
        placeholder="e.g. What do users forget about their photos?",
    )

    if question:
        # Clear pending so it doesn't stick forever
        if question == st.session_state.get("pending_q"):
            st.session_state["pending_q"] = ""

        answer_md, evidence_df = answer_question(question, df)
        st.markdown("---")
        st.markdown(answer_md)

        if evidence_df is not None and not evidence_df.empty:
            with st.expander(f"📎 Show {min(len(evidence_df), 10)} supporting quotes"):
                for _, row in evidence_df.head(10).iterrows():
                    st.markdown(
                        f"**{row['row_id']}** · *{row['source']}* · {row['sentiment']}"
                    )
                    st.markdown(f"> {row['quote_or_summary']}")
