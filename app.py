import re
import os
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="Photo Retrieval Discovery Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# GROQ CLIENT
# ============================================================
@st.cache_resource
def get_groq_client():
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


client = get_groq_client()
LLM_MODEL = "llama-3.3-70b-versatile"


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


# ============================================================
# LAYER 1: MEMORY CUES
# ============================================================
CUE_CATEGORIES = {
    "Person": ["dad", "mom", "mother", "father", "daughter", "son", "wife", "husband",
               "friend", "family", "baby", "child", "kid", "grandma", "grandpa",
               "jane", "john", "selfie"],
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


# ============================================================
# RAG: EMBEDDING + RETRIEVAL
# ============================================================
@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Building knowledge base...")
def build_index(df):
    """Embed every quote and cache the vectors."""
    embedder = load_embedder()
    docs = []
    for _, row in df.iterrows():
        text = f"{row['quote_or_summary']} | topic: {row['topic']} | source: {row['source']} | sentiment: {row['sentiment']}"
        docs.append(text)
    embeddings = embedder.encode(docs, show_progress_bar=False, normalize_embeddings=True)
    return docs, embeddings


def retrieve(query, docs, embeddings, df, k=6):
    """Hybrid retrieval: embedding similarity + keyword boost."""
    embedder = load_embedder()
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q_vec  # cosine (already normalized)

    # Keyword boost
    q_lower = query.lower()
    keywords = [w for w in re.findall(r"\b[a-z]{4,}\b", q_lower)
                if w not in {"what", "how", "does", "the", "and", "for", "with",
                             "about", "from", "they", "this", "that", "are",
                             "can", "you", "show", "give", "tell", "me", "some",
                             "any", "many", "users", "user", "photo", "photos"}]
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        for kw in keywords:
            if kw in doc_lower:
                scores[i] += 0.15

    top_idx = np.argsort(scores)[::-1][:k]
    return df.iloc[top_idx].copy(), scores[top_idx]


# ============================================================
# RAG: PROMPT + LLM
# ============================================================
SYSTEM_PROMPT = """You are the **Photo Retrieval Discovery Engine** — an expert analyst who answers questions about user feedback on photo retrieval (finding, organizing, searching, backing up photos).

You ONLY answer questions about:
- photo retrieval, search, recall, memory
- face recognition, people search, albums, folders
- backup, storage, data loss, recovery
- UI changes affecting photo discovery
- user workarounds, frustrations, feature requests
- the dataset: sources, sentiment, records, trends

If the user asks something OFF-TOPIC (weather, coding, general knowledge, math, personal advice, etc.), you MUST politely refuse with this exact format:

"That's outside what I do. I'm a photo retrieval discovery engine — I analyze user feedback about finding, organizing, and searching photos.

Here are some things I can tell you:
- What are the top retrieval problems users report?
- What do users remember about their photos?
- What do users forget about their photos?
- How do users formulate searches?
- Which source complains most about AI search?
- Show me quotes about face recognition
- How many negative records do we have?

Try asking one of those, or something like 'show me quotes about old photos'."

Rules:
1. Ground EVERY claim in the retrieved evidence below. Cite row IDs like `[RP-01]`.
2. If the evidence doesn't cover the question, say so honestly — do not hallucinate.
3. Use markdown: bold for emphasis, bullet lists, blockquotes for raw quotes.
4. Be concise but substantive — 3-6 sentences or a short list for most answers.
5. When quoting a user, use a blockquote (`>`) and include the row ID and source.
"""


def build_context(df_subset, scores):
    lines = []
    for i, (_, row) in enumerate(df_subset.iterrows()):
        score = scores[i] if i < len(scores) else 0
        lines.append(
            f"[{row['row_id']}] (source={row['source']}, sentiment={row['sentiment']}, "
            f"topic={row['topic']}, relevance={score:.2f})\n"
            f'"{row["quote_or_summary"]}"'
        )
    return "\n\n".join(lines)


def ask_llm(question, df, docs, embeddings, history):
    """Full RAG pipeline: retrieve → build prompt → call Groq → return answer + evidence."""
    # 1. Retrieve
    evidence_df, scores = retrieve(question, docs, embeddings, df, k=8)

    # 2. Build context
    context = build_context(evidence_df, scores)

    # 3. Build messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add prior conversation (last 4 turns)
    for turn in history[-8:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add current question with retrieved context
    messages.append({
        "role": "user",
        "content": f"""Dataset summary: {len(df)} records from {df['source'].nunique()} sources. {int((df['sentiment']=='negative').sum())} are negative.

Retrieved evidence (top {len(evidence_df)} most relevant records):

{context}

---

User question: {question}

Answer using ONLY the evidence above. Cite row IDs. If off-topic, use the exact refusal format from your system prompt."""
    })

    # 4. Call Groq
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1200,
    )
    answer = response.choices[0].message.content

    return answer, evidence_df


# ============================================================
# WELCOME SCREEN + CHAT
# ============================================================
SUGGESTIONS = [
    "What are the top retrieval problems users report?",
    "What do users remember about their photos?",
    "What do users forget about their photos?",
    "How do users formulate searches?",
    "Which source complains most about AI search?",
    "Show me quotes about face recognition",
    "How many negative records do we have?",
    "What workarounds do users mention?",
]


def render_sidebar():
    with st.sidebar:
        st.header("📂 Data")
        uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                    label_visibility="collapsed")
        st.caption("Or use the bundled data.csv")
        st.divider()

        st.header("ℹ️ About")
        st.markdown(
            "This engine answers questions about **photo retrieval** using "
            "**RAG** over user feedback from Reddit, Google Play, App Store, "
            "and Google Support."
        )
        st.markdown(
            "**Powered by:** Groq (Llama 3.3 70B) + local embeddings  \n"
            "**Cost:** $0  \n"
            "**Scope:** Photo retrieval only"
        )
        st.divider()

        if st.button("🔄 Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        return uploaded


def render_welcome():
    """Landing screen shown before the first question."""
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h1 style="font-size: 2.6rem; margin-bottom: 0.3rem;">🔍 Photo Retrieval Discovery Engine</h1>
            <p style="font-size: 1.15rem; color: #888; max-width: 700px; margin: 0 auto;">
                Ask anything about how users find, remember, forget, and search for their photos.
                Grounded in real feedback from Reddit, Google Play, App Store, and Google Support.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 💬 Try asking")
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"welcome_sugg_{i}", use_container_width=True):
            st.session_state.pending_question = s
            st.rerun()

    st.markdown(
        """
        <div style="text-align: center; margin-top: 2rem; color: #666; font-size: 0.9rem;">
            ⚡ Powered by Groq + Llama 3.3 70B &nbsp;·&nbsp; 🔒 Runs on your data &nbsp;·&nbsp; 💸 Free
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat():
    """Render the full conversation history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("evidence") is not None and not msg["evidence"].empty:
                with st.expander(f"📎 {len(msg['evidence'])} supporting records"):
                    for _, row in msg["evidence"].head(6).iterrows():
                        st.markdown(
                            f"**{row['row_id']}** · *{row['source']}* · {row['sentiment']}"
                        )
                        st.markdown(f"> {row['quote_or_summary']}")


# ============================================================
# MAIN
# ============================================================
def main():
    uploaded = render_sidebar()
    df = load_data(uploaded)

    if df.empty:
        st.warning("⚠️ No data found. Upload a CSV from the sidebar.")
        st.stop()

    # Clean columns
    df["quote_or_summary"] = df["quote_or_summary"].fillna("").astype(str)
    df["topic"] = df["topic"].fillna("").astype(str)
    df["sentiment"] = df["sentiment"].fillna("neutral").astype(str).str.lower()
    df["workaround"] = df["workaround"].fillna("N").astype(str).str.upper()
    df["source"] = df["source"].fillna("unknown").astype(str)

    # Add derived columns
    df["problems"] = df["topic"].apply(get_problems)
    df["cues"] = df["quote_or_summary"].apply(extract_cues)
    df["query_form"] = df["quote_or_summary"].apply(detect_query_formulation)

    # Build retrieval index (cached)
    docs, embeddings = build_index(df)

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # API key check
    if client is None:
        st.error(
            "🔑 **Groq API key not found.**  \n"
            "Add `GROQ_API_KEY` to Streamlit secrets (Settings → Secrets) "
            "or set it as an environment variable."
        )
        st.stop()

    # Welcome screen vs chat
    if not st.session_state.messages:
        render_welcome()
    else:
        st.markdown("## 🔍 Photo Retrieval Discovery Engine")
        st.caption("Ask about photo retrieval — anything off-topic gets politely declined.")
        render_chat()

    # Chat input (always at the bottom)
    prompt = st.chat_input("Ask about photo retrieval...")

    # Handle suggestion click
    if st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None

    if prompt:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching evidence and thinking..."):
                try:
                    answer, evidence = ask_llm(
                        prompt, df, docs, embeddings,
                        st.session_state.messages[:-1],
                    )
                    st.markdown(answer)
                    if not evidence.empty:
                        with st.expander(f"📎 {len(evidence)} supporting records"):
                            for _, row in evidence.head(6).iterrows():
                                st.markdown(
                                    f"**{row['row_id']}** · *{row['source']}* · {row['sentiment']}"
                                )
                                st.markdown(f"> {row['quote_or_summary']}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "evidence": evidence,
                    })
                except Exception as e:
                    error_msg = f"⚠️ Error calling LLM: `{e}`"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "evidence": None,
                    })

        st.rerun()


if __name__ == "__main__":
    main()
