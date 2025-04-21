import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# -------------------------
# 1. Load environment variables
# -------------------------
# Explicitly load the .env file for API keys
load_dotenv()
OPENAI_API_KEY   = st.secrets.get("OPENAI_API_KEY")   or os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
INDEX_NAME       = st.secrets.get("PINECONE_INDEX")   or os.getenv("PINECONE_INDEX", "sephora-review-full")

# -------------------------
# 2. Initialize clients
# -------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
pc     = Pinecone(api_key=PINECONE_API_KEY)
index  = pc.Index(INDEX_NAME)

# -------------------------
# 3. Load cluster metadata
# -------------------------
@st.cache_data
def load_cluster_summaries(path='cluster_summaries.json'):
    """
    Load cluster summaries (id → summary, examples, top brands/products) from JSON.
    Cached to avoid reloading on every rerun.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_cluster_map(path='clustered_reviews.csv'):
    """
    Load mapping from review vector_id to cluster_id.
    Returns a dict: { 'P123_rev_045': '3', ... }
    """
    df_map = pd.read_csv(path)
    # Ensure IDs are strings
    df_map['id'] = df_map['id'].astype(str)
    df_map['cluster'] = df_map['cluster'].astype(str)
    return dict(zip(df_map['id'], df_map['cluster']))

cluster_summaries = load_cluster_summaries()
cluster_map = load_cluster_map()

# -------------------------
# 4. Streamlit UI setup
# -------------------------
st.set_page_config(layout='wide')
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 12px;
    }
    .review-box {
        font-size: 0.7em;
        border-bottom: 1px dashed #ccc;
        margin-bottom: 6px;
        padding-bottom: 4px;
    }
    .chat-history {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 6px;
        background-color: #f9f9f9;
    }
    .stTextArea textarea {
        margin-bottom: 0;
        font-size: 12px !important;
    }
    .submit-button-container button {
        margin-top: 2px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history in session state
def init_history():
    if 'history' not in st.session_state:
        st.session_state.history = []
init_history()

# -------------------------
# 5. Query input
# -------------------------
st.title("🧴 Sephora Review Chat with Clusters")
with st.form(key='query_form'):
    query = st.text_area('', placeholder='Type your question...', height=70, label_visibility='collapsed')
    submitted = st.form_submit_button('🔍')

# -------------------------
# 6. Handle user query
# -------------------------
if submitted and query:
    with st.spinner('Searching reviews and clusters...'):
        # 6.1. Create query embedding
        query_embedding = client.embeddings.create(
            input=query,
            model='text-embedding-3-small'
        ).data[0].embedding

        # 6.2. Query Pinecone for nearest reviews
        matches = index.query(
            vector=query_embedding,
            top_k=100,
            include_metadata=True
        )

        # 6.3. Prepare review display and cluster stats
        review_texts = []
        cluster_counter = {}
        for match in matches['matches']:
            meta = match['metadata']
            vec_id = meta.get('id', '')
            # Lookup cluster for this review
            cluster_id = cluster_map.get(vec_id, '-1')
            cluster_counter[cluster_id] = cluster_counter.get(cluster_id, 0) + 1
            # Build display text including cluster info
            brand = meta.get('brand', 'Unknown Brand')
            name = meta.get('name', 'Unknown Product')
            text = meta.get('text', '')
            review_texts.append(
                f"<b>Cluster {cluster_id}</b> | <b>{brand} - {name}:</b><br>{text}"
            )

        # 6.4. Summarize cluster distribution
        total = len(matches['matches'])
        cluster_summary_lines = []
        for cid, count in sorted(cluster_counter.items(), key=lambda x: -x[1])[:3]:
            pct = round(count / total * 100, 1)
            summary = cluster_summaries.get(cid, {}).get('summary', '')
            cluster_summary_lines.append(f"Cluster {cid}: {summary} ({pct}% of results)")
        cluster_overview = "\n".join(cluster_summary_lines)

        # 6.5. Context for GPT
        context = "\n---\n".join(review_texts)
        system_prompt = (
            "You are a product review analyst for Sephora. "
            "Answer based only on the provided customer reviews and cluster summaries. "
            "Be specific and do not invent anything."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Cluster overview:\n{cluster_overview}\n\nReviews:\n{context}\n\nQuestion: {query}"}
        ]

        # 6.6. Request to GPT
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages
        )
        answer = response.choices[0].message.content

        # 6.7. Save to history
        st.session_state.history.append({
            'question': query,
            'answer': answer,
            'reviews': review_texts,
            'cluster_overview': cluster_summary_lines
        })

# -------------------------
# 7. Layout: chat history and reviews
# -------------------------
left, right = st.columns([1.6, 1.4])

with left:
    st.markdown('### Chat history')
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for entry in reversed(st.session_state.history):
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer']}")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('### 📝 Reviews used and clusters')
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.markdown(f"**Total reviews used:** {len(last['reviews'])}")
        st.markdown('**Cluster distribution:**')
        for line in last['cluster_overview']:
            st.markdown(f"- {line}")
        for txt in last['reviews']:
            st.markdown(f"<div class='review-box'>{txt}</div>", unsafe_allow_html=True)
