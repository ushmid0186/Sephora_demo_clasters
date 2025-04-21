import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# -------------------------
# 1. Streamlit page configuration
# -------------------------
# Must be the first Streamlit command to set layout before any UI
st.set_page_config(layout='wide')

# -------------------------
# 2. Load environment variables
# -------------------------
# Local .env for development; in Streamlit Cloud use st.secrets
load_dotenv()
OPENAI_API_KEY   = st.secrets.get('OPENAI_API_KEY')   or os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = st.secrets.get('PINECONE_API_KEY') or os.getenv('PINECONE_API_KEY')
INDEX_NAME       = st.secrets.get('PINECONE_INDEX')   or os.getenv('PINECONE_INDEX', 'sephora-review-full')

# -------------------------
# 3. Initialize clients
# -------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
pc     = Pinecone(api_key=PINECONE_API_KEY)
index  = pc.Index(INDEX_NAME)

# -------------------------
# 4. Load cluster metadata
# -------------------------
@st.cache_data
def load_cluster_summaries(path='cluster_summaries.json'):
    """
    Load cluster summaries (id → summary, examples, top brands/products) from JSON.
    Cached to avoid reloading on each rerun.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_cluster_map(path='cluster_map.csv'):
    """
    Load mapping from review vector_id to cluster_id.
    Returns dict: { 'P123_rev_045': '3', ... }
    """
    df_map = pd.read_csv(path)
    df_map['id'] = df_map['id'].astype(str)
    df_map['cluster'] = df_map['cluster'].astype(str)
    return dict(zip(df_map['id'], df_map['cluster']))

cluster_summaries = load_cluster_summaries()
cluster_map       = load_cluster_map()

# -------------------------
# 5. UI Styling and History init
# -------------------------
st.markdown("""
    <style>
    html, body, [class*="css"] {
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

def init_history():
    if 'history' not in st.session_state:
        st.session_state.history = []
init_history()

# -------------------------
# 6. Query input form
# -------------------------
st.title("🧴 Sephora Review Chat with Clusters")
with st.form(key='query_form'):
    query     = st.text_area('', placeholder='Type your question...', height=70, label_visibility='collapsed')
    submitted = st.form_submit_button('🔍')

# -------------------------
# 7. Handle the search & chat
# -------------------------
if submitted and query:
    with st.spinner('Searching reviews and clusters...'):
        # 7.1 Embed query
        query_embedding = client.embeddings.create(
            input=query,
            model='text-embedding-3-small'
        ).data[0].embedding

        # 7.2 Fetch top matches
        matches = index.query(
            vector=query_embedding,
            top_k=100,
            include_metadata=True
        )

        # 7.3 Collect reviews + cluster counts
        review_texts   = []
        cluster_counts = {}
        for match in matches['matches']:
            meta    = match['metadata']
            vec_id  = meta.get('id', '')
            cid     = cluster_map.get(vec_id, '-1')
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            brand   = meta.get('brand', 'Unknown')
            name    = meta.get('name', 'Unknown')
            text    = meta.get('text', '')
            review_texts.append(
                f"<b>Cluster {cid}</b> | <b>{brand} - {name}:</b><br>{text}"
            )
        total = len(review_texts)

        # 7.4 Top 3 cluster summaries
        overview_lines = []
        for cid, count in sorted(cluster_counts.items(), key=lambda x: -x[1])[:3]:
            pct = round(count/total*100,1)
            summary = cluster_summaries.get(cid, {}).get('summary','')
            overview_lines.append(f"Cluster {cid}: {summary} ({pct}% of results)")

        # 7.5 Build context and call GPT
        context = "\n---\n".join(review_texts)
        system_prompt = (
            "You are a product review analyst for Sephora. "
            "Answer based only on the provided reviews and cluster summaries."
        )
        messages = [
            {'role':'system','content':system_prompt},
            {'role':'user','content':f"Cluster overview:\n{chr(10).join(overview_lines)}\n\nReviews:\n{context}\n\nQuestion: {query}"}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages
        )
        answer = response.choices[0].message.content
        st.session_state.history.append({
            'question': query,
            'answer': answer,
            'reviews': review_texts,
            'cluster_overview': overview_lines
        })

# -------------------------
# 8. Render history & reviews on UI
# -------------------------
left, right = st.columns([2,1])
with left:
    st.markdown('### Chat history')
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for entry in reversed(st.session_state.history):
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer']}")
    st.markdown('</div>', unsafe_allow_html=True)
with right:
    st.markdown('### 📝 Reviews & clusters')
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.markdown(f"**Total reviews:** {len(last['reviews'])}")
        st.markdown('**Cluster distribution:**')
        for line in last['cluster_overview']:
            st.markdown(f"- {line}")
        for r in last['reviews']:
            st.markdown(f"<div class='review-box'>{r}</div>", unsafe_allow_html=True)
