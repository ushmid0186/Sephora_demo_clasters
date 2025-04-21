import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# -------------------------
# 1. Streamlit page configuration (must be first)
# -------------------------
st.set_page_config(layout='wide')

# -------------------------
# 2. Load environment variables
# -------------------------
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
    Load cluster summaries (id → summary) from JSON.
    Cached to avoid reloading on every rerun.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_cluster_map(path='cluster_map.csv'):
    """
    Load mapping from review vector_id to cluster_id.
    Returns a dict: { 'P123_rev_045': '3', ... }
    """
    df = pd.read_csv(path, usecols=['id', 'cluster'])
    df['id'] = df['id'].astype(str)
    df['cluster'] = df['cluster'].astype(str)
    return dict(zip(df['id'], df['cluster']))

cluster_summaries = load_cluster_summaries()
cluster_map       = load_cluster_map()

# -------------------------
# 5. UI Styling and initialize history
# -------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-size:12px; }
.review-box { font-size:0.8em; border-bottom:1px dashed #ccc; margin-bottom:6px; padding-bottom:4px; }
.chat-history { max-height:400px; overflow-y:auto; border:1px solid #ccc; padding:6px; background-color:#f9f9f9; }
.stTextArea textarea { margin-bottom:0; font-size:12px !important; }
.submit-button-container button { margin-top:2px; width:100%; }
</style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# -------------------------
# 6. Query input form
# -------------------------
st.title("🧴 Sephora Review Chat with Clusters")
with st.form(key='query_form'):
    query = st.text_area('', placeholder='Type your question...', height=70, label_visibility='collapsed')
    submitted = st.form_submit_button('🔍')

# -------------------------
# 7. Handle user queries
# -------------------------
if submitted and query:
    with st.spinner('Searching reviews and clusters...'):
        # 7.1 – Create query embedding
        qe = client.embeddings.create(
            input=query,
            model='text-embedding-3-small'
        ).data[0].embedding

        # 7.2 – Query Pinecone for top 100
        result = index.query(
            vector=qe,
            top_k=100,
            include_metadata=True
        )

        # 7.3 – Prepare review texts + cluster counts
        review_texts   = []
        cluster_counts = {}
        for match in result['matches']:
            meta   = match['metadata']
            vec_id = meta.get('id', '')
            cid    = cluster_map.get(vec_id, '-1')
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

            # Always take review_text field for the review
            text = meta.get('review_text', '') or ''
            if not text:
                continue

            brand   = meta.get('brand', 'Unknown')
            product = meta.get('product_name', 'Unknown Product')
            review_texts.append(
                f"<b>Cluster {cid}</b> | <b>{brand} - {product}:</b><br>{text}"
            )
        total = len(review_texts)

        # 7.4 – Summarize top 3 clusters
        overview_lines = []
        for cid, count in sorted(cluster_counts.items(), key=lambda x: -x[1])[:3]:
            pct     = round(count/total*100, 1) if total else 0
            summary = cluster_summaries.get(str(cid), {}).get('summary', '')
            overview_lines.append(f"Cluster {cid}: {summary} ({pct}% of results)")

        # 7.5 – Build context & call GPT
        context = "\n---\n".join(review_texts)
        system_prompt = (
            "You are a product review analyst for Sephora. "
            "Answer based only on the provided reviews and cluster summaries."
        )
        messages = [
            {'role':'system', 'content':system_prompt},
            {'role':'user',   'content':f"Cluster overview:\n{'\n'.join(overview_lines)}\n\nReviews:\n{context}\n\nQuestion: {query}"}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages
        )
        answer = response.choices[0].message.content

        # 7.6 – Save to history
        st.session_state.history.append({
            'question': query,
            'answer': answer,
            'reviews': review_texts,
            'cluster_overview': overview_lines
        })

# -------------------------
# 8. Render chat history & reviews
# -------------------------
left, right = st.columns([2, 1])
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
        st.markdown(f"**Total reviews used:** {len(last['reviews'])}")
        st.markdown('**Cluster distribution:**')
        for line in last['cluster_overview']:
            st.markdown(f"- {line}")
        for txt in last['reviews']:
            st.markdown(f"<div class='review-box'>{txt}</div>", unsafe_allow_html=True)
