import os
import streamlit as st
from dotenv import load_dotenv

# -------------------------
# 1. Streamlit page configuration
# -------------------------
# Must be the first Streamlit command
st.set_page_config(layout='wide')

# -------------------------
# 2. Load environment variables
# -------------------------
load_dotenv()  # for local development; in Cloud will use st.secrets
OPENAI_API_KEY   = st.secrets.get('OPENAI_API_KEY')   or os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = st.secrets.get('PINECONE_API_KEY') or os.getenv('PINECONE_API_KEY')
INDEX_NAME       = st.secrets.get('PINECONE_INDEX')   or os.getenv('PINECONE_INDEX', 'sephora-review-full')

# -------------------------
# 3. Define singletons to delay heavy imports until needed
# -------------------------
@st.cache_resource
def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)

@st.experimental_singleton
def get_pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

@st.experimental_singleton
def load_cluster_summaries(path='cluster_summaries.json'):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.experimental_singleton
def load_cluster_map(path='cluster_map.csv'):
    import pandas as pd
    df = pd.read_csv(path, usecols=['id', 'cluster'])
    df['id'] = df['id'].astype(str)
    df['cluster'] = df['cluster'].astype(str)
    return dict(zip(df['id'], df['cluster']))

# -------------------------
# 4. UI Styling (light) and History init
# -------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-size:12px; }
.review-box { font-size:0.7em; border-bottom:1px dashed #ccc; margin-bottom:6px; padding-bottom:4px; }
.chat-history { max-height:400px; overflow-y:auto; border:1px solid #ccc; padding:6px; background-color:#f9f9f9; }
.stTextArea textarea { margin-bottom:0; font-size:12px !important; }
.submit-button-container button { margin-top:2px; width:100%; }
</style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# -------------------------
# 5. Query input form
# -------------------------
st.title("🧴 Sephora Review Chat with Clusters")
with st.form(key='query_form'):
    query     = st.text_area('', placeholder='Type your question...', height=70, label_visibility='collapsed')
    submitted = st.form_submit_button('🔍')

# -------------------------
# 6. Handle the search & chat (execute heavy work only on submit)
# -------------------------
if submitted and query:
    # instantiate clients and load data once
    client            = get_openai_client()
    index             = get_pinecone_index()
    cluster_summaries = load_cluster_summaries()
    cluster_map       = load_cluster_map()

    with st.spinner('Searching reviews and clusters...'):
        # 6.1 Embed user query
        query_embedding = client.embeddings.create(
            input=query,
            model='text-embedding-3-small'
        ).data[0].embedding

        # 6.2 Retrieve top matches from Pinecone
        matches = index.query(
            vector=query_embedding,
            top_k=100,
            include_metadata=True
        )

        # 6.3 Collect texts and cluster counts
        review_texts   = []
        cluster_counts = {}
        for match in matches['matches']:
            meta   = match['metadata']
            vec_id = meta.get('id', '')
            cid    = cluster_map.get(vec_id, '-1')
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
            brand = meta.get('brand', 'Unknown')
            name  = meta.get('product_name', 'Unknown Product')
            text  = meta.get('text', '')
            review_texts.append(
                f"<b>Cluster {cid}</b> | <b>{brand} - {name}:</b><br>{text}"
            )
        total = len(review_texts)

        # 6.4 Summarize top 3 clusters
        overview_lines = []
        for cid, count in sorted(cluster_counts.items(), key=lambda x: -x[1])[:3]:
            pct     = round(count/total*100, 1) if total else 0
            summary = cluster_summaries.get(cid, {}).get('summary', '')
            overview_lines.append(f"Cluster {cid}: {summary} ({pct}% of results)")

        # 6.5 Build context and call GPT
        context = "\n---\n".join(review_texts)
        system_prompt = (
            "You are a product review analyst for Sephora. "
            "Answer based only on the provided reviews and cluster summaries."
        )
        messages = [
            {'role':'system','content':system_prompt},
            {'role':'user','content':f"Cluster overview:\n{'\n'.join(overview_lines)}\n\nReviews:\n{context}\n\nQuestion: {query}"}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages
        )
        answer = response.choices[0].message.content

        # 6.6 Save to session history
        st.session_state.history.append({
            'question': query,
            'answer': answer,
            'reviews': review_texts,
            'cluster_overview': overview_lines
        })

# -------------------------
# 7. Render chat history & reviews
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
        st.markdown(f"**Total reviews:** {len(last['reviews'])}")
        st.markdown('**Cluster distribution:**')
        for line in last['cluster_overview']:
            st.markdown(f"- {line}")
        for r in last['reviews']:
            st.markdown(f"<div class='review-box'>{r}</div>", unsafe_allow_html=True)
