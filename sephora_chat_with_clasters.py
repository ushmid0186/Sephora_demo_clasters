import os
import streamlit as st
from dotenv import load_dotenv

# 1. Page config — первым
st.set_page_config(layout='wide')

# 2. Load env
load_dotenv()
OPENAI_API_KEY   = os.getenv('OPENAI_API_KEY')   or st.secrets.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') or st.secrets.get('PINECONE_API_KEY')
INDEX_NAME       = os.getenv('PINECONE_INDEX', 'sephora-review-full') or st.secrets.get('PINECONE_INDEX')

# 3. Minimal styling
st.markdown("""
<style>
html, body, [class*="css"] { font-size:12px; }
.review-box { font-size:0.8em; border-bottom:1px dashed #ccc; padding:4px 0; }
.chat-history { max-height:300px; overflow-y:auto; border:1px solid #eee; padding:6px; }
</style>
""", unsafe_allow_html=True)

# 4. Init history
if 'history' not in st.session_state:
    st.session_state.history = []

# 5. Form
st.title("🧴 Sephora Review Chat")
with st.form(key='q'):
    query = st.text_area("", placeholder="Type your question...", height=80)
    submitted = st.form_submit_button("🔍")

# 6. On submit — do all heavy lifting
if submitted and query:
    with st.spinner("Loading clients, embeddings, clusters…"):
        # a) Imports here to avoid startup
        import json, pandas as pd
        from openai import OpenAI
        from pinecone import Pinecone

        # b) Init clients
        client = OpenAI(api_key=OPENAI_API_KEY)
        pc     = Pinecone(api_key=PINECONE_API_KEY)
        index  = pc.Index(INDEX_NAME)

        # c) Load cluster metadata
        with open("cluster_summaries.json","r",encoding="utf-8") as f:
            cluster_summaries = json.load(f)
        df_map = pd.read_csv("cluster_map.csv", usecols=['id','cluster'])
        cluster_map = dict(zip(df_map['id'].astype(str), df_map['cluster'].astype(str)))

        # d) Embed and query
        qe = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
        resp = index.query(vector=qe, top_k=100, include_metadata=True)

    # 7. Process results
    reviews = []
    counts = {}
    for m in resp["matches"]:
        md   = m["metadata"]
        vid  = md["id"]
        cid  = cluster_map.get(vid, "-1")
        counts[cid] = counts.get(cid,0)+1
        b    = md.get("brand","Unknown")
        p    = md.get("product_name","Unknown")
        t    = md.get("text","")
        reviews.append(f"<b>Cluster {cid}</b> | <b>{b} - {p}</b>: {t}")

    total = len(reviews)
    top3 = sorted(counts.items(), key=lambda x:-x[1])[:3]
    overview = [f"Cluster {c}: {cluster_summaries.get(str(c),{}).get('summary','')} ({round(cnt/total*100,1)}%)"
                for c,cnt in top3]

    # 8. Call ChatGPT
    with st.spinner("Calling GPT…"):
        ctx = "\n---\n".join(reviews)
        sp  = "You are a Sephora product review analyst. Answer strictly from reviews+clusters."
        msgs= [
            {"role":"system","content":sp},
            {"role":"user","content":f"Overview:\n{chr(10).join(overview)}\n\nReviews:\n{ctx}\n\nQ: {query}"}
        ]
        ans = client.chat.completions.create(model="gpt-4o", messages=msgs).choices[0].message.content

    # 9. Save and render
    st.session_state.history.append({"q":query,"a":ans,"ov":overview,"rev":reviews})

# 10. Display history + last reviews
col1, col2 = st.columns([2,1])
with col1:
    st.markdown("### Chat history")
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for e in reversed(st.session_state.history):
        st.markdown(f"**Q:** {e['q']}")
        st.markdown(f"**A:** {e['a']}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.markdown(f"**Total reviews:** {len(last['rev'])}")
        st.markdown("**Cluster distribution:**")
        for line in last["ov"]:
            st.markdown(f"- {line}")
        for r in last["rev"]:
            st.markdown(f"<div class='review-box'>{r}</div>", unsafe_allow_html=True)
