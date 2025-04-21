import os 
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "sephora-reviews"

# === Clients ===TEST3
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# === UI Setup ===
st.set_page_config(layout="wide")
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

# === Session state ===
if "history" not in st.session_state:
    st.session_state.history = []

# === Query input at the top ===
st.title("🧴 Sephora Review Chat")
with st.form(key="query_form"):
    query = st.text_area("", placeholder="Type your question...", height=70, label_visibility="collapsed")
    submitted = st.form_submit_button("🔍")

if submitted and query:
    with st.spinner("Searching reviews..."):
        query_embed = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        ).data[0].embedding

        matches = index.query(
            vector=query_embed,
            top_k=100,
            include_metadata=True
        )

        review_texts = []
        for match in matches["matches"]:
            metadata = match["metadata"]
            brand = metadata.get("brand", "Unknown Brand")
            name = metadata.get("name", "Unknown Product")
            text = metadata.get("text", "")
            if len(text.split()) <= 150:
                review_texts.append(f"<b>{brand} - {name}:</b><br>{text}")

#        context = "\n---\n".join([
#            match["metadata"].get("text", "")
#            for match in matches["matches"]
#            if len(match["metadata"].get("text", "").split()) <= 150
#        ])

        context = "\n---\n".join([f"{m['metadata'].get('brand', '')} - {m['metadata'].get('name', '')}: {m['metadata'].get('text', '')}" for m in matches["matches"] if len(m['metadata'].get("text", "").split()) <= 150])


        system_prompt = (
            "You are a product review analyst for Sephora. "
            "Answer based only on the provided customer reviews. Be specific and do not invent anything."
            "If possible, return a table showing how frequently issues or praises are mentioned, as counts and percentages."
            "Do not include LaTeX or formulas in your response."
        )

#        response = client.chat.completions.create(
#            model="gpt-4o",
#            messages=[
#                {"role": "system", "content": system_prompt},
#                {"role": "user", "content": f"Here are the reviews:\n{context}\n\nQuestion: {query}"}
#            ]
#        )



        # Подготовка истории из последних 3 сообщений
        previous_history = []
        for entry in st.session_state.history[-3:]:
            previous_history.append({"role": "user", "content": entry["question"]})
            previous_history.append({"role": "assistant", "content": entry["answer"]})

        # Собираем весь список сообщений
        messages = [
            {"role": "system", "content": system_prompt},
            *previous_history,
            {"role": "user", "content": f"Here are the reviews:\n{context}\n\nQuestion: {query}"}
        ]

        # Запрос к GPT
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        answer = response.choices[0].message.content
        st.session_state.history.append({
            "question": query,
            "answer": answer,
            "reviews": review_texts
        })

# === Layout ===
left, right = st.columns([1.6, 1.4])

with left:
    st.markdown("### Chat history")
    with st.container():
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for entry in st.session_state.history[::-1]:
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(f"**A:** {entry['answer']}")
        st.markdown('</div>', unsafe_allow_html=True)

#with right:
#    st.markdown("### 📝 Reviews used:")
#    if st.session_state.history:
#        for txt in st.session_state.history[-1]["reviews"]:
#            st.markdown(f"<div class='review-box'>{txt}</div>", unsafe_allow_html=True)


with right:
    st.markdown("### 📝 Reviews used:")
    if st.session_state.history:
        st.markdown(f"**Total reviews used:** {len(st.session_state.history[-1]['reviews'])}")
        for txt in st.session_state.history[-1]["reviews"]:
            st.markdown(f"<div class='review-box'>{txt}</div>", unsafe_allow_html=True)
