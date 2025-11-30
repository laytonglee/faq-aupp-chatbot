import streamlit as st
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

st.title("🎓 AUPP FAQ Chatbot")
st.write("Ask me anything about AUPP!")

pipeline = joblib.load("pipeline_svm.joblib")
df_faq = joblib.load("faq_with_embeddings.joblib")
sbert = SentenceTransformer("all-MiniLM-L6-v2")    


def hybrid_chatbot(query):

    pred_category = pipeline.predict([query])[0]

    df_cat = df_faq[df_faq["category"] == pred_category]

    query_emb = sbert.encode(query)

    embeddings = np.vstack(df_cat["embedding"].values)
    cosine_sim = embeddings @ query_emb / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    idx = np.argmax(cosine_sim)
    faq_row = df_cat.iloc[idx]

    return {
        "category": faq_row["category"],
        "answer": faq_row["answer"],
        "similarity": float(cosine_sim[idx])
    }

user_query = st.text_input("Enter your question here:")

if user_query:
    result = hybrid_chatbot(user_query)

    st.write("### 🔍 Prediction Result")
    st.write(f"**Predicted Category:** `{result['category']}`")
    st.write(f"**Similarity Score:** `{result['similarity']:.4f}`")

    st.success(result["answer"])
