import streamlit as st
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder

st.title("🎓 AUPP FAQ Chatbot")
st.write("Ask me anything about AUPP!")

pipeline = joblib.load("pipeline_svm.joblib")
df_faq = joblib.load("faq_with_embeddings.joblib")
sbert = SentenceTransformer("all-MiniLM-L6-v2")  

# Initialize stopwords and lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()                     # lowercase
    text = re.sub(r'\d+', '', text)             # remove numbers
    text = re.sub(r'[^\w\s]', '', text)         # remove punctuation
    text = text.strip()                          # remove leading/trailing spaces

    # Tokenize
    words = word_tokenize(text)

    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Load Cross-Encoder for Reranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_chatbot(query):
    query = clean_text(query)

    # STEP 1 — Category Prediction
    pred_category = pipeline.predict([query])[0]
    df_cat = df_faq[df_faq["category"] == pred_category]

    # STEP 2 — SBERT Embedding
    query_emb = sbert.encode(query)

    embeddings = np.vstack(df_cat["embedding"].values)

    cosine_sim = embeddings @ query_emb / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    )

    # ---- TOP K SELECTION ----
    K = 5  # Retrieve top 5 candidates
    top_k_idx = np.argsort(cosine_sim)[-K:][::-1]  
    top_k_rows = df_cat.iloc[top_k_idx]

    # STEP 3 — PREPARE PAIRS FOR CROSS-ENCODER RERANKING
    pairs = [(query, row["question"]) for _, row in top_k_rows.iterrows()]

    # STEP 4 — CROSS-ENCODER SCORES
    cross_scores = cross_encoder.predict(pairs)

    # Final selected FAQ after reranking
    best_idx = np.argmax(cross_scores)
    faq_row = top_k_rows.iloc[best_idx]

    final_similarity = float(cosine_sim[top_k_idx[best_idx]])

    return {
        "category": faq_row["category"],
        "answer": faq_row["answer"],
        "similarity": final_similarity
    }

user_query = st.text_input("Enter your question here:")

if user_query:
    result = hybrid_chatbot(user_query)

    st.write("### 🔍 Prediction Result")
    st.write(f"**Predicted Category:** `{result['category']}`")
    st.write(f"**Similarity Score:** `{result['similarity']:.4f}`")

    if result['similarity'] > 0.6:
        st.success(result["answer"])
    else:
        st.success("Please reach out to our Academic Admission Officer for more information.")
        st.error(result["answer"])
    
