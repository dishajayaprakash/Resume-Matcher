# app.py

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def extract_text_from_pdf(file):
    text = ''
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

def get_similarity_scores(job_desc, resumes_dict):
    job_embed = model.encode([job_desc])
    scores = {}
    for name, resume_text in resumes_dict.items():
        if resume_text.strip():
            resume_embed = model.encode([resume_text])
            score = cosine_similarity(job_embed, resume_embed)[0][0]
            scores[name] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Streamlit UI
st.title("📄🔍 Resume Matcher")
st.markdown("Match candidate resumes to a job description using embeddings.")

job_desc = st.text_area("Paste Job Description", height=200)

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Match Candidates"):
    if not job_desc or not uploaded_files:
        st.warning("Please provide both job description and resumes.")
    else:
        resumes = {file.name: extract_text_from_pdf(file) for file in uploaded_files}
        results = get_similarity_scores(job_desc, resumes)

        st.subheader("🏆 Top Matches")
        for name, score in results[:10]:
            st.markdown(f"**{name}** — Similarity Score: `{score:.4f}`")
