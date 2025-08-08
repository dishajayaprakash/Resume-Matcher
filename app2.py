import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import numpy as np
import os

# ======== CACHED LOADERS ========
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

embedder = load_embedder()
summarizer = load_summarizer()

# ======== PDF TEXT EXTRACTION ========
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text.strip()

# ======== KEYWORD EXTRACTION ========
def extract_common_keywords(text1, text2, top_n=8):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text1, text2])
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores1 = tfidf.toarray()[0]
    scores2 = tfidf.toarray()[1]
    common_scores = scores1 + scores2
    top_indices = np.argsort(common_scores)[::-1][:top_n]
    return feature_names[top_indices]

# ======== AI SUMMARY GENERATION ========
def generate_ai_summary(job_desc, resume_text, similarity_score, rank):
    keywords = extract_common_keywords(job_desc, resume_text, top_n=8)
    reasoning = f"""
Match Score: {similarity_score:.2f}
Candidate Rank: {rank}

Key Overlaps: {", ".join(keywords)}

Observations:
- Candidate shows strong skills in {keywords[0]} and {keywords[1]}.
- Potential gaps in areas not listed above.
"""
    try:
        summary = summarizer(reasoning, max_length=60, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error generating AI summary: {e}"

# ======== MATCHING ========
def compute_similarity(text1, text2):
    vecs = embedder.encode([text1, text2])
    return cosine_similarity([vecs[0]], [vecs[1]])[0][0]

# ======== STREAMLIT APP ========
st.title("Resume–Job Description Matcher (Free & Local)")

job_desc = st.text_area("Paste the job description here:")
uploaded_resumes = st.file_uploader("Upload candidate resumes (PDFs)", type=["pdf"], accept_multiple_files=True)

if st.button("Analyze Matches") and job_desc and uploaded_resumes:
    results = []

    for idx, pdf in enumerate(uploaded_resumes, start=1):
        resume_text = extract_text_from_pdf(pdf)
        if not resume_text:
            continue

        score = compute_similarity(job_desc, resume_text)
        summary = generate_ai_summary(job_desc, resume_text, score, idx)

        results.append({
            "name": os.path.splitext(pdf.name)[0],
            "score": score,
            "summary": summary,
            "resume_text": resume_text
        })

    if results:
        results.sort(key=lambda x: x["score"], reverse=True)
        st.subheader(f"Top {len(results)} Candidates")
        for i, res in enumerate(results, start=1):
            st.markdown(f"**{i}. {res['name']}** — Match Score: {res['score']:.2f}")
            st.write(res["summary"])
            with st.expander("View resume preview"):
                st.text(res["resume_text"])
    else:
        st.warning("No valid results found. Please check your files.")

# ======== REQUIREMENTS.TXT ========
# Add these in your requirements.txt:
# streamlit
# pdfplumber
# sentence-transformers
# scikit-learn
# transformers
# torch
