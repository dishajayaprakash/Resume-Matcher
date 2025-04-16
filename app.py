import streamlit as st
from app.resume_parser import extract_text
from app.job_parser import extract_job_description
from sentence_transformers import SentenceTransformer, util

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="AI Resume Scanner", layout="centered")
st.title("Becky's AI Resume Scanner")
st.caption("Compare your resume to a job description using AI")

# Uploads
resume_file = st.file_uploader("Upload your Resume", type=["pdf", "docx", "txt"])
job_file = st.file_uploader("Upload Job Description", type=["txt"])

if resume_file and job_file:
    # Save uploaded files
    resume_path = f"temp_resume.{resume_file.name.split('.')[-1]}"
    job_path = "temp_job.txt"

    with open(resume_path, "wb") as f:
        f.write(resume_file.read())
    with open(job_path, "wb") as f:
        f.write(job_file.read())

    # Extract text
    resume_text = extract_text(resume_path)
    job_text = extract_job_description(job_path)

    # Semantic matching
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(resume_embedding, job_embedding)[0][0].item()
    match_score = round(similarity * 100, 2)

    # Display Match Score
    st.markdown("### 🤖 Semantic Match Score")
    if match_score >= 75:
        st.success(f"✅ Excellent match! Your resume aligns very well.\n**Score: {match_score}%**")
    elif match_score >= 50:
        st.info(f"🟡 Moderate match – some alignment.\n**Score: {match_score}%**")
    else:
        st.warning(f"🔴 Low match – your resume may not fit this job well.\n**Score: {match_score}%**")

    # Find missing keywords
    def find_missing_keywords(job_text, resume_text):
        job_words = set(job_text.lower().split())
        resume_words = set(resume_text.lower().split())
        missing = job_words - resume_words
        return sorted([w for w in missing if len(w) > 3])

    missing_keywords = find_missing_keywords(job_text, resume_text)

    # Display Missing Keywords
    st.markdown("### 🧩 Missing Keywords from Resume")
    if missing_keywords:
        st.write(", ".join(missing_keywords[:20]))
    else:
        st.success("No major keywords missing. Your resume is solid! 💪")
