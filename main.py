from app.resume_parser import extract_text
from app.job_parser import extract_job_description
from sentence_transformers import SentenceTransformer, util

# Load resume
resume_path = "data/resumes/Rebecca_Zhu.pdf"
resume_text = extract_text(resume_path)
print("\n--- RESUME TEXT ---\n")
print(resume_text[:1000])

# Load job description
job_path = "data/job_descriptions/ai_engineer.txt"
job_text = extract_job_description(job_path)
print("\n--- JOB DESCRIPTION TEXT ---\n")
print(job_text)

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode both texts
resume_embedding = model.encode(resume_text, convert_to_tensor=True)
job_embedding = model.encode(job_text, convert_to_tensor=True)

# Compute similarity
similarity = util.pytorch_cos_sim(resume_embedding, job_embedding)[0][0].item()
match_score = round(similarity * 100, 2)

print(f"\n🤖 Semantic Match Score (BERT): {match_score}%")

# Optional: Keyword comparison
def find_missing_keywords(job_text, resume_text):
    job_keywords = set(job_text.lower().split())
    resume_keywords = set(resume_text.lower().split())

    missing = job_keywords - resume_keywords
    missing_filtered = [word for word in missing if len(word) > 3]
    return sorted(missing_filtered)

missing_keywords = find_missing_keywords(job_text, resume_text)
print("\n🧩 Missing Keywords from Resume:")
print(", ".join(missing_keywords[:15]))  # Show top 15
