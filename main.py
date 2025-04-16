from app.resume_parser import extract_text
from app.job_parser import extract_job_description
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Resume
resume_path = "data/resumes/Rebecca_Zhu.pdf"
resume_text = extract_text(resume_path)
print("\n--- RESUME TEXT ---\n")
print(resume_text[:1000])

# Job
job_path = "data/job_descriptions/ai_engineer.txt"
job_text = extract_job_description(job_path)
print("\n--- JOB DESCRIPTION TEXT ---\n")
print(job_text)

# Match Scoring
documents = [resume_text, job_text]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
match_score = round(similarity * 100, 2)

print(f"\n🔍 Match Score: {match_score}%")


def find_missing_keywords(job_text, resume_text):
    job_keywords = set(job_text.lower().split())
    resume_keywords = set(resume_text.lower().split())

    missing = job_keywords - resume_keywords

    # Filter out common short words and duplicates
    missing_filtered = [word for word in missing if len(word) > 3]
    return sorted(missing_filtered)

missing_keywords = find_missing_keywords(job_text, resume_text)

print("\n🧩 Missing Keywords from Resume:")
print(", ".join(missing_keywords[:15]))  # show top 15



