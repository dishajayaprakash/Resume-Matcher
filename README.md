# AI Resume Scanner

A lightweight, AI-assisted resume screening tool built with Python. This project parses resumes and job descriptions, evaluates keyword alignment using TF-IDF and cosine similarity, and identifies skill or keyword gaps — enabling smarter resume tailoring for job applications.

## 🔍 Key Features

- ✅ Extracts text from resumes (PDF, DOCX, TXT)
- ✅ Parses job descriptions (TXT)
- ✅ Calculates a match score using TF-IDF + cosine similarity
- ✅ Identifies missing keywords from the resume
- 🧠 Built for extensibility (semantic matching, UI, or ATS-ready formats)

## 📁 Project Structure

\`\`\`
AI-Resume-Scanner/
├── app/
│   ├── resume_parser.py
│   ├── job_parser.py
│   └── __init__.py
├── data/
│   ├── resumes/
│   └── job_descriptions/
├── main.py
├── requirements.txt
└── README.md
\`\`\`

## ⚙️ How to Use

1. Clone the repository:
\`\`\`bash
git clone https://github.com/Becky0x01/AI-Resume-Scanner.git
cd AI-Resume-Scanner
\`\`\`

2. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Add your resume and job description (in \`data/\`)

4. Run the script:
\`\`\`bash
python3 main.py
\`\`\`

## 🧪 Example Output:

\`\`\`
🔍 Match Score: 83.42%
🧩 Missing Keywords from Resume:
aws, cloud, communication, nlp, deployment
\`\`\`

## 👩‍💻 Built By

Becky Zhu  
Aspiring AI/Cloud Engineer | Lifelong Learner  
[LinkedIn](https://www.linkedin.com/in/rebeccaiit) | [GitHub](https://github.com/Becky0x01)
