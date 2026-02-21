import pdfplumber
import spacy
import sys
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_resume(resume_text, job_desc):
    if not resume_text.strip() or not job_desc.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform([resume_text, job_desc])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(score*100, 2)
    except ValueError:
        return 0.0

# Ensure the model is available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm since it's missing...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

SKILLS = ["python", "java", "c++", "sql", "html", "css", "machine learning", "flask"]

ALL_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "ruby", "php", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch", "scikit-learn",
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "git", "linux"
]

def extract_resume_text(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Failed to read PDF {path}: {e}")
        return ""
    return text

def extract_skills(path):
    text = extract_resume_text(path).lower()
    if not text:
        return 0, ""
    
    count = 0
    for skill in SKILLS:
        if skill in text:
            count += 1
            
    doc = nlp(text)
    found_skills = set()
    for token in doc:
        if token.text.lower() in ALL_SKILLS:
            found_skills.add(token.text.lower())
            
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        for skill in ALL_SKILLS:
            if skill in chunk_text:
                found_skills.add(skill)
                
    return count, ", ".join(sorted(list(found_skills)))
