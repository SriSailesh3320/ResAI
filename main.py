import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

# OCR path setup if needed (uncomment on Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# PDF text extraction
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# Image text extraction using OCR
def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

# Resume ranking function
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(job_description_vector.reshape(1, -1), resume_vectors).flatten()
    return cosine_similarities

# JD insight generation function
def generate_insights(job_description, resume_text):
    import re

    # Normalize and tokenize by word (excluding numbers, small words)
    def clean_tokenize(text):
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return set(words)

    jd_keywords = clean_tokenize(job_description)
    resume_keywords = clean_tokenize(resume_text)

    missing_keywords = jd_keywords - resume_keywords
    matched_keywords = jd_keywords & resume_keywords

    return matched_keywords, missing_keywords


# Unified bar plot for all resume insights
def plot_insights(insights_dict):
    resumes = list(insights_dict.keys())
    matched_counts = [len(insights_dict[name]['matched']) for name in resumes]
    missing_counts = [len(insights_dict[name]['missing']) for name in resumes]

    x = np.arange(len(resumes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, matched_counts, width, label='Matched Keywords', color='green')
    ax.bar(x + width/2, missing_counts, width, label='Missing Keywords', color='red')

    ax.set_xlabel('Resumes')
    ax.set_ylabel('Keyword Count')
    ax.set_title('JD Keyword Match vs Missing in Resumes')
    ax.set_xticks(x)
    ax.set_xticklabels(resumes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig)

# Streamlit UI
st.title("🧠 AI Resume Screening & Candidate Ranking System")

# Job Description
st.header("📄 Job Description")
job_description = st.text_area("Enter the job description")

# Resume Upload
st.header("📂 Upload Resumes (PDF/Image)")
uploaded_files = st.file_uploader("Upload PDF/Image files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and job_description:
    # Button to trigger processing
    if st.button("✅ Confirm Upload & Analyze"):
        resumes = []
        filenames = []

        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            elif file.type.startswith("image/"):
                text = extract_text_from_image(file)
            else:
                continue
            resumes.append(text)
            filenames.append(file.name)

        # Ranking
        scores = rank_resumes(job_description, resumes)
        results = pd.DataFrame({"Resume": filenames, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)

        st.subheader("📊 Ranked Resumes")
        st.write(results)

        # Insights for all
        st.subheader("🔍 Unified Resume Insights")
        insights = {}
        for i, resume_text in enumerate(resumes):
            matched, missing = generate_insights(job_description, resume_text)
            insights[filenames[i]] = {
                'matched': matched,
                'missing': missing
            }

        plot_insights(insights)
        st.caption("✅ Green bars indicate how many JD keywords were found in each resume. 🔴 Red bars show what’s missing.")
