# AI Resume Screening & Candidate Ranking System

This is a Streamlit-based web application that allows recruiters to upload multiple resumes (PDF or image), input a job description (JD), and:

- Automatically extract text (even from images using OCR)
- Rank resumes based on similarity to the JD
- Visualize JD keyword matches/missing keywords
- Display personalized insights over the resumes

---

##  Features

- Upload PDF/image resumes
- OCR-based text extraction for image files
- Resume ranking based on cosine similarity with JD
- Visual insight charts showing missing JD keywords
- Simple UI powered by Streamlit

---

## 🛠️ Tech Stack

- Python
- Streamlit
- PyPDF2
- pytesseract
- scikit-learn
- matplotlib
- pandas
- Pillow (for image handling)

---

##  Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
````

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Make sure Tesseract is installed

* 🔗 Download: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* Add to PATH or update `pytesseract.pytesseract.tesseract_cmd` in your script.


## 4. Run the App

```bash
streamlit run main.py
```
---

## Future Improvements

* Add resume improvement suggestions
* Integrate with advanced BERT models for better ranking
* Export resume ranking reports
* Integration with ATS or HR systems

---

## Contact

Created by [Sri Sailesh Reddy Batchu](mailto:saileshreddysr3320@gmail.com)
Feel free to reach out for contributions, collaborations, or queries!

