# AI Interview Coach

A lightweight, NLP-based interview practice tool built with Streamlit. Scores
free-text answers against a model answer using TF-IDF + cosine similarity,
blended with a keyword-coverage check, and gives instant feedback — no manual
grading needed.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Files

- `app.py` — the Streamlit app (UI + scoring engine)
- `questions.json` — the question bank (18 questions across 4 categories;
  edit this file to add more questions, no code changes needed)
- `.streamlit/config.toml` — theme colors for the app
- `requirements.txt` — Python dependencies for local install or deployment

## Deploy

See deployment instructions provided alongside this project for deploying to
Streamlit Community Cloud (free, hosted, shareable link).
