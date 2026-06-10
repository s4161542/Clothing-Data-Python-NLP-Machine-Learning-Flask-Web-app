# Clothing Shop Recommendation App

A Flask web app for browsing clothing items, reading reviews, searching by keyword, and predicting whether a new review recommends an item using a saved Logistic Regression model.

## Features

- Clothing catalogue home page with featured items
- Department category pages
- Keyword search across item title, description, department, and class
- Item detail pages with rating and recommendation statistics
- Customer review display
- Add-review flow with machine-learning recommendation prediction

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Lightweight local review-word embeddings
- Scikit-learn
- HTML, CSS, and JavaScript

## Run Locally

1. Create and activate a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Start the app.

```bash
python app.py
```

4. Open the local Flask URL, usually `http://127.0.0.1:5000`.

The app uses a compact local embedding cache, `embedding_cache.npz`, so deployment does not need to download a large external word-vector model.

## Deploy on Render

This project is ready to deploy as a Python web service on Render.

1. Push the project to GitHub.
2. In Render, choose **New** then **Web Service**.
3. Connect this GitHub repository.
4. Render can read `render.yaml` automatically. If entering settings manually, use:
   - Build command: `pip install --upgrade pip setuptools wheel && pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Add a `SECRET_KEY` environment variable if Render does not generate one automatically.

The app avoids downloading large machine-learning assets during deployment, so free hosting should start more reliably.
