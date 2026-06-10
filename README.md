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
- Gensim GloVe embeddings
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

The app loads the `glove-wiki-gigaword-100` embedding model through Gensim. On a first run, Gensim may download and cache that model.

## Deploy on Render

This project is ready to deploy as a Python web service on Render.

1. Push the project to GitHub.
2. In Render, choose **New** then **Web Service**.
3. Connect this GitHub repository.
4. Render can read `render.yaml` automatically. If entering settings manually, use:
   - Build command: `pip install --upgrade pip setuptools wheel && pip install -r requirements.txt && python -c "import gensim.downloader as api; api.load('glove-wiki-gigaword-100')"`
   - Start command: `gunicorn app:app`
5. Add a `SECRET_KEY` environment variable if Render does not generate one automatically.

The first deployment may take a few minutes because the GloVe word embedding model is downloaded during the build.
