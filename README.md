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
