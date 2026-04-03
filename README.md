# toxic-comment-classification

A TF-IDF + scikit-learn classifier that detects toxic comments in real-time, with three deployment surfaces: a Flask web app for freeform text, a Twitter/X bot that scans replies on any tweet, and a YouTube bot that audits comments on any video.

---

## How it works

```
Raw comment text
    │
    ▼
Preprocessing  →  lowercase, strip URLs / mentions / hashtags / punctuation
    │
    ▼
TF-IDF vectorizer  (tokenize_text: alphanumeric whitespace-split tokens)
    │
    ▼
Trained classifier  →  predict_proba → toxicity score [0.0 – 1.0]
    │
    ▼
Threshold 0.5  →  toxic (1) / clean (0)
```

The model and vectorizer are serialized to `model/` via `joblib` and loaded at startup across all three interfaces.

---

## Deployment surfaces

### 1. `app.py` — Flask web app (single comment)
POST a comment through the HTML form (`templates/index.html`). Returns the toxicity probability and binary label rendered back on the page.

```bash
python app.py
# visit http://localhost:5000
```

### 2. `bot.py` — Twitter/X reply scanner
Paste any tweet URL. The bot fetches up to 50 replies via the Twitter v2 API (Tweepy), runs each through the classifier, and prints flagged comments with their toxicity scores.

```bash
python bot.py
# Enter a tweet URL: https://x.com/user/status/...
```

Requires a `TWITTER_BEARER_TOKEN` in `.env`. Note: Twitter's free API tier restricts `search_recent_tweets` access — a Basic or Pro plan is needed for reply fetching.

### 3. `bot2.py` — YouTube comment scanner (Flask API)
Accepts a YouTube video URL via a POST form (`templates/index2.html`). Fetches up to 20 top-level comments via the YouTube Data API v3, classifies each, and returns a JSON array with `comment`, `toxicity_score`, and `toxic` fields.

```bash
python bot2.py
# visit http://localhost:5000
# POST a YouTube URL to /analyze
```

Requires a `YOUTUBE_API_KEY` in `.env`.

---

## Setup

```bash
git clone https://github.com/td-02/toxic-comment-classification.git
cd toxic-comment-classification
pip install -r requirements.txt
```

Create a `.env` file:

```
TWITTER_BEARER_TOKEN=your_twitter_bearer_token   # for bot.py
YOUTUBE_API_KEY=your_youtube_data_api_key         # for bot2.py
```

The trained model files (`model/toxic_comment_model.joblib` and `model/tfidf_vectorizer.joblib`) must be present. These are generated from the training notebook in `model/`.

---

## Project structure

```
toxic-comment-classification/
├── app.py               # Flask web app — classify freeform text
├── bot.py               # Twitter/X reply scanner (CLI)
├── bot2.py              # YouTube comment scanner (Flask API)
├── preprocess.py        # Shared tokenizer (alphanumeric whitespace-split)
├── model/
│   ├── *.ipynb          # Training notebook (TF-IDF + classifier)
│   ├── toxic_comment_model.joblib
│   └── tfidf_vectorizer.joblib
├── templates/
│   ├── index.html       # Web UI for app.py
│   └── index2.html      # Web UI for bot2.py
└── requirements.txt
```

---

## Tech stack

- **scikit-learn** — TF-IDF vectorization + classification (Logistic Regression / SVM)
- **joblib** — model serialization
- **Flask** — web interface and REST endpoint
- **Tweepy** — Twitter v2 API client
- **google-api-python-client** — YouTube Data API v3
- **python-dotenv** — credential management

---

## Limitations

- The tokenizer (`preprocess.py`) is a simple whitespace splitter on alphanumeric tokens — no stemming, no stopword removal. More sophisticated NLP preprocessing would improve recall on obfuscated toxic text (e.g. `h4te`, `f**k`).
- Twitter reply fetching requires at minimum the Basic API tier. The free tier will return a 403 on `search_recent_tweets`.
- The classifier is binary (toxic / not toxic). Multi-label classification across toxicity subtypes (threat, insult, identity hate, etc.) would require a different training setup.
- No rate limiting or queue is implemented — large comment threads may hit API quotas.

---

## License

MIT
