import os
import joblib
import re
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

# ✅ Load API Key from .env
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ✅ Define Flask App
app = Flask(__name__)

# ✅ Define tokenizer function
def tokenize_text(text):
    tokens = str(text).lower().split()
    return [word for word in tokens if word.isalnum()]

# ✅ Load trained vectorizer and model
VECTORIZER_PATH = "model/tfidf_vectorizer.joblib"
MODEL_PATH = "model/toxic_comment_model.joblib"

vectorizer = joblib.load(VECTORIZER_PATH)
vectorizer.tokenizer = tokenize_text  # Assign tokenizer manually
model = joblib.load(MODEL_PATH)

print("✅ Model and vectorizer loaded successfully!")

# ✅ Function to preprocess text
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\S+", "", text)  # Remove mentions
    text = re.sub(r"#\S+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation
    return text.lower().strip()

# ✅ Function to extract video ID
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

# ✅ Function to fetch YouTube comments
def get_video_comments(video_id, max_results=20):
    comments = []
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        ).execute()

        for item in response["items"]:
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment_text)

    except Exception as e:
        print(f"❌ Error fetching comments: {e}")

    return comments

# ✅ Function to analyze YouTube comments
def check_toxic_comments(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return {"error": "Invalid YouTube URL."}

    print(f"🔵 Fetching comments from video: {video_url}")
    comments = get_video_comments(video_id)
    if not comments:
        return {"message": "⚠️ No comments found."}

    results = []
    for comment in comments:
        processed_comment = preprocess_text(comment)
        text_vector = vectorizer.transform([processed_comment])
        toxicity_prob = model.predict_proba(text_vector)[0][1]
        predicted_class = int(toxicity_prob > 0.5)

        results.append({
            "comment": comment,
            "toxicity_score": round(toxicity_prob, 2),
            "toxic": bool(predicted_class)
        })

    return {"comments": results}

# ✅ Flask Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index2.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    video_url = request.form.get("video_url")
    if not video_url:
        return jsonify({"error": "Missing video URL"}), 400
    
    analysis_result = check_toxic_comments(video_url)
    return jsonify(analysis_result)

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)