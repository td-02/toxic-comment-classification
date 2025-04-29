from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

# Load the ML model and vectorizer
MODEL_PATH = "model/toxic_comment_model.joblib"
VECTORIZER_PATH = "model/tfidf_vectorizer.joblib"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model or vectorizer file not found.")
    model, vectorizer = None, None

# Text Preprocessing
def preprocess_comment(text):
    """Cleans text: removes URLs, mentions, hashtags, and non-alphabetic characters."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\S+", "", text)  # Remove mentions
    text = re.sub(r"#\S+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation
    return text.lower().strip()

@app.route("/", methods=["GET", "POST"])
def index():
    """Main page with form for user input."""
    if request.method == "POST":
        if model is None or vectorizer is None:
            return jsonify({"error": "Model or vectorizer not loaded. Check logs."}), 500

        text = request.form["text"]
        preprocessed_text = preprocess_comment(text)
        text_vector = vectorizer.transform([preprocessed_text])

        # Get prediction
        probability = model.predict_proba(text_vector)[0][1]  # Probability of being toxic (class 1)
        predicted_class = int(probability > 0.5)  # 1 if toxic, 0 if not

        return render_template("index.html", text=text, predicted_class=predicted_class, toxicity_probability=probability)

    return render_template("index.html")  # Load initial page

if __name__ == "__main__":
    app.run(debug=True)  # Run Flask app (use `debug=False` for production)
