import os
import joblib
import tweepy
import re
import time
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

# Twitter API credentials
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# ✅ Define the tokenizer function
def tokenize_text(text):
    tokens = str(text).lower().split()
    return [word for word in tokens if word.isalnum()]

# ✅ Load trained vectorizer & model
VECTORIZER_PATH = "model/tfidf_vectorizer.joblib"
MODEL_PATH = "model/toxic_comment_model.joblib"

vectorizer = TfidfVectorizer(tokenizer=tokenize_text)  # Ensure correct tokenizer
vectorizer = joblib.load(VECTORIZER_PATH)  # Load trained vectorizer
model = joblib.load(MODEL_PATH)  # Load trained model

print("✅ Model and vectorizer loaded successfully!")

# Function to preprocess tweet text
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\S+", "", text)  # Remove mentions
    text = re.sub(r"#\S+", "", text)  # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation
    return text.lower().strip()

# Function to extract tweet ID from URL
def get_tweet_id(tweet_url):
    match = re.search(r"status/(\d+)", tweet_url)
    return match.group(1) if match else None

# Function to fetch a tweet and its replies
def check_toxic_comments(tweet_url):
    try:
        tweet_id = get_tweet_id(tweet_url)
        if not tweet_id:
            print("❌ Invalid Tweet URL!")
            return

        # ✅ Fetch the original tweet
        tweet = client.get_tweet(tweet_id, tweet_fields=["text"])
        if not tweet or not tweet.data:
            print("❌ Tweet not found!")
            return

        print(f"🔵 Analyzing comments on: {tweet.data['text']}\n")

        # ✅ Fetch replies (comments) for the tweet
        query = f"conversation_id:{tweet_id} -is:retweet"
        response = client.search_recent_tweets(query=query, tweet_fields=["text"], max_results=50)

        if not response or not response.data:
            print("No comments found.")
            return

        toxic_comments = []
        for reply in response.data:
            text = preprocess_text(reply.text)
            text_vector = vectorizer.transform([text])  # Vectorize text
            
            # ✅ Ensure compatibility with all models
            if hasattr(model, "predict_proba"):
                toxicity_prob = model.predict_proba(text_vector)[0][1]
                predicted_class = int(toxicity_prob > 0.5)
            else:
                predicted_class = model.predict(text_vector)[0]
                toxicity_prob = 1.0 if predicted_class == 1 else 0.0

            if predicted_class == 1:
                toxic_comments.append((reply.text, toxicity_prob))

        if toxic_comments:
            print("🚨 Toxic Comments Found:")
            for comment, score in toxic_comments:
                print(f"🔴 {comment} (Toxicity Score: {score:.2f})")
        else:
            print("✅ No toxic comments detected!")

    except tweepy.errors.Forbidden:
        print("❌ API access denied. Check your API plan.")
    except Exception as e:
        print(f"❌ Error: {e}")

# Run the function
if __name__ == "__main__":
    tweet_url = input("Enter a tweet URL: ")
    check_toxic_comments(tweet_url)
