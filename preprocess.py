# preprocess.py

def tokenize_text(text):
    tokens = str(text).lower().split()
    return [word for word in tokens if word.isalnum()]
