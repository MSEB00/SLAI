import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# ==== Initial Training Data ====
initial_data = [
    ("what time is it", "get_time"),
    ("what's the date", "get_date"),
    ("open notepad", "open_notepad"),
    ("open calculator", "open_calculator"),
    ("bye", "exit"),
]

# ==== Mood/Intent Keywords ====
moods = {
    "happy": ["great", "good", "awesome", "love", "excited"],
    "sad": ["sad", "tired", "upset", "depressed"],
    "angry": ["angry", "mad", "frustrated"],
    "curious": ["wonder", "curious", "think"],
    "confused": ["confused", "don’t understand", "lost"]
}

# ==== Vectorizer and Labels ====
vectorizer = TfidfVectorizer()
X_train_texts, y_train = zip(*initial_data)
X_train = vectorizer.fit_transform(X_train_texts)
label_to_index = {label: i for i, label in enumerate(set(y_train))}
y_train_idx = np.array([label_to_index[label] for label in y_train])

# ==== Model Path ====
MODEL_PATH = "slai_model.pkl"
VECTORIZER_PATH = "slai_vectorizer.pkl"
LABEL_MAP_PATH = "slai_label_map.pkl"

# ==== Neural Network ====
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_to_index = joblib.load(LABEL_MAP_PATH)
else:
    model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)
    model.fit(X_train, y_train_idx)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_to_index, LABEL_MAP_PATH)

# ==== Mood Detection ====
def detect_mood(text):
    text = text.lower()
    for mood, keywords in moods.items():
        if any(word in text for word in keywords):
            print(f"Detected mood: {mood}")
            return mood
    return "neutral"

# ==== Inference Function ====
def predict_intent(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    index_to_label = {v: k for k, v in label_to_index.items()}
    intent = index_to_label.get(pred, "unknown")
    mood = detect_mood(text)
    return intent, mood

# ==== Test ====
test_input = "I feel tired, what time is it?"
intent, mood = predict_intent(test_input)
(intent, mood)
