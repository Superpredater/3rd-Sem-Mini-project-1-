# ml_model.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "fake_news_model.joblib"
DATASET_PATH = "dynamic_dataset.csv"   # Your existing dataset file


# ============================================================
#                TRAINING FUNCTION
# ============================================================
def train_ml_model():

    if not os.path.exists(DATASET_PATH):
        raise Exception("Dataset not found! Run your bot first to generate dynamic_dataset.csv.")

    print("📥 Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise Exception("CSV must contain 'text' and 'label' columns.")

    # Text and Labels
    X = df["text"].astype(str)
    y = df["label"].astype(str).str.upper()

    print("🧪 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("🧠 Building ML Pipeline (TF-IDF + Logistic Regression)...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    print("🚀 Training the model...")
    model.fit(X_train, y_train)

    print("📊 Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🔥 Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    print("💾 Saving trained model...")
    joblib.dump(model, MODEL_PATH)

    print("🎉 Training complete!")


# ============================================================
#                LOADING MODEL
# ============================================================
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("⚠️ No ML model found! Train one using train_ml_model().")
        return None
    return joblib.load(MODEL_PATH)


# ============================================================
#                PREDICTION FUNCTION
# ============================================================
def predict_ml(text: str):

    model = load_model()
    if model is None:
        return {
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "error": "ML model not trained yet"
        }

    label = model.predict([text])[0]

    # Confidence score
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        confidence = float(max(proba) * 100)
    else:
        confidence = 90.0

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "isFake": label.upper() == "FAKE"
    }


# ============================================================
#                TEST (OPTIONAL)
# ============================================================
if __name__ == "__main__":
    print("Testing ML model prediction:")
    sample = "Breaking miracle cure discovered!"
    print(predict_ml(sample))
