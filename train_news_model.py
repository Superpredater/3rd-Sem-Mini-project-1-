# train_news_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_PATH = "news_data.csv"        # your dataset
MODEL_PATH = "news_model.joblib"   # output model file

def main():
    # 1. Load data
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Ensure required columns
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # Clean labels to be SAFE
    df["label"] = df["label"].str.upper().str.strip()
    X = df["text"]
    y = df["label"]

    # 2. Split data
    print("🧪 Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Build model pipeline
    print("🧠 Building model...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,     # limit vocabulary size
            ngram_range=(1, 2),    # unigrams + bigrams
            stop_words="english"   # remove common English words
        )),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    # 4. Train
    print("🚀 Training...")
    model.fit(X_train, y_train)

    # 5. Evaluate
    print("📊 Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # 6. Save model
    print(f"💾 Saving model to {MODEL_PATH} ...")
    joblib.dump(model, MODEL_PATH)
    print("🎉 Done!")

if __name__ == "__main__":
    main()
