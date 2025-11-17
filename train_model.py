import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATASET = "dynamic_dataset.csv"
MODEL_OUT = "fake_news_model.joblib"

def train_model():

    print("📥 Loading dataset...")
    df = pd.read_csv(DATASET)

    if "text" not in df.columns or "label" not in df.columns:
        raise Exception("dynamic_dataset.csv must contain text,label columns")

    X = df["text"]
    y = df["label"]

    print("🧪 Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🧠 Building model...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    print("🚀 Training...")
    model.fit(X_train, y_train)

    print("📊 Evaluating...")
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    print("💾 Saving model...")
    joblib.dump(model, MODEL_OUT)

    print("🎉 Training complete!")

if __name__ == "__main__":
    train_model()
