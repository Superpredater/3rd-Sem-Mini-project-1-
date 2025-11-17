# news_predictor.py

import joblib

MODEL_PATH = "news_model.joblib"

# Load once at startup
model = joblib.load(MODEL_PATH)

def predict_news(text: str):
    """
    Input:
        text: str  -> news text / headline
    Output:
        dict like:
        {
          "prediction": "FAKE" or "REAL",
          "confidence": 87.32
        }
    """
    # Predict label
    label = model.predict([text])[0]

    # Predict probabilities to get confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        confidence = float(max(proba) * 100.0)
    else:
        confidence = 90.0  # fallback if no proba available

    return {
        "prediction": label,
        "confidence": round(confidence, 2)
    }

if __name__ == "__main__":
    # quick manual test
    sample = "SHOCKING! You won't believe what happened next!"
    result = predict_news(sample)
    print(result)
