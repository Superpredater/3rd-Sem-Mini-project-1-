from fastapi import FastAPI, Response, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import joblib
import os
import json
import csv
from datetime import datetime
import re
import secrets
import base64
from redis import Redis
import subprocess
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from openai import OpenAI, RateLimitError
import requests
from bs4 import BeautifulSoup

# =====================================================
# APP
# =====================================================
app = FastAPI()

# =====================================================
# OPENAI CLIENT
# =====================================================
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# =====================================================
# REDIS SESSION STORE
# =====================================================
redis_db = Redis(host="localhost", port=6379, db=0)
SESSION_EXPIRY = 86400  # 1 day

def create_session(email: str):
    session_id = secrets.token_hex(32)
    redis_db.setex(f"session:{session_id}", SESSION_EXPIRY, email)
    return session_id

def get_session(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        return None
    email = redis_db.get(f"session:{session_id}")
    return email.decode() if email else None

def destroy_session(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id:
        redis_db.delete(f"session:{session_id}")

def get_chat_history(email: str):
    key = f"chat:{email}"
    history = redis_db.get(key)
    return json.loads(history) if history else []

def save_chat_history(email: str, history):
    redis_db.setex(f"chat:{email}", SESSION_EXPIRY, json.dumps(history))

# =====================================================
# URL TEXT EXTRACTION (ADDED, NOTHING REMOVED)
# =====================================================
def extract_text_from_url(url: str) -> str:
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = " ".join(p.get_text() for p in soup.find_all("p"))
        return text[:5000]
    except:
        return ""

# =====================================================
# EXPLAINABLE AI MESSAGE
# =====================================================
def generate_message(user, text, label, confidence):

    if label == "CHAT":
        return (
            "Hey 👋\n\n"
            "You can paste a news headline, article text, or URL.\n"
            "I will tell you whether it is Fake or Real and explain why."
        )

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an explainable AI assistant for fake news detection. "
                        "Explain clearly:\n"
                        "1) Why the news is FAKE or REAL\n"
                        "2) Why the confidence percentage has that value\n"
                        "Use simple language."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"News:\n{text}\n\n"
                        f"Prediction: {label}\n"
                        f"Confidence: {confidence}%"
                    )
                }
            ],
            max_tokens=250
        )
        return res.choices[0].message.content

    except RateLimitError:
        if label == "FAKE":
            return (
                "This news is classified as FAKE because it contains sensational language, "
                "exaggerated claims, or emotionally manipulative wording.\n\n"
                f"The confidence is {confidence}% because multiple fake-news indicators were detected."
            )
        else:
            return (
                "This news is classified as REAL because the language appears factual, "
                "neutral, and similar to credible news reporting.\n\n"
                f"The confidence is {confidence}% because very few suspicious indicators were found."
            )
def is_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")

def is_conversation(text: str):
    # URLs are NOT conversation
    if is_url(text):
        return False

    return len(text.split()) <= 3


# =====================================================
# GOOGLE LOGIN CONFIG
# =====================================================
GOOGLE_CLIENT_ID = "YOUR_GOOGLE_CLIENT_ID_HERE"

@app.get("/api/google/login")
def google_login():
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        "?response_type=code"
        f"&client_id={GOOGLE_CLIENT_ID}"
        "&redirect_uri=http://localhost:5000/api/google/callback"
        "&scope=openid%20email%20profile"
        "&access_type=offline"
        "&prompt=select_account"
    )
    return {"url": google_auth_url}

@app.get("/api/google/callback")
def google_callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        return {"success": False}

    token_url = "https://oauth2.googleapis.com/token"
    payload = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": "YOUR_CLIENT_SECRET",
        "redirect_uri": "http://localhost:5000/api/google/callback",
        "grant_type": "authorization_code"
    }

    token_res = requests.post(token_url, data=payload).json()
    idinfo = id_token.verify_oauth2_token(
        token_res["id_token"],
        google_requests.Request(),
        GOOGLE_CLIENT_ID
    )

    email = idinfo["email"]
    users = load_users()

    if not any(u["email"] == email for u in users):
        users.append({
            "name": idinfo.get("name", "Google User"),
            "email": email,
            "phone": "",
            "password": "",
            "occupation": "Google Login",
            "profileImage": idinfo.get("picture", "")
        })
        save_users(users)

    session_id = create_session(email)
    resp = JSONResponse({"success": True})
    resp.set_cookie("session_id", session_id, httponly=True)
    return resp

# =====================================================
# CORS
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null"   # REQUIRED for file:// fake5.html
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# MODELS
# =====================================================
class DetectRequest(BaseModel):
    text: str | None = None
    url: str | None = None

class SignupModel(BaseModel):
    name: str
    email: str
    phone: str
    password: str
    occupation: str
    profileImage: str = ""

class LoginModel(BaseModel):
    identifier: str
    password: str

class UpdateProfileModel(BaseModel):
    name: str
    email: str
    phone: str
    occupation: str
    profileImage: str = ""

# =====================================================
# FILE PATHS
# =====================================================
MODEL_PATH = "fake_news_model.joblib"
USERS_PATH = "users.json"

# =====================================================
# USER DATABASE
# =====================================================
if not os.path.exists(USERS_PATH):
    with open(USERS_PATH, "w") as f:
        json.dump([], f)

def load_users():
    with open(USERS_PATH, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=4)

# =====================================================
# AUTH ROUTES
# =====================================================
@app.post("/api/signup")
def signup(user: SignupModel):
    users = load_users()
    for u in users:
        if u["email"] == user.email:
            return {"success": False}
    users.append(user.dict())
    save_users(users)
    return {"success": True}

@app.post("/api/login")
def login(data: LoginModel):
    users = load_users()
    for u in users:
        if data.identifier in [u["email"], u["phone"], u["name"]]:
            if data.password == u["password"]:
                session_id = create_session(u["email"])
                resp = JSONResponse({"success": True, "user": u})
                resp.set_cookie("session_id", session_id, httponly=True)
                return resp
    return {"success": False}

@app.get("/api/me")
def me(request: Request):
    email = get_session(request)
    if not email:
        return {"loggedIn": False}
    for u in load_users():
        if u["email"] == email:
            return {"loggedIn": True, "user": u}
    return {"loggedIn": False}

@app.post("/api/logout")
def logout(request: Request):
    destroy_session(request)
    resp = JSONResponse({"success": True})
    resp.delete_cookie("session_id")
    return resp

@app.post("/api/update-profile")
def update_profile(data: UpdateProfileModel):
    users = load_users()
    for u in users:
        if u["email"] == data.email:
            u.update(data.dict())
            save_users(users)
            return {"success": True, "user": u}
    return {"success": False}

@app.post("/api/upload-profile-image")
def upload_profile_image(file: UploadFile = File(...)):
    return {"base64": base64.b64encode(file.file.read()).decode()}

# =====================================================
# FAKE NEWS LOGIC
# =====================================================
def rule_based_predict(text: str):
    t = text.lower()
    suspicious = [
        "shocking", "breaking", "miracle", "cure",
        "secret", "exposed", "urgent", "click here"
    ]
    score = sum(1 for w in suspicious if w in t) * 15
    confidence = min(score + 30, 95)
    label = "FAKE" if confidence >= 50 else "REAL"
    return label, confidence

# =====================================================
# DETECT API
# =====================================================
@app.post("/api/detect")
def detect(req: DetectRequest, request: Request):
    email = get_session(request) or "guest"

    # ---------------- CHAT MODE ----------------
    if req.text and is_conversation(req.text):
        return {
            "prediction": "CHAT",
            "message": generate_message(email, req.text, "CHAT", 0)
        }

    # ---------------- URL MODE (KEY FIX) ----------------
    if req.text and is_url(req.text):
        article_text = extract_text_from_url(req.text)

        if not article_text.strip():
            return {
                "prediction": "ERROR",
                "message": "⚠️ Unable to extract article content from this URL."
            }

        label, confidence = rule_based_predict(article_text)
        message = generate_message(email, article_text, label, confidence)

        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "isFake": label == "FAKE",
            "message": message
        }

    # ---------------- NORMAL TEXT NEWS ----------------
    text = req.text or ""
    label, confidence = rule_based_predict(text)
    message = generate_message(email, text, label, confidence)

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "isFake": label == "FAKE",
        "message": message
    }


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
