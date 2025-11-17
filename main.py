from fastapi import FastAPI, Response, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

app = FastAPI()

# -----------------------
# REDIS SESSION STORE
# -----------------------
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
    if email:
        return email.decode()
    return None

def destroy_session(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id:
        redis_db.delete(f"session:{session_id}")


# -----------------------
# GOOGLE LOGIN CONFIG
# -----------------------
GOOGLE_CLIENT_ID = "YOUR_GOOGLE_CLIENT_ID_HERE"


@app.get("/api/google/login")
def google_login():
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        "?response_type=token"
        f"&client_id={GOOGLE_CLIENT_ID}"
        "&redirect_uri=http://localhost:5000/api/google/callback"
        "&scope=openid%20email%20profile"
    )
    return {"url": google_auth_url}


@app.get("/api/google/callback")
def google_callback(request: Request):

    token = request.query_params.get("id_token")

    try:
        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except:
        return {"success": False, "message": "Google authentication failed"}

    email = idinfo["email"]
    name = idinfo.get("name", "Google User")
    picture = idinfo.get("picture", "")

    users = load_users()

    # auto-create user if not exists
    if not any(u["email"] == email for u in users):
        users.append({
            "name": name,
            "email": email,
            "phone": "",
            "password": "",
            "occupation": "Google Login",
            "profileImage": picture
        })
        save_users(users)

    # create Redis session
    session_id = create_session(email)

    resp = JSONResponse({"success": True})
    resp.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=SESSION_EXPIRY
    )
    return resp


# -----------------------
# CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "null"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# -----------------------
# MODELS
# -----------------------
class DetectRequest(BaseModel):
    text: str

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


# -----------------------
# PATHS
# -----------------------
MODEL_PATH = "fake_news_model.joblib"
CSV_PATH = "dynamic_dataset.csv"
HISTORY_PATH = "history.json"
USERS_PATH = "users.json"


# -----------------------
# USER DATABASE
# -----------------------
if not os.path.exists(USERS_PATH):
    with open(USERS_PATH, "w") as f:
        json.dump([], f)

def load_users():
    with open(USERS_PATH, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=4)


# -----------------------
# SIGNUP
# -----------------------
@app.post("/api/signup")
def signup(user: SignupModel):
    users = load_users()

    for u in users:
        if u["email"] == user.email or u["phone"] == user.phone:
            return {"success": False, "message": "User already exists"}

    users.append(user.dict())
    save_users(users)

    return {"success": True, "message": "Signup successful"}


# -----------------------
# LOGIN WITH REDIS SESSION
# -----------------------
@app.post("/api/login")
def login(data: LoginModel, response: Response):
    users = load_users()

    for u in users:
        if (
            data.identifier == u["email"]
            or data.identifier == u["phone"]
            or data.identifier.lower() == u["name"].lower()
        ):
            if data.password == u["password"]:

                # create Redis session
                session_id = create_session(u["email"])

                resp = JSONResponse({"success": True, "user": u})
                resp.set_cookie(
                    key="session_id",
                    value=session_id,
                    httponly=True,
                    secure=False,
                    samesite="lax",
                    max_age=SESSION_EXPIRY
                )
                return resp

            return {"success": False, "message": "Wrong password"}

    return {"success": False, "message": "User not found"}


# -----------------------
# AUTO-LOGIN / SESSION CHECK
# -----------------------
@app.get("/api/me")
def me(request: Request):
    email = get_session(request)

    if not email:
        return {"loggedIn": False}

    users = load_users()

    for u in users:
        if u["email"] == email:
            return {"loggedIn": True, "user": u}

    return {"loggedIn": False}


# -----------------------
# LOGOUT
# -----------------------
@app.post("/api/logout")
def logout(request: Request):
    destroy_session(request)
    resp = JSONResponse({"success": True})
    resp.delete_cookie("session_id")
    return resp


# -----------------------
# UPDATE PROFILE
# -----------------------
@app.post("/api/update-profile")
def update_profile(data: UpdateProfileModel):
    users = load_users()

    for u in users:
        if u["email"] == data.email:
            u.update(data.dict())
            save_users(users)
            return {"success": True, "user": u}

    return {"success": False, "message": "User not found"}


# -----------------------
# PROFILE IMAGE UPLOAD
# -----------------------
@app.post("/api/upload-profile-image")
def upload_profile_image(file: UploadFile = File(...)):
    content = base64.b64encode(file.file.read()).decode("utf-8")
    return {"base64": content}


# -----------------------
# FAKE NEWS DETECTION
# -----------------------
ml_model = None

if os.path.exists(MODEL_PATH):
    try:
        ml_model = joblib.load(MODEL_PATH)
    except:
        pass

def rule_based_predict(text: str):
    t = text.lower()

    suspicious = [
        "shocking","breaking","miracle","cure","secret",
        "you won't believe","exposed","bombshell","urgent",
        "alert","viral","fake","scam","click here"
    ]
    emotional = ["terrifying","amazing","unbelievable","disaster","panic","fear","rage"]
    promise = ["guaranteed","100% true","trust me","no evidence"]

    score = 0
    score += sum(1 for w in suspicious if w in t) * 18
    score += sum(1 for w in emotional if w in t) * 12
    score += len(re.findall(r"!!+|!{2,}", t)) * 10
    score += len(re.findall(r"\b[A-Z]{4,}\b", text)) * 8
    score += sum(1 for w in promise if w in t) * 15

    confidence = min(score, 100)
    label = "FAKE" if confidence >= 50 else "REAL"
    return label, confidence

def predict(text):
    if ml_model:
        try:
            label = ml_model.predict([text])[0]
            if hasattr(ml_model, "predict_proba"):
                proba = ml_model.predict_proba([text])[0]
                confidence = float(max(proba) * 100)
            else:
                confidence = 92.0
            return label, confidence
        except:
            pass

    return rule_based_predict(text)


@app.post("/api/detect")
def detect(req: DetectRequest):
    label, confidence = predict(req.text)

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "isFake": label == "FAKE"
    }


# -----------------------
# RUN SERVER
# -----------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
