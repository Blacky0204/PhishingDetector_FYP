from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from datetime import timedelta
from urllib.parse import urlparse
from typing import List, Optional
import json
import os
import joblib

from feature_extraction import extract_features
from db import (
    init_database,
    add_search_history,
    get_user_search_history,
    create_user,
    get_user_by_username,
    get_user_by_email,
    get_user_by_id,
    SQLiteDB
)
from auth import (
    get_current_user,
    create_access_token,
    authenticate_user,
    get_password_hash,
    validate_email
)
from schemas import UserCreate, UserLogin, Token, UserResponse, URLItem, FrontendLogItem

app = FastAPI(title="Phishing Detection API", version="render-full-1.0")

# -----------------------------
# CORS (for FYP demo)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Startup: DB init + Model load
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")

model = None
MODEL_COLUMNS = None
load_err = ""

@app.on_event("startup")
def startup_event():
    # DB
    try:
        init_database()
        print("[INFO] Database initialized")
    except Exception as e:
        # Don’t crash the whole app; keep endpoints alive
        print(f"[WARN] Database init failed: {e}")

    # Model
    global model, MODEL_COLUMNS, load_err
    try:
        pkg = joblib.load(MODEL_PATH)
        model = pkg["model"]
        MODEL_COLUMNS = list(pkg["columns"])
        print(f"[INFO] Loaded model with {len(MODEL_COLUMNS)} columns.")
    except Exception as e:
        load_err = str(e)
        model = None
        MODEL_COLUMNS = None
        print(f"[ERROR] Failed to load model: {load_err}")

# -----------------------------
# Helpers
# -----------------------------
def to_model_frame(features: dict):
    """Build DataFrame row aligned to training columns."""
    import pandas as pd
    if not MODEL_COLUMNS:
        raise RuntimeError("MODEL_COLUMNS not loaded.")
    row = {col: features.get(col, 0) for col in MODEL_COLUMNS}
    return pd.DataFrame([row], columns=MODEL_COLUMNS)

def _save_history(url: str, label: int, prediction: str, confidence, explanations: List[str], user_id: Optional[int]):
    """Save record; never crash response if DB write fails."""
    try:
        return add_search_history(
            url=url,
            user_id=user_id,
            prediction=prediction,
            confidence=confidence,
            explanation=json.dumps(explanations or [])
        )
    except Exception as e:
        print(f"[WARN] Failed to save history: {e}")
        return None

def _build_response(url: str, prediction: str, label: int, confidence, explanations: List[str], top_features: List[dict] | None = None):
    return {
        "url": url,
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "explanation": explanations or [],
        "top_features": top_features or [],
    }

# -----------------------------
# Public routes / HTML serving
# -----------------------------
@app.get("/")
def home():
    # Your HTML redirects to login if no token anyway
    return RedirectResponse(url="/login.html")

@app.get("/health")
def health():
    # should not crash the service; helps Render health checks
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "db_ready": True,
        "model_error": load_err if model is None else None
    }

@app.get("/login.html", response_class=HTMLResponse)
def serve_login():
    p = os.path.join(BASE_DIR, "login.html")
    if not os.path.exists(p):
        return HTMLResponse("<h2>Login page not found</h2>", status_code=404)
    return FileResponse(p)

@app.get("/signup.html", response_class=HTMLResponse)
def serve_signup():
    p = os.path.join(BASE_DIR, "signup.html")
    if not os.path.exists(p):
        return HTMLResponse("<h2>Signup page not found</h2>", status_code=404)
    return FileResponse(p)

@app.get("/dashboard.html", response_class=HTMLResponse)
def serve_dashboard():
    p = os.path.join(BASE_DIR, "dashboard.html")
    if not os.path.exists(p):
        return HTMLResponse("<h2>Dashboard not found</h2>", status_code=404)
    return FileResponse(p)

# -----------------------------
# Auth endpoints (match UI flow)
# -----------------------------
@app.post("/auth/register", response_model=UserResponse)
def register(user: UserCreate):
    if not validate_email(user.email):
        raise HTTPException(status_code=400, detail="Invalid email format")
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if len(user.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

    if get_user_by_username(user.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    if get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    user_id = create_user(user.username, user.email, hashed_password)
    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to create user")

    db_user = get_user_by_id(user_id)
    if not db_user:
        raise HTTPException(status_code=500, detail="User created but not found")

    return {
        "id": db_user["id"],
        "username": db_user["username"],
        "email": db_user["email"],
        "created_at": db_user["created_at"]
    }

@app.post("/auth/login", response_model=Token)
def login(user_data: UserLogin):
    user = authenticate_user(user_data.username, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=60 * 24)
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user["username"],
        "user_id": user["id"]
    }

@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"]
    }

# -----------------------------
# Predict (AUTH REQUIRED) — matches dashboard.html
# -----------------------------
@app.post("/predict")
def predict_url(item: URLItem, current_user: dict = Depends(get_current_user)):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_err}")

    url = item.url.strip()
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    host_no_port = host.split(":")[0]
    bare_host = host_no_port[4:] if host_no_port.startswith("www.") else host_no_port
    parts = bare_host.split(".") if bare_host else []
    left_part = parts[0] if parts else ""

    explanations: List[str] = []
    top_features: List[dict] = []

    # Incomplete URL
    if (not bare_host) or (len(parts) < 2) or (len(left_part) < 2):
        result = "Incomplete URL (domain appears incomplete)"
        explanations.append("The domain looks incomplete. Please enter a full website address such as https://example.com.")
        label = 2
        conf = None
        _save_history(url, label, result, conf, explanations, current_user["id"])
        return _build_response(url, result, label, conf, explanations, top_features)

    # Strong phishing heuristic
    if "--" in bare_host or "--" in parsed.path:
        result = "Phishing (suspicious double dash in URL)"
        explanations.append("URL contains consecutive dashes '--', a pattern often seen in phishing links.")
        label = 1
        conf = None
        _save_history(url, label, result, conf, explanations, current_user["id"])
        return _build_response(url, result, label, conf, explanations, top_features)

    # Heuristic suspicious checks (soft)
    suspicious = False
    COMMON_TLDS = (".com", ".org", ".net", ".edu", ".gov", ".my", ".co", ".io")
    if not bare_host.endswith(COMMON_TLDS):
        suspicious = True
        explanations.append("Domain does not end with a common TLD (e.g., .com, .org, .net) – treat this URL with caution.")

    main_label = bare_host.split(".")[0]
    if "-" in main_label:
        suspicious = True
        explanations.append("Main part of the domain contains '-' which is sometimes used to imitate legitimate sites.")

    # ML prediction
    feats = extract_features(url)
    df = to_model_frame(feats)
    pred = int(model.predict(df)[0])
    conf = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        conf = float(proba[pred])

    if pred == 1:
        result = "Phishing"
    elif suspicious:
        result = "Potential phishing (suspicious URL pattern, model prediction: Safe)"
    else:
        result = "Safe"

    # Feature explanations
    try:
        if "NumSensitiveWords" in df.columns and df["NumSensitiveWords"].iloc[0] > 0:
            explanations.append("Contains words like login/secure/account, which are common in phishing pages.")
        if "NoHttps" in df.columns and df["NoHttps"].iloc[0] == 1:
            explanations.append("Not using HTTPS (connection not secure).")
        if "SubdomainLevel" in df.columns and df["SubdomainLevel"].iloc[0] >= 2:
            explanations.append("Has many subdomains, which can be used to mimic legitimate sites.")
        if "NumDash" in df.columns and df["NumDash"].iloc[0] >= 2:
            explanations.append("Contains several '-' characters, often used in spoofed domains.")
        if "IpAddress" in df.columns and df["IpAddress"].iloc[0] == 1:
            explanations.append("Uses an IP address instead of a normal domain name, which is suspicious.")
    except Exception:
        pass

    if not explanations:
        explanations.append("URL structure appears normal based on extracted features.")

    # Top feature snapshot
    try:
        for name in ["NoHttps", "NumSensitiveWords", "SubdomainLevel", "NumDash", "IpAddress", "UrlLength", "NumDots"]:
            if name in df.columns:
                val = df[name].iloc[0]
                try:
                    val = val.item()
                except Exception:
                    pass
                top_features.append({"name": name, "value": val})
    except Exception as e:
        print(f"[WARN] Failed building top_features: {e}")

    _save_history(url, pred, result, conf, explanations, current_user["id"])
    return _build_response(url, result, pred, conf, explanations, top_features)

# -----------------------------
# History endpoints (match UI)
# -----------------------------
@app.get("/history")
def get_history(limit: int = 50, current_user: dict = Depends(get_current_user)):
    rows = get_user_search_history(current_user["id"], limit)
    history_list = []
    for r in rows:
        try:
            expl = json.loads(r["explanation"]) if r.get("explanation") else []
        except Exception:
            expl = [str(r.get("explanation"))]

        history_list.append({
            "id": r.get("id"),
            "url": r.get("url"),
            "label": r.get("label"),
            "prediction": r.get("prediction"),
            "confidence": r.get("confidence"),
            "explanation": expl,
            "created_at": r.get("created_at"),
            "user_id": r.get("user_id")
        })
    return {"history": history_list}

@app.delete("/history/clear")
def clear_history(current_user: dict = Depends(get_current_user)):
    try:
        db = SQLiteDB()
        db.execute_query("DELETE FROM search_history WHERE user_id = ?", (current_user["id"],))
        db.close()
        return {"status": "ok", "message": "history cleared"}
    except Exception as e:
        print(f"[ERROR] Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear history")

@app.post("/history/log")
def log_frontend_result(item: FrontendLogItem, current_user: dict = Depends(get_current_user)):
    try:
        record_id = add_search_history(
            url=item.url,
            user_id=current_user["id"],
            prediction=item.prediction,
            confidence=1.0,
            explanation=json.dumps(item.explanation or [])
        )
        if not record_id:
            raise HTTPException(status_code=500, detail="Failed to save to database")
        return {"status": "ok", "message": "saved", "record_id": record_id}
    except Exception as e:
        print(f"[ERROR] Failed to save frontend history: {e}")
        raise HTTPException(status_code=500, detail="Failed to save frontend history")