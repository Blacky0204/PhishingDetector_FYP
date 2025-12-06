# main.py - UPDATED FOR SQLITE
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
from datetime import timedelta
from urllib.parse import urlparse
from typing import List, Optional
import pandas as pd
import joblib
import json
import os
import uvicorn

from feature_extraction import extract_features
from db import (
    init_database,  # Initialize database
    add_search_history,
    get_user_search_history,
    create_user,
    get_user_by_username,
    get_user_by_email,
    get_user_by_id
)
from auth import (
    get_current_user,
    create_access_token,
    authenticate_user,
    get_password_hash,
    verify_password,
    validate_email
)
from schemas import UserCreate, UserLogin, Token, UserResponse, URLItem, FrontendLogItem

app = FastAPI(title="Phishing Detection API", version="3.0")

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    init_database()
    print("[INFO] Database initialized")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
model = None
MODEL_COLUMNS: List[str] | None = None
load_err = ""

try:
    pkg = joblib.load("phishing_model.pkl")
    model = pkg["model"]
    MODEL_COLUMNS = list(pkg["columns"])
    print(f"[INFO] Loaded model with {len(MODEL_COLUMNS)} columns.")
except Exception as e:
    load_err = str(e)
    print(f"[ERROR] Failed to load model: {load_err}")


def to_model_frame(features_dict: dict) -> pd.DataFrame:
    if MODEL_COLUMNS is None:
        raise RuntimeError("Model columns not loaded.")
    
    df = pd.DataFrame([features_dict])
    for c in MODEL_COLUMNS:
        if c not in df.columns:
            df[c] = 0
    return df[MODEL_COLUMNS]

# -------------------------------------------------
# AUTHENTICATION ENDPOINTS (UPDATED FOR SQLITE)
# -------------------------------------------------
@app.post("/auth/register", response_model=UserResponse)
def register(user: UserCreate):
    """Register a new user"""
    # Validation
    if not validate_email(user.email):
        raise HTTPException(
            status_code=400, 
            detail="Invalid email format"
        )
    
    if len(user.password) < 6:
        raise HTTPException(
            status_code=400, 
            detail="Password must be at least 6 characters"
        )
    
    if len(user.username) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Username must be at least 3 characters"
        )
    
    # Check if username exists
    db_user = get_user_by_username(user.username)
    if db_user:
        raise HTTPException(
            status_code=400, 
            detail="Username already registered"
        )
    
    # Check if email exists
    db_user = get_user_by_email(user.email)
    if db_user:
        raise HTTPException(
            status_code=400, 
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    user_id = create_user(user.username, user.email, hashed_password)
    
    if not user_id:
        raise HTTPException(
            status_code=500, 
            detail="Failed to create user"
        )
    
    # Get created user
    db_user = get_user_by_id(user_id)
    if not db_user:
        raise HTTPException(
            status_code=500, 
            detail="User created but not found"
        )
    
    return {
        "id": db_user["id"],
        "username": db_user["username"],
        "email": db_user["email"],
        "created_at": db_user["created_at"]
    }

@app.post("/auth/login", response_model=Token)
def login(user_data: UserLogin):
    """Login user"""
    user = authenticate_user(user_data.username, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=60 * 24)  # 24 hours
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user["username"],
        "user_id": user["id"]
    }

@app.get("/auth/me", response_model=UserResponse)
def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    return {
        "id": current_user["id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "created_at": current_user["created_at"]
    }

# -------------------------------------------------
# PUBLIC ENDPOINTS (No auth required)
# -------------------------------------------------
@app.get("/")
def home():
    """Redirect to login page"""
    return RedirectResponse(url="/login.html")

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_err}")
    return {"status": "ok", "authenticated": False}

# Serve static HTML files
@app.get("/login.html", response_class=HTMLResponse)
def serve_login():
    if not os.path.exists("login.html"):
        return HTMLResponse("<h2>Login page not found</h2>", status_code=404)
    return FileResponse("login.html")

@app.get("/signup.html", response_class=HTMLResponse)
def serve_signup():
    if not os.path.exists("signup.html"):
        return HTMLResponse("<h2>Signup page not found</h2>", status_code=404)
    return FileResponse("signup.html")

@app.get("/dashboard.html", response_class=HTMLResponse)
def serve_dashboard():
    if not os.path.exists("dashboard.html"):
        return HTMLResponse("<h2>Dashboard not found</h2>", status_code=404)
    return FileResponse("dashboard.html")

# -------------------------------------------------
# PROTECTED ENDPOINTS (Require auth)
# -------------------------------------------------
@app.get("/api/health")
def protected_health(current_user: dict = Depends(get_current_user)):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_err}")
    return {
        "status": "ok", 
        "authenticated": True, 
        "username": current_user["username"],
        "user_id": current_user["id"]
    }

@app.post("/predict")
def predict_url(
    item: URLItem, 
    current_user: dict = Depends(get_current_user)
):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_err}")

    url = item.url.strip()
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()

    explanations: List[str] = []
    top_features: List[dict] = []

    # ----------------- BASIC URL VALIDITY / INCOMPLETE CHECKS -----------------
    host_no_port = host.split(":")[0]

    if host_no_port.startswith("www."):
        bare_host = host_no_port[4:]
    else:
        bare_host = host_no_port

    parts = bare_host.split(".") if bare_host else []
    left_part = parts[0] if parts else ""

    # INCOMPLETE URL
    if (not bare_host) or (len(parts) < 2) or (len(left_part) < 2):
        result = "Incomplete URL (domain appears incomplete)"
        explanations.append(
            "The domain looks incomplete. Please enter a full website address such as https://example.com."
        )
        label = 2
        conf = None

        _save_history(url, label, result, conf, explanations, current_user["id"])
        return _build_response(url, result, label, conf, explanations, top_features)

    # ----------------- STRONG PHISHING HEURISTIC -----------------
    if "--" in bare_host or "--" in parsed.path:
        result = "Phishing (suspicious double dash in URL)"
        explanations.append("URL contains consecutive dashes '--', a pattern often seen in phishing links.")
        label = 1
        conf = None
        _save_history(url, label, result, conf, explanations, current_user["id"])
        return _build_response(url, result, label, conf, explanations, top_features)

    # ----------------- HEURISTIC "SUSPICIOUS BUT NOT 100% BAD" -----------------
    COMMON_TLDS = (".com", ".org", ".net", ".edu", ".gov", ".my", ".co", ".io")
    suspicious = False

    if not bare_host.endswith(COMMON_TLDS):
        suspicious = True
        explanations.append(
            "Domain does not end with a common TLD (e.g., .com, .org, .net) – treat this URL with caution."
        )

    main_label = bare_host.split(".")[0]
    if "-" in main_label:
        suspicious = True
        explanations.append(
            "Main part of the domain contains '-' (e.g., 'google-admin'), "
            "which is sometimes used to imitate legitimate sites."
        )

    # ----------------- MACHINE LEARNING PREDICTION -----------------
    feats = extract_features(url)
    df = to_model_frame(feats)
    pred = int(model.predict(df)[0])
    conf = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        conf = float(proba[pred])

    # High-level textual prediction
    if pred == 1:
        result = "Phishing"
    elif suspicious:
        result = "Potential phishing (suspicious URL pattern, model prediction: Safe)"
    else:
        result = "Safe"

    # ----------------- FEATURE-BASED EXPLANATIONS -----------------
    try:
        if "NumSensitiveWords" in df.columns and df["NumSensitiveWords"].iloc[0] > 0:
            explanations.append(
                "Contains words like login/secure/account, which are common in phishing pages."
            )
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

    if suspicious and pred == 0:
        explanations.append(
            "Heuristic URL checks flagged this link as unusual even though the ML model rated it Safe. "
            "Treat it as suspicious and double-check before trusting it."
        )

    if not explanations:
        explanations.append("URL structure appears normal based on basic checks.")

    # ----------------- ML TOP FEATURE SNAPSHOT -----------------
    try:
        candidate_features = [
            "NoHttps",
            "NumSensitiveWords",
            "SubdomainLevel",
            "NumDash",
            "IpAddress",
            "UrlLength",
            "NumDots",
        ]

        for name in candidate_features:
            if name in df.columns:
                val = df[name].iloc[0]
                if hasattr(val, "item"):
                    val = val.item()
                top_features.append({"name": name, "value": val})
    except Exception as e:
        print(f"[WARN] Failed building top_features: {e}")

    # save in DB with user_id
    _save_history(url, pred, result, conf, explanations, current_user["id"])
    return _build_response(url, result, pred, conf, explanations, top_features)

@app.get("/history")
def get_history(
    limit: int = 50, 
    current_user: dict = Depends(get_current_user)
):
    """Get user's search history"""
    rows = get_user_search_history(current_user["id"], limit)
    
    history_list = []
    for r in rows:
        try:
            expl = json.loads(r["explanation"]) if r["explanation"] else []
        except Exception:
            expl = [str(r["explanation"])]

        history_list.append(
            {
                "id": r["id"],
                "url": r["url"],
                "label": r["label"],
                "prediction": r["prediction"],
                "confidence": r["confidence"],
                "explanation": expl,
                "created_at": r["created_at"],
                "user_id": r["user_id"]
            }
        )

    return {"history": history_list}

@app.delete("/history/clear")
def clear_history(current_user: dict = Depends(get_current_user)):
    """Clear user's search history"""
    # Note: We need to add this function to db.py
    # For now, we'll use a workaround
    from db import SQLiteDB
    
    try:
        db = SQLiteDB()
        db.execute_query(
            "DELETE FROM search_history WHERE user_id = ?", 
            (current_user["id"],)
        )
        db.close()
        print(f"[INFO] History cleared for user: {current_user['username']}")
        return {"status": "ok", "message": "history cleared"}
    except Exception as e:
        print(f"[ERROR] Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear history")

@app.post("/history/log")
def log_frontend_result(
    item: FrontendLogItem, 
    current_user: dict = Depends(get_current_user)
):
    """Log a frontend phishing result"""
    try:
        record_id = add_search_history(
            url=item.url,
            user_id=current_user["id"],
            prediction=item.prediction,
            confidence=1.0,
            explanation=json.dumps(item.explanation or [])
        )
        
        if record_id:
            print(f"[INFO] Logged frontend phishing for user {current_user['username']}: {item.url}")
            return {"status": "ok", "message": "saved", "record_id": record_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to save to database")
    except Exception as e:
        print(f"[ERROR] Failed to save frontend history: {e}")
        raise HTTPException(status_code=500, detail="Failed to save frontend history")

# -------------------------------------------------
# NEW: GUEST/ANONYMOUS PREDICTION ENDPOINT
# -------------------------------------------------
@app.post("/predict/guest")
def predict_url_guest(item: URLItem):
    """Predict URL without authentication (guest mode)"""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_err}")

    url = item.url.strip()
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()

    explanations: List[str] = []
    top_features: List[dict] = []

    # Basic checks (same as authenticated version)
    host_no_port = host.split(":")[0]
    if host_no_port.startswith("www."):
        bare_host = host_no_port[4:]
    else:
        bare_host = host_no_port

    parts = bare_host.split(".") if bare_host else []
    left_part = parts[0] if parts else ""

    # INCOMPLETE URL
    if (not bare_host) or (len(parts) < 2) or (len(left_part) < 2):
        result = "Incomplete URL (domain appears incomplete)"
        explanations.append(
            "The domain looks incomplete. Please enter a full website address such as https://example.com."
        )
        label = 2
        conf = None
        return _build_response(url, result, label, conf, explanations, top_features)

    # Phishing heuristic
    if "--" in bare_host or "--" in parsed.path:
        result = "Phishing (suspicious double dash in URL)"
        explanations.append("URL contains consecutive dashes '--', a pattern often seen in phishing links.")
        label = 1
        conf = None
        return _build_response(url, result, label, conf, explanations, top_features)

    # ML Prediction
    feats = extract_features(url)
    df = to_model_frame(feats)
    pred = int(model.predict(df)[0])
    conf = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        conf = float(proba[pred])

    # Result
    result = "Phishing" if pred == 1 else "Safe"
    
    # Save to database without user_id (guest mode)
    _save_history(url, pred, result, conf, explanations, user_id=None)
    
    return _build_response(url, result, pred, conf, explanations, top_features)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def _save_history(
    url: str,
    label: int,
    prediction: str,
    confidence,
    explanations: List[str],
    user_id: Optional[int] = None
):
    """Save search history to database"""
    try:
        record_id = add_search_history(
            url=url,
            user_id=user_id,
            prediction=prediction,
            confidence=confidence,
            explanation=json.dumps(explanations)
        )
        return record_id
    except Exception as e:
        print(f"[WARN] Failed to save history: {e}")
        return None

def _build_response(
    url: str,
    prediction: str,
    label: int,
    confidence,
    explanations: List[str],
    top_features: List[dict] | None = None,
):
    """Build API response"""
    return {
        "url": url,
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "explanation": explanations,
        "top_features": top_features or [],
    }

# -------------------------------------------------
# NEW: STATIC FILE SERVING FOR DASHBOARD
# -------------------------------------------------
@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files"""
    static_path = os.path.join(".", file_path)
    if os.path.exists(static_path):
        return FileResponse(static_path)
    raise HTTPException(status_code=404, detail="File not found")

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)