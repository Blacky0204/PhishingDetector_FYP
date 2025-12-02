# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, validator
from urllib.parse import urlparse
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
import joblib
import json

from feature_extraction import extract_features
from db import get_db, SearchHistory

# -------------------------------------------------
# FastAPI APP
# -------------------------------------------------
app = FastAPI(title="Phishing Detection API", version="1.0")

# CORS (frontend will call this API from browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
model = None
MODEL_COLUMNS = None
load_err = ""

try:
    pkg = joblib.load("phishing_model.pkl")
    model = pkg["model"]
    MODEL_COLUMNS = list(pkg["columns"])
    print(f"[INFO] Loaded model with {len(MODEL_COLUMNS)} columns.")
except Exception as e:
    load_err = str(e)
    print(f"[ERROR] {load_err}")


def to_model_frame(features_dict: dict) -> pd.DataFrame:
    """
    Align features from extract_features() with the columns
    that the model was trained on.
    Any extra features are dropped; any missing ones are filled with 0.
    """
    if MODEL_COLUMNS is None:
        raise RuntimeError("Model columns not loaded.")

    df = pd.DataFrame([features_dict])
    for c in MODEL_COLUMNS:
        if c not in df.columns:
            df[c] = 0
    return df[MODEL_COLUMNS]


# -------------------------------------------------
# Pydantic models
# -------------------------------------------------
class URLItem(BaseModel):
    url: str

    @validator("url")
    def ensure_scheme(cls, v: str) -> str:
        """
        Ensure URL has http:// or https:// so urlparse() works properly.
        The frontend already tries to enforce this, but we double-check here.
        """
        v = v.strip()
        if not urlparse(v).scheme:
            v = "http://" + v
        return v


class FrontendLogItem(BaseModel):
    url: str
    prediction: str
    explanation: List[str] = []


# -------------------------------------------------
# Basic endpoints
# -------------------------------------------------
@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_err}")
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return "<h2>Phishing Detection API is running ✅</h2>"


# -------------------------------------------------
# PREDICT ENDPOINT (ML + heuristics, always logs to DB)
# -------------------------------------------------
@app.post("/predict")
def predict_url(item: URLItem, db: Session = Depends(get_db)):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_err}")

    url = item.url.strip()
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()

    explanations: List[str] = []
    top_features: List[dict] = []

    # ----------------- BASIC URL VALIDITY CHECKS -----------------
    # Strip port if any (e.g., example.com:8080)
    host_no_port = host.split(":")[0]

    # Remove leading "www."
    if host_no_port.startswith("www."):
        bare_host = host_no_port[4:]
    else:
        bare_host = host_no_port

    # Case 1: No host at all => clearly bad / incomplete
    if not bare_host:
        result = "Phishing (invalid or missing domain)"
        explanations.append("URL is missing a valid domain name (e.g., example.com).")
        label = 1
        conf = None
        _save_history(db, url, label, result, conf, explanations)
        return _build_response(url, result, label, conf, explanations, top_features)

    # Case 2: Domain without any dot (e.g., https://goo, https://localhost)
    if "." not in bare_host:
        result = "Phishing (incomplete domain name)"
        explanations.append("Domain does not contain a valid dot-separated name like example.com.")
        label = 1
        conf = None
        _save_history(db, url, label, result, conf, explanations)
        return _build_response(url, result, label, conf, explanations, top_features)

    # Case 3: Consecutive dashes anywhere in host or path -> very suspicious
    if "--" in bare_host or "--" in parsed.path:
        result = "Phishing (suspicious double dash in URL)"
        explanations.append("URL contains consecutive dashes '--', a pattern often seen in phishing links.")
        label = 1
        conf = None
        _save_history(db, url, label, result, conf, explanations)
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
    pred = int(model.predict(df)[0])      # 0 = safe, 1 = phishing (as in your dataset)
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
        # The same features you’re using on the frontend
        if df["NumSensitiveWords"].iloc[0] > 0:
            explanations.append(
                "Contains words like login/secure/account, which are common in phishing pages."
            )
        if df["NoHttps"].iloc[0] == 1:
            explanations.append("Not using HTTPS (connection not secure).")
        if df["SubdomainLevel"].iloc[0] >= 2:
            explanations.append("Has many subdomains, which can be used to mimic legitimate sites.")
        if df["NumDash"].iloc[0] >= 2:
            explanations.append("Contains several '-' characters, often used in spoofed domains.")
        if df["IpAddress"].iloc[0] == 1:
            explanations.append("Uses an IP address instead of a normal domain name, which is suspicious.")
    except Exception:
        # If any column name mismatch happens, just skip explanation section.
        pass

    if suspicious and pred == 0:
        explanations.append(
            "Heuristic URL checks flagged this link as unusual even though the ML model rated it Safe. "
            "Treat it as suspicious and double-check before trusting it."
        )

    if not explanations:
        explanations.append("URL structure appears normal based on basic checks.")

    # ----------------- ML TOP FEATURE SNAPSHOT (for Option C table) -----------------
    try:
        # These names match what your frontend expects in describeFeatureImpact()
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
                # cast to plain Python type (not numpy)
                if hasattr(val, "item"):
                    val = val.item()
                top_features.append({"name": name, "value": val})
    except Exception as e:
        print(f"[WARN] Failed building top_features: {e}")
        # in worst case, just leave top_features empty

    _save_history(db, url, pred, result, conf, explanations)
    return _build_response(url, result, pred, conf, explanations, top_features)


def _save_history(db: Session, url: str, label: int, prediction: str,
                  confidence, explanations: List[str]):
    """Helper to save one record into SearchHistory."""
    try:
        new_search = SearchHistory(
            url=url,
            label=label,
            prediction=prediction,
            confidence=confidence,
            explanation=json.dumps(explanations),
        )
        db.add(new_search)
        db.commit()
    except Exception as e:
        print(f"[WARN] Failed to save history: {e}")


def _build_response(
    url: str,
    prediction: str,
    label: int,
    confidence,
    explanations: List[str],
    top_features: List[dict] | None = None,
):
    """Helper to build API JSON response."""
    return {
        "url": url,
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
        "explanation": explanations,
        "top_features": top_features or [],
    }


# -------------------------------------------------
# VIEW HISTORY ENDPOINT  (frontend uses this to show table)
# -------------------------------------------------
@app.get("/history")
def get_history(limit: int = 50, db: Session = Depends(get_db)):
    """
    Return the last N search records as clean JSON.
    """
    rows = (
        db.query(SearchHistory)
        .order_by(SearchHistory.id.desc())
        .limit(limit)
        .all()
    )

    history_list = []
    for r in rows:
        try:
            expl = json.loads(r.explanation) if r.explanation else []
        except Exception:
            # fallback – keep raw string in an array
            expl = [str(r.explanation)]

        history_list.append(
            {
                "id": r.id,
                "url": r.url,
                "label": r.label,
                "prediction": r.prediction,
                "confidence": r.confidence,
                "explanation": expl,
                # kept for future expansion, even if UI doesn't show it
                "created_at": getattr(r, "created_at", None),
            }
        )

    return {"history": history_list}


# -------------------------------------------------
# CLEAR HISTORY ENDPOINT  (for "Clear History" button)
# -------------------------------------------------
@app.delete("/history/clear")
def clear_history(db: Session = Depends(get_db)):
    """
    Delete all records from SearchHistory.
    Used by frontend 'Clear History' button.
    """
    try:
        db.query(SearchHistory).delete()
        db.commit()
        print("[INFO] History table cleared.")
        return {"status": "ok", "message": "history cleared"}
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear history")


# -------------------------------------------------
# LOG FRONTEND-DECIDED RESULTS (RULE-BASED)
# -------------------------------------------------
@app.post("/history/log")
def log_frontend_result(item: FrontendLogItem, db: Session = Depends(get_db)):
    """
    Save results that were classified as phishing by frontend rules
    (dash+brand, incomplete domain, invalid URL, etc.).
    For this project, anything that uses this endpoint is treated as phishing (label=1).
    """
    label = 1  # anything using this endpoint is phishing

    try:
        new_row = SearchHistory(
            url=item.url,
            label=label,
            prediction=item.prediction,          # e.g. "Phishing (frontend dash+brand rule)"
            confidence=1.0,                      # treat frontend rule as 100% confident (avoids NULL)
            explanation=json.dumps(item.explanation or []),
        )

        db.add(new_row)
        db.commit()

        print(f"[INFO] Logged frontend phishing: {item.url} -> {item.prediction}")
        return {"status": "ok", "message": "saved"}
    except Exception as e:
        print(f"[ERROR] Failed to save frontend history: {e}")
        raise HTTPException(status_code=500, detail="Failed to save frontend history")