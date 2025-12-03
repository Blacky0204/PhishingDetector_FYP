# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, validator
from urllib.parse import urlparse
from sqlalchemy.orm import Session
from typing import List
import pandas as pd
import joblib
import json
import os
import uvicorn

from feature_extraction import extract_features
from db import get_db, SearchHistory

# -------------------------------------------------
# FastAPI APP
# -------------------------------------------------
app = FastAPI(title="Phishing Detection API", version="1.0")

# CORS (frontend will call this API from browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later you can restrict if needed
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
    pkg = joblib.load("phishing_model.pkl")  # make sure this file name matches your .pkl
    model = pkg["model"]
    MODEL_COLUMNS = list(pkg["columns"])
    print(f"[INFO] Loaded model with {len(MODEL_COLUMNS)} columns.")
except Exception as e:
    load_err = str(e)
    print(f"[ERROR] Failed to load model: {load_err}")


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
        Frontend already enforces, but we double-check here.
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
# BASIC ENDPOINTS
# -------------------------------------------------
@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_err}")
    return {"status": "ok"}


# 👉 Serve your frontend here
@app.get("/", response_class=HTMLResponse)
def home():
    """
    Serve the main HTML UI.
    Make sure index.html is in the same folder as main.py.
    """
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            content="<h2>index.html not found next to main.py</h2>",
            status_code=500,
        )
    return FileResponse(index_path)


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

    # ----------------- BASIC URL VALIDITY / INCOMPLETE CHECKS -----------------
    # Strip port if any (e.g., example.com:8080)
    host_no_port = host.split(":")[0]

    # Remove leading "www."
    if host_no_port.startswith("www."):
        bare_host = host_no_port[4:]
    else:
        bare_host = host_no_port

    # Split domain parts and main label (to match frontend logic)
    parts = bare_host.split(".") if bare_host else []
    left_part = parts[0] if parts else ""

    # INCOMPLETE URL (same idea as in index.html)
    # - no host at all
    # - less than 2 dot-separated parts (e.g., "google", "localhost")
    # - very short left part (1 char)
    if (not bare_host) or (len(parts) < 2) or (len(left_part) < 2):
        result = "Incomplete URL (domain appears incomplete)"
        explanations.append(
            "The domain looks incomplete. Please enter a full website address such as https://example.com."
        )
        label = 2  # 0 = safe, 1 = phishing, 2 = incomplete/info
        conf = None

        _save_history(db, url, label, result, conf, explanations)
        return _build_response(url, result, label, conf, explanations, top_features)

    # ----------------- STRONG PHISHING HEURISTIC -----------------
    # Consecutive dashes anywhere in host or path -> very suspicious
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
    pred = int(model.predict(df)[0])      # 0 = safe, 1 = phishing
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
        # These names align with your frontend's describeFeatureImpact()
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

    # save in DB with the *model* label (pred)
    _save_history(db, url, pred, result, conf, explanations)
    return _build_response(url, result, pred, conf, explanations, top_features)


# -------------------------------------------------
# HELPERS FOR HISTORY + RESPONSE
# -------------------------------------------------
def _save_history(
    db: Session,
    url: str,
    label: int,
    prediction: str,
    confidence,
    explanations: List[str],
):
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
            expl = [str(r.explanation)]

        history_list.append(
            {
                "id": r.id,
                "url": r.url,
                "label": r.label,
                "prediction": r.prediction,
                "confidence": r.confidence,
                "explanation": expl,
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
    (dash+brand, fake TLD, long URL, etc.).
    For this project, anything that uses this endpoint is treated as phishing (label=1).
    """
    label = 1  # anything using this endpoint is phishing

    try:
        new_row = SearchHistory(
            url=item.url,
            label=label,
            prediction=item.prediction,          # e.g. "Phishing (frontend dash+brand rule)"
            confidence=1.0,                      # treat frontend rule as 100% confident
            explanation=json.dumps(item.explanation or []),
        )
        db.add(new_row)
        db.commit()

        print(f"[INFO] Logged frontend phishing: {item.url} -> {item.prediction}")
        return {"status": "ok", "message": "saved"}
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to save frontend history: {e}")
        raise HTTPException(status_code=500, detail="Failed to save frontend history")


# -------------------------------------------------
# RUN (for local and Railway)
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)