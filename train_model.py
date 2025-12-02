# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

CSV = "dataset.csv"
LABEL = "CLASS_LABEL"

FEATURE_COLUMNS = [
    "UrlLength", "NumDots", "NumDash", "AtSymbol", "TildeSymbol", "NumUnderscore", "NumPercent",
    "NumQueryComponents", "NumAmpersand", "NumHash", "NumNumericChars", "NoHttps", "IpAddress",
    "HostnameLength", "PathLength", "QueryLength", "DoubleSlashInPath", "SubdomainLevel",
    "PathLevel", "HttpsInHostname", "DomainInPaths", "NumSensitiveWords",
]

df = pd.read_csv(CSV)

missing = [c for c in FEATURE_COLUMNS + [LABEL] if c not in df.columns]
if missing:
    raise ValueError(f"dataset.csv is missing columns: {missing}")

X = df[FEATURE_COLUMNS]
y = df[LABEL]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
)

scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean CV score:", scores.mean())

model.fit(X, y)

y_pred = model.predict(X)

print("\nClassification Report:")
print(classification_report(y, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

joblib.dump({"model": model, "columns": FEATURE_COLUMNS}, "phishing_model.pkl")
print(f"Model saved as phishing_model.pkl with {len(FEATURE_COLUMNS)} columns.")