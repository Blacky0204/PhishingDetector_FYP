from urllib.parse import urlparse

# Sensitive words that are commonly found in phishing websites
SENSITIVE_WORDS = [
    "login", "secure", "bank", "account", "verify", "update", "password",
    "signin", "pay", "paypal", "wallet", "credential", "webscr", "confirm",
]

def _ensure_scheme(u: str) -> str:
    """Ensure the URL has a scheme (http:// or https://)"""
    return u if "://" in u else "http://" + u

def _is_ipv4(host: str) -> int:
    """Check if the host is an IP address (IPv4)."""
    parts = host.split(".")
    if len(parts) != 4: return 0
    try:
        return 1 if all(p.isdigit() and 0 <= int(p) <= 255 for p in parts) else 0
    except:
        return 0

def _count_digits(s: str) -> int:
    """Count the number of digits in a string."""
    return sum(ch.isdigit() for ch in s)

def extract_features(url: str) -> dict:
    """Extract features from the given URL."""
    u = _ensure_scheme(url.strip())  # Ensure the URL has a scheme (http:// or https://)
    p = urlparse(u)
    host = (p.netloc or "").lower()
    path = p.path or "/"
    query = p.query or ""
    full = (u or "").lower()

    host_labels = host.split(".")
    subdomain_level = max(len(host_labels) - 2, 0)
    path_level = max(path.count("/") - 1, 0)
    double_slash_in_path = 1 if "//" in path else 0

    # Count the number of sensitive words in the URL
    num_sensitive = sum(1 for w in SENSITIVE_WORDS if w in full)

    # New feature: Check for consecutive dashes
    num_consecutive_dashes = 1 if "--" in full else 0

    # New feature: Check if ".com" is missing in the URL
    missing_dot_com = 1 if ".com" not in full else 0

    # New feature: Check if the URL is incomplete (missing domain, e.g., 'https://goo')
    incomplete_url = 1 if not p.netloc or len(p.netloc.split('.')) < 2 else 0

    feats = {
        "UrlLength": len(u),
        "NumDots": full.count("."),
        "NumDash": full.count("-"),
        "NumConsecutiveDashes": num_consecutive_dashes,  # Add this feature
        "AtSymbol": 1 if "@" in full else 0,
        "TildeSymbol": 1 if "~" in full else 0,
        "NumUnderscore": full.count("_"),
        "NumPercent": full.count("%"),
        "NumQueryComponents": (query.count("&") + 1) if query else 0,
        "NumAmpersand": full.count("&"),
        "NumHash": full.count("#"),
        "NumNumericChars": _count_digits(full),
        "NoHttps": 0 if full.startswith("https://") else 1,
        "IpAddress": _is_ipv4(host),
        "HostnameLength": len(host),
        "PathLength": len(path),
        "QueryLength": len(query),
        "DoubleSlashInPath": double_slash_in_path,
        "SubdomainLevel": subdomain_level,
        "PathLevel": path_level,
        "HttpsInHostname": 1 if "https" in host else 0,
        "DomainInPaths": 1 if host.replace("www.", "") in path.lower() else 0,
        "NumSensitiveWords": num_sensitive,
        "MissingDotCom": missing_dot_com,  # Add this feature
        "IncompleteUrl": incomplete_url,  # Add this feature
    }

    return feats
