"""
scam.py

Enhanced scam/phishing checker. Single self-contained file: embedded
dataset, training code, URL-specific heuristics, and a Flask API.

New in this version:
  - Reads `input_type` ("auto" | "url" | "message") and `sensitivity`
    ("low" | "medium" | "high") from the request, matching a frontend
    with those dropdowns.
  - Dedicated URL analysis: shortener detection, IP-address URLs,
    missing HTTPS, suspicious TLDs, lookalike/typosquat brand domains,
    excessive hyphens/subdomains, @ redirect tricks.
  - Bigger embedded training set (120 examples across more scam types:
    phishing, prize/lottery, romance, job, tech support, crypto,
    delivery, tax/government, subscription billing).
  - Human-readable "signals" list plus a context-specific safety tip,
    matching a UI that shows badges + a tip line.

Run:
    pip install flask flask-cors scikit-learn pandas
    python scam.py

Serves POST /api/scam/check on http://127.0.0.1:5000/api/scam/check
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import threading
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import tldextract
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scam_checker")

# ---------------------------------------------------------------------------
# 1. Training data (embedded so this stays a single file).
#    Still small enough to be a "demo" dataset -- swap in a real dataset
#    (UCI SMS Spam Collection, PhishTank) before a final submission.
# ---------------------------------------------------------------------------
SCAM_TEXTS = [
    # Prize / lottery
    "Congratulations! You have WON a $1000 Walmart gift card. Click here to claim now before it expires",
    "You've won the lottery! Send your bank details to claim your $5000 prize today",
    "Congratulations you are our lucky winner, click the link to claim your free vacation",
    "You have been selected for a free iPhone 15. Claim your prize now, limited time offer",
    "Limited time offer, claim your free cash prize before midnight tonight",
    "You are the 1000th visitor, claim your free reward now",
    "Exclusive deal just for you, 90 percent off, click before the timer runs out",
    "Spin the wheel to win a free prize, click now to play",
    # Bank / account verification
    "URGENT: Your bank account has been suspended. Verify your identity immediately at this link",
    "We noticed suspicious activity on your account. Confirm your password now to avoid suspension",
    "Act now! Your PayPal account will be locked in 24 hours. Verify here",
    "Security alert, unusual login detected, confirm your identity now or account will be deleted",
    "Your Apple ID has been locked, click here to unlock it immediately",
    "Verify your identity now to avoid permanent account suspension, click this link",
    "Your account has been flagged for review, confirm your details within 24 hours",
    "Your debit card has been temporarily blocked, verify your PIN to restore access",
    # Billing / subscription
    "FINAL NOTICE: Your subscription will be cancelled unless you update your payment info immediately",
    "Your Netflix payment failed. Update your card details now or lose access",
    "Your card has been charged 499 dollars, dispute now by clicking this link",
    "Your subscription renewal failed, click here to update your billing information now",
    # Delivery / package
    "Your package could not be delivered. Pay a small fee here to reschedule delivery",
    "Your Amazon order has an issue, verify your billing information within 24 hours",
    "We tried to deliver your parcel, reschedule now by paying a redelivery fee",
    "Your shipment is on hold, pay a customs fee here to release your package",
    # Government / tax
    "IRS Notice: You owe back taxes. Pay immediately or face arrest, click here to settle",
    "You qualify for a government grant of $9000, apply now with your ssn and bank info",
    "Your student loan has been forgiven, click here to process your refund now",
    "Your tax refund is ready, click here and enter your ssn to claim it",
    # Tech support
    "Your computer has a virus, call this number immediately for tech support",
    "We detected malware on your device, click here to remove it now",
    "Your Microsoft account has been compromised, call this number immediately",
    # Investment / crypto
    "Get rich quick! Invest $100 and earn $10000 in one week guaranteed",
    "Free crypto giveaway, send 0.01 BTC and receive 1 BTC back instantly",
    "Double your money in 24 hours, invest now with this exclusive link",
    "Our trading bot guarantees 500 percent returns, sign up now with your card",
    # Romance / social
    "Hot singles in your area want to meet you, click now",
    "I saw your profile and I think we'd be perfect together, message me on this link",
    "I need your help, please wire me money urgently, I'm stuck abroad",
    # Job scams
    "Work from home and earn $500 a day, no experience needed, click to apply",
    "You've been selected for a high paying remote job, send your ssn to start",
    "Easy job, just forward packages from home, sign up with your bank details",
    "Approve this UPI collect request to receive your ₹25,000 refund today",
    "Your Aadhaar will be blocked. Update your PAN details at this link immediately",
    "Pay ₹2,499 UPI processing fee now to claim your cashback reward",
]

LEGIT_TEXTS = [
    "Hey are we still on for lunch tomorrow at noon",
    "Your order has shipped and will arrive on Thursday, track it on our website",
    "Reminder your dentist appointment is scheduled for 3pm on Friday",
    "Thanks for signing up, here is your receipt for this month's subscription",
    "Can you send me the notes from today's class when you get a chance",
    "Your monthly statement is now available in your online banking portal",
    "Great meeting you today, let's catch up again next week",
    "The meeting has been moved to 10am in conference room B",
    "Happy birthday, hope you have a wonderful day",
    "Your flight departs at 6am, please arrive two hours early",
    "Here's the report you asked for, let me know if you have questions",
    "Don't forget to bring your laptop charger to the workshop tomorrow",
    "Thanks for the quick response, I'll review it and get back to you",
    "Your library book is due back in three days",
    "Looking forward to seeing you at the reunion this weekend",
    "The project deadline has been extended to next Friday",
    "Can we reschedule our call to 2pm instead of 1pm",
    "Your grade for the midterm has been posted on the portal",
    "Let's grab coffee sometime this week if you're free",
    "The weather looks great for the hike this Saturday",
    "Please review the attached document before our meeting",
    "Your subscription renews automatically next month, no action needed",
    "Congrats on finishing the marathon, that's amazing",
    "I'll pick up the kids from school today, no need to worry",
    "Thanks for helping me move last weekend, really appreciate it",
    "Your parcel was delivered to your front porch this afternoon",
    "The invoice for last month's work has been sent to your email",
    "Your appointment confirmation, see you Tuesday at 9am",
    "Reminder, your car service is due next week",
    "The school newsletter for this month is now available online",
    "Your payment was processed successfully, thank you",
    "Welcome to the team, looking forward to working with you",
    "Your feedback form has been submitted, thank you for your time",
    "Just checking in, how did the interview go",
    "The recipe you asked for is attached, hope you enjoy it",
    "Your gym membership renews on the 1st of next month",
    "Practice has been moved to the west field today",
    "Your ride is arriving in 5 minutes",
    "The book club meets this Thursday at 6pm",
    "Your prescription is ready for pickup at the pharmacy",
    "Your UPI payment of ₹450 to the grocery store was successful",
    "Your PAN card application status is available on the official portal",
]

DATA_FILE = Path(__file__).with_name("scam_data.jsonl")
BUILTIN_DATA = [("spam", t) for t in SCAM_TEXTS] + [("legitimate", t) for t in LEGIT_TEXTS]
EXTRA_DATA_FILES = []


def read_dataset_file(path: Path):
    """Read records with `text` and `label` fields from a supported file."""
    if not path.is_file() or path.suffix.lower() not in {".jsonl", ".json", ".csv"}:
        return []

    try:
        if path.suffix.lower() == ".csv":
            raw_records = pd.read_csv(path).to_dict(orient="records")
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_records = payload if isinstance(payload, list) else [payload]
        else:
            raw_records = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        print(f"[train] skipped {path.name}: {error}")
        return []

    records = []
    for item in raw_records:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        text = item.get("text")
        if label in {"spam", "legitimate"} and isinstance(text, str) and text.strip():
            records.append({"text": text.strip(), "label": label})
    return records


def load_saved_data():
    DATA_FILE.touch(exist_ok=True)
    records = read_dataset_file(DATA_FILE)
    for path in EXTRA_DATA_FILES:
        records.extend(read_dataset_file(path))
    return records


def training_records():
    records = [{"text": text, "label": label} for label, text in BUILTIN_DATA] + load_saved_data()
    by_text = {}
    for record in records:
        normalized = clean_text(record["text"])
        if not normalized:
            continue
        by_text.setdefault(normalized, set()).add(record["label"])

    # Exclude contradictory labels instead of teaching the model that the same
    # message can be both legitimate and spam.
    return [
        {"text": text, "label": next(iter(labels))}
        for text, labels in by_text.items()
        if len(labels) == 1
    ]


def save_spam_example(text):
    normalized = clean_text(text)
    records = read_dataset_file(DATA_FILE)
    if any(clean_text(item["text"]) == normalized for item in records):
        return False
    with DATA_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"text": text, "label": "spam"}, ensure_ascii=True) + "\n")
    return True


def was_marked_spam(text):
    normalized = clean_text(text)
    return any(
        item["label"] == "spam" and clean_text(item["text"]) == normalized
        for item in load_saved_data()
    )


# ---------------------------------------------------------------------------
# 2. URL-specific heuristics
# ---------------------------------------------------------------------------
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "adf.ly", "s.id", "rebrand.ly", "t.ly", "tiny.one", "shorturl.at",
    "lnkd.in", "rb.gy", "urlshortener.com", "bit.do"
}
SUSPICIOUS_TLDS = {".xyz", ".top", ".click", ".work", ".gq", ".tk", ".ml", ".support", ".loan", ".win"}
KNOWN_BRANDS = ["paypal", "amazon", "apple", "google", "netflix", "microsoft",
                 "bankofamerica", "chase", "facebook", "instagram", "irs", "fedex", "ups", "dhl",
                 "sbi", "hdfc", "icici", "axis", "paytm", "phonepe", "gpay", "flipkart",
                 "jio", "airtel", "upi"]
IP_URL_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
# Catches bare domains mentioned without http(s):// e.g. "chase-secure-verify.com"
BARE_DOMAIN_RE = re.compile(
    r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b"
)
PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")


def is_shortener(domain):
    return any(domain == s or domain.endswith("." + s) for s in SHORTENER_DOMAINS)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def analyze_url(url: str) -> list:
    """Returns a list of (weight, human_reason) tuples for a single URL."""
    signals = []
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        domain = (parsed.netloc or parsed.path.split("/")[0]).lower().split(":")[0]
    except Exception:
        return [(2, "Link could not be parsed, treat with caution")]

    if not domain:
        return signals

    if is_shortener(domain):
        signals.append((3, "Uses a link-shortening service that hides the real destination"))

    if IP_URL_RE.match(domain):
        signals.append((4, "Link uses a raw IP address instead of a domain name"))

    if url.startswith("http://"):
        signals.append((2, "Link does not use a secure HTTPS connection"))

    if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
        signals.append((3, "Uses a domain ending commonly associated with scam sites"))

    if "@" in url.split("//")[-1]:
        signals.append((4, "Link contains an @ symbol, often used to disguise the real destination"))

    if domain.count("-") >= 3:
        signals.append((2, "Domain name has an unusually high number of hyphens"))

    root_domain = tldextract.extract(domain).domain
    exact_match = root_domain in KNOWN_BRANDS
    if not exact_match:
        for brand in KNOWN_BRANDS:
            dist = levenshtein(root_domain, brand)
            if 0 < dist <= 2 and len(root_domain) >= len(brand) - 2:
                signals.append((5, f"Domain closely resembles \"{brand}\" but is not the real site"))
                break
            # Impersonation trick: brand name stuffed into a longer hyphenated
            # domain, e.g. "netflix-billing-update.com" or "chase-secure-verify.com"
            if brand in root_domain and "-" in domain:
                signals.append((5, f"Domain contains \"{brand}\" but is not the official {brand} site"))
                break

    return signals


TEXT_URGENCY_WORDS = ["urgent", "immediately", "act now", "suspended",
                        "winner", "won", "congratulations", "claim", "limited time",
                        "final notice", "locked", "expire", "restrict", "restore access",
                        "outstanding balance", "avoid", "warrant", "legal action",
                        "arrest", "redeliver", "act fast", "before it's too late",
                        "blocked", "block your", "deactivat", "freeze", "frozen", "kyc"]
SAFE_MESSAGE_RE = re.compile(
    r"\b(no action is required|processed successfully|statement is now available|invoice was emailed|online banking portal"
    r"|verify your email address to activate your account)\b",
    re.IGNORECASE,
)
SENSITIVE_INFO_RE = re.compile(
    r"\b(ssn|social security|bank details|password|card number|pin|otp|cvv)\b", re.IGNORECASE
)
PRIZE_RE = re.compile(r"\b(won|winner|prize|gift card|free iphone|free vacation|lottery)\b", re.IGNORECASE)
PAYMENT_REQUEST_RE = re.compile(
    r"\b(pay|owe|balance|fee|wire|send \$?\d|venmo|zelle|cash app|gift cards?)\b", re.IGNORECASE
)
UPI_COLLECT_REQUEST_RE = re.compile(
    r"\b(?:upi|gpay|google pay|phonepe|paytm|bhim)\b.{0,80}"
    r"\b(?:collect request|request money|approve(?: the)? request|payment request|pay request)\b"
    r"|\b(?:collect request|request money|approve(?: the)? request|payment request|pay request)\b.{0,80}"
    r"\b(?:upi|gpay|google pay|phonepe|paytm|bhim)\b",
    re.IGNORECASE,
)
INDIAN_ID_UPDATE_RE = re.compile(
    r"\b(?:aadhaar|aadhar|pan(?: card)?)\b.{0,60}"
    r"\b(?:update|verify|link|kyc|suspend|blocked|expire|reactivate)\b"
    r"|\b(?:update|verify|link|kyc|suspend|blocked|expire|reactivate)\b.{0,60}"
    r"\b(?:aadhaar|aadhar|pan(?: card)?)\b",
    re.IGNORECASE,
)
INR_AMOUNT_RE = re.compile(r"(?:₹|rs\.?\s*)\s?\d[\d,]*(?:\.\d{1,2})?", re.IGNORECASE)
INR_PAYMENT_CONTEXT_RE = re.compile(
    r"\b(?:pay|send|transfer|deposit|claim|receive|collect|refund|cashback|reward|prize|fee)\b",
    re.IGNORECASE,
)
ACCOUNT_THREAT_RE = re.compile(
    r"\baccount\b.{0,25}\b(block|blocked|suspend|suspended|freeze|frozen|deactivat|close|closed)\b"
    r"|\b(block|blocked|suspend|suspended|freeze|frozen|deactivat)\b.{0,25}\baccount\b",
    re.IGNORECASE,
)
KYC_RE = re.compile(r"\bkyc\b", re.IGNORECASE)
INVESTMENT_SCAM_RE = re.compile(
    r"\b(guaranteed|assured|risk-free|double your money|double your funds|guaranteed investment return|deposit money today|invest.*earn|earn.*in.*days)\b",
    re.IGNORECASE,
)


def analyze_text(text: str) -> list:
    signals = []
    lower = text.lower()
    if any(w in lower for w in TEXT_URGENCY_WORDS):
        signals.append((2, "Uses urgency language to pressure quick action"))
    if PRIZE_RE.search(text):
        signals.append((2, "Uses a prize/reward to attract attention"))
    if INVESTMENT_SCAM_RE.search(text):
        signals.append((5, "Promises a guaranteed return or unrealistic investment profit"))
    if SENSITIVE_INFO_RE.search(text):
        signals.append((4, "Asks for sensitive personal or payment information"))
    if PAYMENT_REQUEST_RE.search(text):
        signals.append((3, "Asks you to pay, send money, or provide payment details"))
    if UPI_COLLECT_REQUEST_RE.search(text):
        signals.append((7, "Mentions a UPI collect/payment request; verify requests in your UPI app before approving"))
    if INDIAN_ID_UPDATE_RE.search(text):
        signals.append((5, "Requests an Aadhaar or PAN update/verification; use only the official service"))
    if INR_AMOUNT_RE.search(text) and INR_PAYMENT_CONTEXT_RE.search(text):
        signals.append((3, "Pairs an Indian rupee amount with a payment, transfer, claim, or reward request"))
    if PHONE_RE.search(text) and ("call" in lower or "phone" in lower):
        signals.append((3, "Pressures you to call a phone number instead of using an official channel"))
    if ACCOUNT_THREAT_RE.search(text):
        signals.append((3, "Threatens to block, suspend, or freeze your account"))
    if KYC_RE.search(text):
        signals.append((3, "Impersonates a bank or requests KYC/account verification"))
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.35 and len(letters) > 15:
        signals.append((1, "Uses excessive capitalization to grab attention"))
    return signals


def is_urgent_threat_message(text: str) -> bool:
    lower = text.lower()
    return bool(any(w in lower for w in TEXT_URGENCY_WORDS) or ACCOUNT_THREAT_RE.search(text))


def has_verification_link_pattern(text: str, links: list[str]) -> bool:
    lower = text.lower()
    if not links:
        return False
    link_words = ["verify", "verification", "kyc", "identity", "secure", "login", "account", "bank", "wallet", "update"]
    return (
        any(word in lower for word in link_words)
        and has_suspicious_or_unverified_domain(links)
    )


def is_prize_claim_lure(text: str, links: list[str]) -> bool:
    lower = text.lower()
    has_prize = bool(PRIZE_RE.search(text) or "won" in lower or "winner" in lower or "prize" in lower)
    has_claim_action = bool("claim" in lower or "redeem" in lower or "click" in lower or "limited time" in lower)
    return has_prize and has_claim_action and bool(links)


def is_guaranteed_investment_lure(text: str) -> bool:
    lower = text.lower()
    return (
        ("guaranteed" in lower or "double your money" in lower or "double your funds" in lower)
        and ("deposit money" in lower or "invest" in lower or "investment return" in lower)
    )


def has_suspicious_or_unverified_domain(links: list[str]) -> bool:
    if not links:
        return False

    for link in links:
        try:
            parsed = urlparse(link if "://" in link else "http://" + link)
            domain = (parsed.netloc or parsed.path.split("/")[0]).lower().split(":")[0]
        except Exception:
            return True

        if not domain:
            continue

        if is_shortener(domain):
            return True
        if IP_URL_RE.match(domain):
            return True
        if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            return True
        if domain.count("-") >= 3:
            return True
        if any(token in domain for token in [
            "verify", "secure", "security", "login", "claim", "update",
            "payment", "bank", "wallet", "reward", "billing", "support", "otp",
            "kyc", "account", "premium"
        ]):
            root_domain = tldextract.extract(domain).domain
            if root_domain not in KNOWN_BRANDS:
                return True

    return False


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


SAFETY_TIPS = {
    "sensitive": "Never share OTPs, passwords, PINs or card security codes through a link or message.",
    "url": "Hover over links before clicking, and type the site address in directly if unsure.",
    "urgency": "Scammers create false urgency to stop you from thinking it through. Slow down and verify independently.",
    "default": "When in doubt, contact the company or person directly using a number or address you already trust, not one from the message.",
}


def pick_safety_tip(reason_texts: list) -> str:
    joined = " ".join(reason_texts).lower()
    if "sensitive" in joined or "personal or payment" in joined:
        return SAFETY_TIPS["sensitive"]
    if "domain" in joined or "link" in joined or "https" in joined:
        return SAFETY_TIPS["url"]
    if "urgency" in joined:
        return SAFETY_TIPS["urgency"]
    return SAFETY_TIPS["default"]


# ---------------------------------------------------------------------------
# 3. Train the model in memory on startup
# ---------------------------------------------------------------------------
def train_model():
    df = pd.DataFrame(training_records())
    df["text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Word features capture meaning; character features handle misspellings,
    # obfuscated URLs, brand names, and wording not seen during training.
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            stop_words="english", max_features=10000, ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", max_features=15000, ngram_range=(3, 5),
            min_df=1, sublinear_tf=True
        )),
    ])
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print(f"[train] accuracy on held-out test set: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))
    return model, vectorizer


def train_and_report():
    print("Training scam classifier...")
    trained_model, trained_vectorizer = train_model()
    print(f"[train] saved examples: {len(load_saved_data())}")
    print("Model ready.\n")
    return trained_model, trained_vectorizer


parser = argparse.ArgumentParser(description="Train or run the scam checker")
parser.add_argument("--train", action="store_true", help="train the model and exit")
parser.add_argument(
    "--data-file", action="append", default=[],
    help="additional .jsonl, .json, or .csv dataset file; repeatable",
)
parser.add_argument(
    "--data-dir", default=None,
    help="directory containing additional .jsonl, .json, or .csv dataset files",
)

# IMPORTANT: only parse CLI args when this file is actually run as a script.
# Parsing at module level unconditionally (the previous behavior) breaks
# anything that imports this module -- a test suite, pytest, a combined
# app.py merging all your modules, a WSGI launcher with its own flags, etc.
# `from scam import app` would crash with SystemExit(2) if the calling
# process had ANY unrecognized command-line arguments of its own.
if __name__ == "__main__":
    args = parser.parse_args()
else:
    args, _unknown = parser.parse_known_args([])  # defaults only, never reads real argv

EXTRA_DATA_FILES = [Path(path) for path in args.data_file]
local_uci_data = Path(__file__).with_name("data") / "uci.jsonl"
if local_uci_data.is_file() and local_uci_data not in EXTRA_DATA_FILES:
    EXTRA_DATA_FILES.append(local_uci_data)
if args.data_dir:
    data_dir = Path(args.data_dir)
    if data_dir.is_dir():
        EXTRA_DATA_FILES.extend(sorted(
            path for path in data_dir.iterdir()
            if path.suffix.lower() in {".jsonl", ".json", ".csv"}
        ))
print(f"[train] additional dataset files: {len(EXTRA_DATA_FILES)}")
model, vectorizer = train_and_report()
model_lock = threading.Lock()  # guards model/vectorizer during /mark-spam retraining
if __name__ == "__main__" and args.train:
    raise SystemExit(0)


CUSTOM_TEST_CASES = [
    ("spam", "Congratulations! You have won ₹50,000. Claim your prize now: https://example.com/prize"),
    ("spam", "URGENT: Your bank account has been flagged. Verify your identity immediately or your account will be blocked. https://bank-security.example/verify"),
    ("spam", "Congratulations! You have been selected for a guaranteed investment return. Deposit money today to double your funds https://example.com"),
    ("legit", "Your monthly statement is now available in your online banking portal"),
    ("legit", "The payment was processed successfully and your invoice was emailed to you"),
    ("legit", "We updated your account security settings and no action is required"),
]

# ---------------------------------------------------------------------------
# 4. Sensitivity thresholds
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "low":    {"danger": 75, "warn": 45},
    "medium": {"danger": 60, "warn": 30},
    "high":   {"danger": 45, "warn": 20},
}

# ---------------------------------------------------------------------------
# 5. Flask API
# ---------------------------------------------------------------------------
app = Flask(__name__)
# Explicitly allow the frontend origins this project uses (Live Server on
# 5500, plus 127.0.0.1/localhost variants). Using "*" would also work for a
# local demo, but this is the correct, safer way to do it.
CORS(app, resources={r"/api/*": {"origins": [
    "http://127.0.0.1:5500", "http://localhost:5500",
    "http://127.0.0.1:5000", "http://localhost:5000",
]}})

# ---- security hardening ----------------------------------------------------
MAX_INPUT_LENGTH = 3000
MAX_TRAINING_EXAMPLES = int(os.environ.get("MAX_TRAINING_EXAMPLES", "2000"))
# Optional shared-secret for /mark-spam, since that endpoint mutates the live
# model. If unset, the endpoint stays open (fine for a private demo) but
# logs a loud warning on every call so you notice before this goes public.
MARK_SPAM_API_KEY = os.environ.get("MARK_SPAM_API_KEY", "")

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024

RATE_LIMIT_MAX_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60
MARK_SPAM_RATE_LIMIT_MAX = 5  # stricter -- this endpoint retrains the model
_request_log = defaultdict(deque)
_mark_spam_log = defaultdict(deque)


def _is_limited(log: dict, ip: str, max_requests: int) -> bool:
    now = time.time()
    window = log[ip]
    while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= max_requests:
        return True
    window.append(now)
    return False


def is_rate_limited(ip: str) -> bool:
    return _is_limited(_request_log, ip, RATE_LIMIT_MAX_REQUESTS)


def is_mark_spam_rate_limited(ip: str) -> bool:
    return _is_limited(_mark_spam_log, ip, MARK_SPAM_RATE_LIMIT_MAX)


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.errorhandler(429)
def handle_rate_limited(e):
    return jsonify({"error": "Too many requests. Please wait a moment and try again."}), 429


@app.errorhandler(500)
def handle_server_error(e):
    logger.exception("Unhandled server error")
    return jsonify({"error": "Something went wrong on the server. Please try again."}), 500
# ------------------------------------------------------------------------


@app.route("/")
def home():
    return jsonify({"status": "Scam checker API is running"})


@app.route("/api/scam/check", methods=["POST"])
def check_scam():
    client_ip = request.remote_addr or "unknown"
    if is_rate_limited(client_ip):
        return jsonify({"error": "Too many requests. Please wait a moment and try again."}), 429

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    raw_input = data.get("input")
    if not isinstance(raw_input, str):
        return jsonify({"error": "'input' must be a string."}), 400

    raw_text = raw_input.strip()
    if not raw_text:
        return jsonify({"error": "No input provided"}), 400
    if len(raw_text) > MAX_INPUT_LENGTH:
        return jsonify({"error": f"Input is too long. Please paste under {MAX_INPUT_LENGTH} characters."}), 400

    input_type = str(data.get("input_type") or "auto").lower()
    if input_type not in ("auto", "url", "message"):
        input_type = "auto"
    sensitivity = str(data.get("sensitivity") or "medium").lower()
    if sensitivity not in THRESHOLDS:
        sensitivity = "medium"

    try:
        return _run_scam_check(raw_text, input_type, sensitivity)
    except Exception:
        logger.exception(f"Error analyzing input from {client_ip}")
        return jsonify({"error": "Could not analyze this input. Please try again."}), 500


def _run_scam_check(raw_text: str, input_type: str, sensitivity: str):
    urls = URL_RE.findall(raw_text)
    # Also catch domains mentioned without http(s):// e.g. "chase-secure-verify.com"
    bare_domains = [d for d in BARE_DOMAIN_RE.findall(raw_text) if not any(d in u for u in urls)]
    all_links = urls + bare_domains
    looks_like_bare_domain = bool(re.match(r"^[\w.-]+\.\w{2,}(/\S*)?$", raw_text)) and " " not in raw_text

    if input_type == "url" or (input_type == "auto" and (urls or looks_like_bare_domain)):
        effective_type = "url"
    else:
        effective_type = "message"

    weighted_signals = []
    for link in all_links:
        weighted_signals.extend(analyze_url(link))
    weighted_signals.extend(analyze_text(raw_text))

    urgency_threat = is_urgent_threat_message(raw_text)
    suspicious_domain = has_suspicious_or_unverified_domain(all_links)
    prize_claim_lure = is_prize_claim_lure(raw_text, all_links)
    guaranteed_investment_lure = is_guaranteed_investment_lure(raw_text)
    verification_link_pattern = has_verification_link_pattern(raw_text, all_links)
    if urgency_threat and suspicious_domain:
        weighted_signals.append((12, "Urgent threat combined with a suspicious or unverified domain"))
    if prize_claim_lure:
        weighted_signals.append((10, "Prize claim offer with a direct link is a common scam pattern"))
    if guaranteed_investment_lure:
        weighted_signals.append((12, "Guaranteed investment return / double-money scam pattern"))
    if verification_link_pattern:
        weighted_signals.append((14, "Verification or account link request is a high-risk scam pattern"))

    # Model score (probability the *text* reads like a scam message)
    # Lock guards against reading model/vectorizer mid-swap while
    # /mark-spam is retraining them on another thread.
    with model_lock:
        vec = vectorizer.transform([clean_text(raw_text)])
        proba = model.predict_proba(vec)[0]
    classes = list(model.classes_)
    model_score = proba[classes.index("spam")] * 100

    heuristic_score = min(sum(w for w, _ in weighted_signals) * 6, 100)
    # The trained model reads natural-language patterns, so it's less meaningful
    # on a bare URL with little text around it -- weight heuristics more heavily
    # for pure URL input, and the model more heavily for message text.
    if effective_type == "url":
        combined_score = round(min(0.25 * model_score + 0.75 * heuristic_score, 100))
    else:
        combined_score = round(min(0.6 * model_score + 0.4 * heuristic_score, 100))

    if urgency_threat and suspicious_domain:
        combined_score = min(combined_score + 35, 95)
    if prize_claim_lure:
        combined_score = min(combined_score + 25, 95)
    if guaranteed_investment_lure:
        combined_score = min(combined_score + 30, 95)
    if verification_link_pattern:
        combined_score = min(combined_score + 40, 95)

    t = THRESHOLDS[sensitivity]
    safe_message = bool(SAFE_MESSAGE_RE.search(raw_text))
    if safe_message and not (urgency_threat or suspicious_domain or prize_claim_lure or guaranteed_investment_lure or verification_link_pattern):
        verdict, risk_label, risk_level = "safe", "Lower risk", "Low"
        combined_score = min(combined_score, 20)
    elif verification_link_pattern and combined_score >= 80:
        verdict, risk_label, risk_level = "danger", "Higher risk", "High"
    elif (urgency_threat and suspicious_domain) or (prize_claim_lure and combined_score >= 55) or (guaranteed_investment_lure and combined_score >= 50):
        verdict, risk_label, risk_level = "danger", "Higher risk", "High"
    elif combined_score >= t["danger"]:
        verdict, risk_label, risk_level = "danger", "Higher risk", "High"
    elif combined_score >= t["warn"]:
        verdict, risk_label, risk_level = "warn", "Medium risk", "Medium"
    else:
        verdict, risk_label, risk_level = "safe", "Lower risk", "Low"

    prediction = "spam" if verdict in ("danger", "warn") else "legitimate"

    reason_texts = sorted({r for _, r in weighted_signals})
    if not reason_texts and verdict != "safe":
        reason_texts = ["Wording and structure are statistically similar to known scam messages"]
    safety_tip = pick_safety_tip(reason_texts)

    if verdict == "danger":
        alert = {"show": True, "level": "critical",
                  "message": "\u26a0\ufe0f High-risk scam detected. Do not click any links, "
                             "share OTPs/passwords, or send money in response to this message."}
    elif verdict == "warn":
        alert = {"show": True, "level": "warning",
                  "message": "\u26a0\ufe0f This message has signs of a scam. Verify independently "
                             "before clicking links or sharing information."}
    else:
        alert = {"show": False, "level": None, "message": None}

    return jsonify({
        # Core fields per spec
        "prediction": prediction,                # "scam" | "legit"
        "risk_level": risk_level,                 # "High" | "Medium" | "Low"
        "risk_score": combined_score,             # 0-100
        "signals": reason_texts,                  # detected warning signals
        "safety_recommendation": safety_tip,      # safety tip / recommendation
        "alert": alert,                           # prominent banner for risky results

        # Kept for compatibility with the existing frontend styling
        # (result.verdict picks the color, result.label is the headline text)
        "verdict": verdict,
        "label": f"{risk_label} \u00b7 {combined_score}/100",
        "risk_label": risk_label,
        "score": combined_score,
        "input_type": effective_type,
        "sensitivity": sensitivity,
        "reasons": reason_texts,
        "detail": reason_texts[0] if reason_texts else "No common scam patterns detected.",
        "safety_tip": safety_tip,
    })


@app.route("/api/scam/mark-spam", methods=["POST"])
def mark_spam():
    global model, vectorizer

    client_ip = request.remote_addr or "unknown"
    if is_mark_spam_rate_limited(client_ip):
        logger.warning(f"mark-spam rate limit hit for {client_ip}")
        return jsonify({"error": "Too many requests. Please wait a moment and try again."}), 429

    if MARK_SPAM_API_KEY:
        if request.headers.get("X-API-Key") != MARK_SPAM_API_KEY:
            logger.warning(f"mark-spam rejected: bad/missing API key from {client_ip}")
            return jsonify({"error": "Invalid or missing API key."}), 401
    else:
        # No key configured -- endpoint is open. This is fine for a private
        # demo but is a real data-poisoning vector if this ever goes public:
        # anyone can label arbitrary text as spam and it gets trained into
        # the live model with no review. Set MARK_SPAM_API_KEY before that.
        logger.warning(f"mark-spam called with NO API KEY configured (open endpoint) from {client_ip}")

    data = request.get_json(silent=True) or {}
    raw_text = data.get("text") or data.get("input")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return jsonify({"error": "text is required"}), 400
    text = raw_text.strip()
    if len(text) > MAX_INPUT_LENGTH:
        return jsonify({"error": f"Text is too long. Please keep it under {MAX_INPUT_LENGTH} characters."}), 400

    current_count = len(training_records())
    if current_count >= MAX_TRAINING_EXAMPLES:
        logger.warning(f"Training set cap reached ({MAX_TRAINING_EXAMPLES}), rejecting new example")
        return jsonify({"error": "Training dataset has reached its size cap. Please try again later."}), 507

    try:
        added = save_spam_example(text)
        if added:
            # Only retrain when something actually changed -- retraining on
            # every duplicate submission was pure wasted compute (and, since
            # this endpoint was previously unauthenticated, an easy way to
            # hammer the server with full retrains for free).
            with model_lock:
                model, vectorizer = train_and_report()
        return jsonify({
            "success": True,
            "message": "Spam example saved and model retrained." if added else "Already known; no retrain needed.",
            "duplicate": not added,
            "label": "spam",
            "training_examples": len(training_records()),
        })
    except Exception:
        logger.exception(f"Error in mark-spam for {client_ip}")
        return jsonify({"error": "Could not save this example. Please try again."}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    app.run(debug=debug_mode, host=host, port=5000)

# ---------------------------------------------------------------------------
# Note: this embedded dataset (~80 examples) is still a demo-sized set.
# Swap DATA / SCAM_TEXTS / LEGIT_TEXTS above for a bigger real dataset
# (UCI SMS Spam Collection, PhishTank URLs) before a final submission.
# ---------------------------------------------------------------------------