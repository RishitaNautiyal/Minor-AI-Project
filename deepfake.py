"""OffGrid Deepfake Image Checker using Hugging Face InferenceClient."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, g, jsonify, request
from flask_cors import CORS
from huggingface_hub import InferenceClient


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

load_dotenv(".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deepfake_checker")


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

MAX_IMAGE_BYTES = 8 * 1024 * 1024
DATASET_MAX_IMAGES = int(os.getenv("DATASET_MAX_IMAGES", "5000"))

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

HF_MODEL = os.getenv(
    "HF_MODEL",
    "prithivMLmods/deepfake-detector-model-v1"
).strip()

HF_PROVIDER = os.getenv(
    "HF_PROVIDER",
    "hf-inference"
).strip()

DATASET_ROOT = Path(__file__).with_name("deepfake_data")
DATASET_INDEX_PATH = DATASET_ROOT / "index.json"  # hash -> {filename, label}, avoids re-hashing every file on every upload

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


# -------------------------------------------------------------------
# Label keywords
# -------------------------------------------------------------------

FAKE_TOKENS = (
    "fake",
    "deepfake",
    "synthetic",
    "artificial",
    "ai-generated",
    "ai_generated",
    "generated",
    "manipulated",
    "tampered",
    "spoof",
    "forged",
)

REAL_TOKENS = (
    "real",
    "authentic",
    "genuine",
    "original",
    "natural",
    "human",
)


# -------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------

app = Flask(__name__)

# Give a little headroom over the image limit for multipart overhead
# (boundaries, headers, the 'label' field on /feedback) so a genuinely
# 8MB image isn't rejected before your own size check even runs.
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES + (256 * 1024)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://127.0.0.1:8000",
                "http://localhost:8000",
                "http://127.0.0.1:5500",
                "http://localhost:5500",
            ]
        }
    },
)

# ---------------------------------------------------------------------------
# Rate limiting -- /feedback writes to disk, so without this it's a real
# storage-exhaustion DoS vector, not just an abuse nuisance.
# ---------------------------------------------------------------------------
RATE_LIMIT_MAX_REQUESTS = 15
RATE_LIMIT_WINDOW_SECONDS = 60
_request_log: dict[str, deque] = defaultdict(deque)


def is_rate_limited(ip: str) -> bool:
    now = time.time()
    window = _request_log[ip]
    while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT_MAX_REQUESTS:
        return True
    window.append(now)
    return False


@app.before_request
def assign_request_id():
    g.request_id = uuid.uuid4().hex[:8]


@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.errorhandler(429)
def rate_limited(_error):
    return jsonify({
        "success": False,
        "error": "Too many requests. Please wait a moment and try again.",
    }), 429


# -------------------------------------------------------------------
# Hugging Face client
# -------------------------------------------------------------------

_hf_client: InferenceClient | None = None


def get_huggingface_client() -> InferenceClient:
    global _hf_client

    if _hf_client is not None:
        return _hf_client

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing. Add HF_TOKEN=your_token to .env"
        )

    if not HF_MODEL:
        raise RuntimeError(
            "HF_MODEL is missing. Add HF_MODEL=model_name to .env"
        )

    try:
        _hf_client = InferenceClient(
            model=HF_MODEL,
            provider=HF_PROVIDER or None,
            api_key=HF_TOKEN,
            timeout=60,
        )
    except TypeError:
        _hf_client = InferenceClient(
            model=HF_MODEL,
            token=HF_TOKEN,
            timeout=60,
        )

    return _hf_client


def get_label(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("label", ""))

    return str(getattr(item, "label", ""))


def get_score(item: Any) -> float:
    if isinstance(item, dict):
        return float(item.get("score", 0))

    return float(getattr(item, "score", 0))


def classify_label(label: str) -> str | None:
    label = label.lower().strip()

    for token in FAKE_TOKENS:
        if token in label:
            return "fake"

    for token in REAL_TOKENS:
        if token in label:
            return "real"

    return None


# -------------------------------------------------------------------
# Hugging Face analysis
# -------------------------------------------------------------------

def analyze_with_huggingface(image_data: bytes, retries: int = 3) -> dict:
    """Send image to Hugging Face and return the model prediction.

    Retries on the common "model is loading" cold-start case, which the
    free Inference API returns as a 503 the first time a model is called
    after being idle.
    """

    client = get_huggingface_client()

    last_error: Exception | None = None
    predictions = None
    for attempt in range(retries):
        try:
            predictions = client.image_classification(
                image_data,
                model=HF_MODEL,
            )
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001 -- huggingface_hub raises several error types
            last_error = exc
            message = str(exc).lower()
            is_loading = "loading" in message or "503" in message
            if is_loading and attempt < retries - 1:
                wait_for = 5 * (attempt + 1)
                logger.info(f"HF model appears to be loading, retrying in {wait_for}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait_for)
                continue
            break

    if last_error is not None:
        raise last_error

    if not predictions:
        raise RuntimeError(
            f"Hugging Face model '{HF_MODEL}' returned no predictions."
        )

    fake_score = 0.0
    real_score = 0.0

    raw_predictions = []

    for item in predictions:
        label = get_label(item)
        score = get_score(item)

        raw_predictions.append(
            {
                "label": label,
                "score": round(score, 6),
            }
        )

        polarity = classify_label(label)

        if polarity == "fake":
            fake_score += score

        elif polarity == "real":
            real_score += score

    if fake_score == 0 and real_score == 0:
        labels = [item["label"] for item in raw_predictions]

        raise RuntimeError(
            f"Model returned unsupported labels: {labels}. "
            "Use a model that provides real/fake style labels."
        )

    total = fake_score + real_score

    if total > 0:
        fake_score /= total
        real_score /= total

    if fake_score >= real_score:
        prediction = "fake"
        confidence = fake_score
    else:
        prediction = "real"
        confidence = real_score

    # ---------------------------------------------------------------
    # Confidence level
    # ---------------------------------------------------------------

    if confidence >= 0.80:
        confidence_level = "High confidence"
    elif confidence >= 0.60:
        confidence_level = "Moderate confidence"
    else:
        confidence_level = "Uncertain"

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "confidence_level": confidence_level,
        "fake_score": round(fake_score, 4),
        "real_score": round(real_score, 4),
        "raw_predictions": raw_predictions,
    }


# -------------------------------------------------------------------
# Image validation
# -------------------------------------------------------------------

def image_mime_type(data: bytes) -> str | None:

    for signature, mime_type in IMAGE_SIGNATURES:
        if data.startswith(signature):
            return mime_type

    if len(data) >= 12:
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"

    return None


# -------------------------------------------------------------------
# Duplicate detection (index-based -- O(1) instead of re-hashing every
# stored file on every upload, which doesn't scale past a small dataset)
# -------------------------------------------------------------------

def calculate_image_hash(data: bytes) -> str:
    """Create a SHA-256 hash for the exact image bytes."""
    return hashlib.sha256(data).hexdigest()


def load_dataset_index() -> dict:
    if not DATASET_INDEX_PATH.exists():
        return {}
    try:
        return json.loads(DATASET_INDEX_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Dataset index is missing or corrupt, starting fresh")
        return {}


def save_dataset_index(index: dict) -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_INDEX_PATH.write_text(json.dumps(index))


def find_duplicate_image(data: bytes) -> tuple[str | None, str | None]:
    new_hash = calculate_image_hash(data)
    index = load_dataset_index()
    entry = index.get(new_hash)
    if entry:
        return entry["filename"], entry["label"]
    return None, None


# -------------------------------------------------------------------
# Save image without duplicates
# -------------------------------------------------------------------

def save_labeled_image(
    data: bytes,
    mime_type: str,
    label: str,
) -> tuple[str, bool, str | None]:

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    existing_filename, existing_label = find_duplicate_image(data)

    if existing_filename:
        return (
            existing_filename,
            True,
            existing_label,
        )

    folder = DATASET_ROOT / label
    folder.mkdir(parents=True, exist_ok=True)

    image_hash = calculate_image_hash(data)

    extension = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }[mime_type]

    filename = f"{image_hash}{extension}"

    file_path = folder / filename

    if not file_path.exists():
        file_path.write_bytes(data)

    index = load_dataset_index()
    index[image_hash] = {"filename": filename, "label": label}
    save_dataset_index(index)

    return (
        filename,
        False,
        None,
    )


def rebuild_dataset_index() -> int:
    """One-time maintenance helper: rebuilds index.json from files already
    on disk (e.g. if you had a dataset saved before this index existed).
    Not called automatically -- run manually if you ever need it:
        python -c "from deepfake import rebuild_dataset_index; rebuild_dataset_index()"
    """
    index: dict = {}
    for label in ("real", "fake"):
        folder = DATASET_ROOT / label
        if not folder.exists():
            continue
        for existing_file in folder.iterdir():
            if not existing_file.is_file():
                continue
            try:
                file_hash = calculate_image_hash(existing_file.read_bytes())
            except OSError:
                continue
            index[file_hash] = {"filename": existing_file.name, "label": label}
    save_dataset_index(index)
    logger.info(f"Rebuilt dataset index with {len(index)} entries")
    return len(index)


# -------------------------------------------------------------------
# Error handling
# -------------------------------------------------------------------

@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(
        {
            "success": False,
            "error": "Image must be 8 MB or smaller.",
        }
    ), 413


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/")
def home():
    return jsonify(
        {
            "success": True,
            "status": "Deepfake checker API is running",
            "model": HF_MODEL,
            "provider": HF_PROVIDER,
            "token_configured": bool(HF_TOKEN),
        }
    )


@app.get("/api/deepfake/health")
def health():
    return jsonify(
        {
            "success": True,
            "model": HF_MODEL,
            "provider": HF_PROVIDER,
            "token_configured": bool(HF_TOKEN),
        }
    )


# -------------------------------------------------------------------
# Deepfake analysis
# -------------------------------------------------------------------

@app.post("/api/deepfake/analyze")
def analyze_deepfake():

    client_ip = request.remote_addr or "unknown"
    if is_rate_limited(client_ip):
        logger.warning(f"[{g.request_id}] Rate limit hit for {client_ip}")
        return jsonify({
            "success": False,
            "error": "Too many requests. Please wait a moment and try again.",
        }), 429

    image = request.files.get("image")

    if image is None or not image.filename:
        return jsonify(
            {
                "success": False,
                "error": "An image file is required.",
            }
        ), 400

    if image.mimetype not in ALLOWED_MIME_TYPES:
        return jsonify(
            {
                "success": False,
                "error": "Only JPG, PNG, and WEBP images are supported.",
            }
        ), 400

    data = image.read(MAX_IMAGE_BYTES + 1)

    if len(data) > MAX_IMAGE_BYTES:
        return jsonify(
            {
                "success": False,
                "error": "Image must be 8 MB or smaller.",
            }
        ), 413

    detected_type = image_mime_type(data)

    if detected_type is None or detected_type != image.mimetype:
        return jsonify(
            {
                "success": False,
                "error": "The uploaded file is not a valid image.",
            }
        ), 400

    try:
        result = analyze_with_huggingface(data)

    except Exception as exc:
        # Log the full error server-side (with a request id to correlate),
        # but don't hand the raw exception string to the client -- it can
        # leak internal details (paths, library internals, etc).
        logger.error(f"[{g.request_id}] Hugging Face analysis failed: {exc}")
        return jsonify(
            {
                "success": False,
                "error": "Could not reach the deepfake detection service. Please try again shortly.",
                "request_id": g.request_id,
                "model": HF_MODEL,
            }
        ), 503

    prediction = result["prediction"]
    confidence = result["confidence"]
    confidence_level = result["confidence_level"]

    if prediction == "fake":
        verdict = "Deepfake"
        label = "Likely manipulated"
        alert = {
            "show": True,
            "level": "critical" if confidence >= 0.80 else "warning",
            "message": (
                "\u26a0\ufe0f This image shows strong signs of AI manipulation. "
                "Don't trust or share it as authentic without independent verification."
                if confidence >= 0.80 else
                "\u26a0\ufe0f This image has some signs of manipulation. Verify the source before trusting it."
            ),
        }
    else:
        verdict = "Real"
        label = "Likely authentic"
        alert = {"show": False, "level": None, "message": None}

    logger.info(f"[{g.request_id}] Analyzed image (bytes={len(data)}) -> {prediction} ({confidence_level}, {confidence:.2f})")

    return jsonify(
        {
            "success": True,
            "prediction": prediction,
            "label": label,
            "verdict": verdict,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "deepfake_score": round(result["fake_score"] * 100),
            "real_score": round(result["real_score"] * 100),
            "scores": {
                "fake": result["fake_score"],
                "real": result["real_score"],
            },
            "alert": alert,
            "raw_predictions": result["raw_predictions"],
            "source": "Hugging Face Inference API",
            "model": HF_MODEL,
            "filename": Path(image.filename).name,
            "disclaimer": (
                "This AI result is not proof that an image is real or fake."
            ),
        }
    )


# -------------------------------------------------------------------
# User feedback
# -------------------------------------------------------------------

@app.post("/api/deepfake/feedback")
def save_feedback():

    client_ip = request.remote_addr or "unknown"
    if is_rate_limited(client_ip):
        logger.warning(f"[{g.request_id}] Rate limit hit for {client_ip} on /feedback")
        return jsonify({
            "success": False,
            "error": "Too many requests. Please wait a moment and try again.",
        }), 429

    index = load_dataset_index()
    if len(index) >= DATASET_MAX_IMAGES:
        logger.warning(f"[{g.request_id}] Dataset cap reached ({DATASET_MAX_IMAGES}), rejecting new feedback image")
        return jsonify({
            "success": False,
            "error": "The feedback dataset has reached its storage cap. Please try again later.",
        }), 507

    label = (request.form.get("label") or "").lower()

    image = request.files.get("image")

    if label not in {"real", "fake"}:
        return jsonify(
            {
                "success": False,
                "error": "label must be real or fake.",
            }
        ), 400

    if image is None or not image.filename:
        return jsonify(
            {
                "success": False,
                "error": "An image file is required.",
            }
        ), 400

    if image.mimetype not in ALLOWED_MIME_TYPES:
        return jsonify(
            {
                "success": False,
                "error": "Only JPG, PNG, and WEBP images are supported.",
            }
        ), 400

    data = image.read(MAX_IMAGE_BYTES + 1)

    if len(data) > MAX_IMAGE_BYTES:
        return jsonify(
            {
                "success": False,
                "error": "Image must be 8 MB or smaller.",
            }
        ), 413

    detected_type = image_mime_type(data)

    if detected_type is None or detected_type != image.mimetype:
        return jsonify(
            {
                "success": False,
                "error": "The uploaded file is not a valid image.",
            }
        ), 400

    filename, duplicate, existing_label = save_labeled_image(
        data,
        detected_type,
        label,
    )

    if duplicate:
        return jsonify(
            {
                "success": True,
                "duplicate": True,
                "label": label,
                "filename": filename,
                "existing_label": existing_label,
                "message": (
                    f"This image already exists in the "
                    f"'{existing_label}' dataset and was not saved again."
                ),
            }
        )

    return jsonify(
        {
            "success": True,
            "duplicate": False,
            "label": label,
            "filename": filename,
            "saved_to": str(DATASET_ROOT / label),
            "message": "Image saved successfully.",
            "notice": (
                "This image is now stored permanently to help improve the model. "
                "Your frontend should show this notice to the user BEFORE they submit, "
                "not just in this response -- especially important since users may be minors."
            ),
        }
    )


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------

if __name__ == "__main__":

    if not HF_TOKEN:
        print("\n*** WARNING: HF_TOKEN is not set. Requests will fail until you set it in .env ***\n")

    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(
        debug=debug_mode,
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=5001,
    )