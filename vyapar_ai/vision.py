"""Photo analysis: extract listing details from seller photos.

Two layers:

* Pixel checks (Pillow + numpy, always on): resolution, brightness, blur and
  the product's dominant colour. No model needed.
* A vision-language model (optional): reads category, fabric, pattern,
  colour, whether a size chart or fabric label is shown, and which shots
  exist. It is called through an OpenAI-compatible chat-completions API, so
  it works with Sarvam's endpoint or any self-hosted model that accepts
  images. Configure with VISION_API_BASE, VISION_MODEL and VISION_API_KEY.
  Without them (or if the call fails) the pixel checks still answer.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Optional, Protocol

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .catalog import COLORS, FABRICS, PATTERNS, TAXONOMY

SUBCATEGORIES = [s for subs in TAXONOMY.values() for s in subs]
SHOTS = ["front", "back", "close_up", "drape_or_worn", "label", "size_chart"]

MAX_FILES = 8
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
VLM_MAX_SIDE = 1024        # downscale before sending to the model

# Pixel-check thresholds (tuned on phone photos resized to 512 px).
MIN_SIDE = 500
DARK_BELOW = 60
BRIGHT_ABOVE = 235
BLURRY_BELOW = 0.5         # contrast-normalised Laplacian variance (see sharpness())


class PhotoError(ValueError):
    """A photo the seller uploaded cannot be used; the message says why."""


# ------------------------------------------------------------------ loading

@dataclass
class Photo:
    name: str
    image: Image.Image     # RGB, EXIF-rotated

    def jpeg_bytes(self, max_side: int = VLM_MAX_SIDE) -> bytes:
        img = self.image.copy()
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def jpeg_b64(self, max_side: int = VLM_MAX_SIDE) -> str:
        return base64.b64encode(self.jpeg_bytes(max_side)).decode("ascii")


def load_photos(files: list[tuple[str, bytes]]) -> list[Photo]:
    if not files:
        raise PhotoError("Upload at least one photo.")
    if len(files) > MAX_FILES:
        raise PhotoError(f"Upload at most {MAX_FILES} photos at a time.")
    photos = []
    for name, data in files:
        if len(data) > MAX_BYTES:
            raise PhotoError(f"{name} is larger than {MAX_BYTES // (1024 * 1024)} MB.")
        try:
            img = Image.open(io.BytesIO(data))
            fmt = img.format
            img.load()
        except (UnidentifiedImageError, OSError):
            raise PhotoError(f"{name} is not a readable image.") from None
        if fmt not in ALLOWED_FORMATS:
            raise PhotoError(f"{name}: use JPEG, PNG or WebP (got {fmt}).")
        photos.append(Photo(name, ImageOps.exif_transpose(img).convert("RGB")))
    return photos


# ------------------------------------------------------------- pixel checks

def _gray_small(img: Image.Image, side: int = 512) -> np.ndarray:
    g = img.convert("L")
    g.thumbnail((side, side))
    return np.asarray(g, dtype=np.float32)


def sharpness(img: Image.Image) -> float:
    """Laplacian variance divided by image variance, x100: low means blurry.

    Normalising by contrast keeps dark or low-contrast (but sharp) photos from
    being flagged as blurry.
    """
    a = _gray_small(img)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = (a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1])
    contrast = float(a.var())
    return 100.0 * float(lap.var()) / contrast if contrast > 1.0 else 0.0


def brightness(img: Image.Image) -> float:
    return float(_gray_small(img, 256).mean())


def _hue_to_color(h: float, s: float, v: float) -> str:
    """h in degrees, s/v in 0..1 -> one of catalog.COLORS."""
    if v < 0.18:
        return "black"
    if s < 0.15:
        return "white" if v > 0.75 else "black"
    if h < 12 or h >= 345:
        return "pink" if (s < 0.55 and v > 0.7) else "red"
    if h < 40:
        return "orange"
    if h < 70:
        return "yellow"
    if h < 170:
        return "green"
    if h < 255:
        return "blue"
    if h < 300:
        return "purple"
    return "pink"


def dominant_color(img: Image.Image) -> Optional[str]:
    """Main product colour.

    Product photos put the product near the centre, often against a room or
    wall rather than a pure white sweep. So: look at the central region, and
    vote among clearly coloured pixels weighted by saturation x brightness,
    which lets a pink kurti beat an off-white wall, a wooden table or skin.
    Only when almost nothing is coloured is the product white, grey or black.
    """
    w, h = img.size
    crop = img.crop((int(w * 0.2), int(h * 0.1), int(w * 0.8), int(h * 0.95)))
    crop.thumbnail((128, 128))
    hsv = np.asarray(crop.convert("HSV"), dtype=np.float32) / 255.0
    hh, ss, vv = hsv[..., 0].ravel() * 360, hsv[..., 1].ravel(), hsv[..., 2].ravel()
    chromatic = (ss >= 0.25) & (vv >= 0.25)
    if chromatic.mean() >= 0.06:
        votes: dict[str, float] = {}
        for a, b, c in zip(hh[chromatic], ss[chromatic], vv[chromatic]):
            name = _hue_to_color(a, b, c)
            votes[name] = votes.get(name, 0.0) + float(b * c)
        return max(votes, key=votes.get)
    # Neutral product: black if a sizeable dark area sits in the centre.
    return "black" if float((vv < 0.25).mean()) >= 0.15 else "white"


def photo_checks(p: Photo) -> dict:
    w, h = p.image.size
    b, sh = brightness(p.image), sharpness(p.image)
    issues = []
    if min(w, h) < MIN_SIDE:
        issues.append("low_resolution")
    if b < DARK_BELOW:
        issues.append("too_dark")
    elif b > BRIGHT_ABOVE:
        issues.append("overexposed")
    if sh < BLURRY_BELOW:
        issues.append("blurry")
    return {"name": p.name, "width": w, "height": h, "brightness": round(b, 1),
            "sharpness": round(sh, 1), "color": dominant_color(p.image), "issues": issues}


# --------------------------------------------------------- model extraction

ATTRIBUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_product_photo": {"type": "boolean"},
        "subcategory": {"type": "string", "enum": SUBCATEGORIES + ["other"]},
        "fabric": {"type": "string", "enum": FABRICS + ["unknown"]},
        "pattern": {"type": "string", "enum": PATTERNS + ["unknown"]},
        "color": {"type": "string", "enum": COLORS + ["unknown"]},
        "has_size_chart": {"type": "boolean"},
        "has_fabric_label": {"type": "boolean"},
        "shots": {"type": "array", "items": {"type": "string", "enum": SHOTS}},
        "suggested_title": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["is_product_photo", "subcategory", "fabric", "pattern", "color",
                 "has_size_chart", "has_fabric_label", "shots", "suggested_title",
                 "confidence"],
    "additionalProperties": False,
}

PROMPT = (
    "You are helping a new seller on an Indian marketplace list ONE product. "
    "Look at all the photos of that one product and fill in the JSON fields. "
    "Use only the allowed values listed below. Answer 'unknown' (or 'other' for "
    "subcategory) when the photos do not show something clearly.\n\n"
    "Subcategory definitions (women's ethnic garments look alike - use these):\n"
    "- saree: a long, unstitched garment draped around the body over a blouse, with a "
    "pallu (the decorative end worn over the shoulder). A printed or cotton saree with "
    "no special markers is plain 'saree'.\n"
    "- synthetic_saree: a saree clearly in a synthetic fabric (polyester, georgette, "
    "chiffon, art silk).\n"
    "- banarasi_silk_saree: a heavy silk saree woven with zari/brocade (metallic gold or "
    "silver thread and dense ornate motifs).\n"
    "- kurti: a single stitched top/tunic with sleeves and a neckline, worn on its own "
    "(not a drape).\n"
    "- cotton_suit_set: a matching set sold together of a kurti/top PLUS a bottom "
    "(salwar, palazzo or churidar) PLUS a dupatta.\n"
    "- dupatta: only the scarf/stole; no main garment (no kurti, no saree) is shown.\n"
    "- tshirt: a casual short-sleeved knit top with a round or V neck (not a button-up).\n"
    "- casual_shirt: a woven button-up shirt with a collar and a full front button placket.\n"
    "- bedsheet: a flat bed cover/sheet, often sold with matching pillow covers.\n"
    "- cushion_cover: a removable cover for a throw pillow/cushion (home decor).\n\n"
    "Disambiguation rules - follow these when the garment is ambiguous:\n"
    "- If the garment is draped or unstitched and has a pallu, it is a SAREE: choose "
    "'saree', 'synthetic_saree' or 'banarasi_silk_saree'. NEVER answer 'kurti' for a "
    "draped garment.\n"
    "- When you can see it is a saree but not which kind, prefer the generic 'saree'.\n"
    "- 'kurti' is only for a stitched, sleeved top worn on its own; if it comes with a "
    "matching bottom and dupatta as one set, use 'cotton_suit_set' instead.\n"
    "- 'dupatta' is only when the scarf/stole is the product on its own.\n\n"
    "Other fields: has_size_chart is true only if a measurement or size chart is visible "
    "in a photo. has_fabric_label is true only if a fabric, GSM or care label/card is "
    "visible. shots lists the kinds of shots present. suggested_title is a short, plain "
    "English listing title such as 'Pink printed cotton saree'. confidence is your "
    "confidence from 0 to 1.\n\n"
    "Allowed values:\n"
    f"subcategory: {', '.join(SUBCATEGORIES)}, other\n"
    f"fabric: {', '.join(FABRICS)}, unknown\n"
    f"pattern: {', '.join(PATTERNS)}, unknown\n"
    f"color: {', '.join(COLORS)}, unknown\n"
    f"shots: {', '.join(SHOTS)}\n\n"
    "Reply with the JSON object only."
)


def _clean(raw: dict) -> dict:
    """Keep only valid values; anything outside the allowed sets becomes None."""
    def pick(key, allowed):
        v = raw.get(key)
        return v if v in allowed else None
    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "is_product_photo": bool(raw.get("is_product_photo", True)),
        "subcategory": pick("subcategory", SUBCATEGORIES),
        "fabric": pick("fabric", FABRICS),
        "pattern": pick("pattern", PATTERNS),
        "color": pick("color", COLORS),
        "has_size_chart": raw.get("has_size_chart") is True,
        "has_fabric_label": raw.get("has_fabric_label") is True,
        "shots": [s for s in (raw.get("shots") or []) if s in SHOTS],
        "suggested_title": str(raw.get("suggested_title") or "").strip()[:120] or None,
        "confidence": max(0.0, min(1.0, conf)),
    }


def _parse_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("model did not return a JSON object")
    return data


class VisionExtractor(Protocol):
    name: str

    def extract(self, photos: list[Photo]) -> Optional[dict]: ...


class NoModelExtractor:
    """No vision model configured: only pixel checks run."""

    name = "pixel-checks"

    def extract(self, photos: list[Photo]) -> Optional[dict]:
        return None


class ChatCompletionsVLM:
    """Vision model behind an OpenAI-compatible /chat/completions endpoint (e.g. Sarvam)."""

    name = "vlm"

    def __init__(self, base_url: str, model: str, api_key: str,
                 timeout: float = 60.0, client: Any = None, json_schema: bool = True):
        import httpx
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.json_schema = json_schema
        self.headers = {"Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"}
        self.client = client or httpx.Client(timeout=timeout)
        self.last_error: Optional[str] = None

    def build_request(self, photos: list[Photo], json_schema: Optional[bool] = None,
                      context: Optional[str] = None) -> dict:
        if json_schema is None:
            json_schema = self.json_schema
        text = PROMPT
        if context:
            text += ("\n\nText read from the photos by the OCR step (partial; the images are "
                     "the source of truth, use this only as a hint):\n" + context)
        content: list[dict] = [{"type": "text", "text": text}]
        for p in photos:
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{p.jpeg_b64()}"}})
        body: dict = {"model": self.model, "temperature": 0,
                      "messages": [{"role": "user", "content": content}]}
        if json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "listing_attributes", "strict": True, "schema": ATTRIBUTE_SCHEMA}}
        return body

    def _post(self, body: dict) -> dict:
        import httpx

        r = self.client.post(self.url, headers=self.headers, json=body)
        if r.is_error:
            # Include the provider's explanation, not just its HTTP status.
            try:
                error = r.json().get("error", {})
                detail = error.get("message", "") if isinstance(error, dict) else ""
            except (ValueError, AttributeError):
                detail = ""
            if detail:
                key = self.headers["Authorization"].removeprefix("Bearer ")
                detail = str(detail).replace(key, "[REDACTED]") if key else str(detail)
                raise httpx.HTTPStatusError(
                    f"HTTP {r.status_code}: {detail[:240]}", request=r.request, response=r)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return _clean(_parse_json(text))

    def extract(self, photos: list[Photo], context: Optional[str] = None) -> Optional[dict]:
        self.last_error = None
        try:
            return self._post(self.build_request(photos, context=context))
        except Exception as first:  # noqa: BLE001 - may be a json_schema rejection
            if not self.json_schema:
                self.last_error = f"{type(first).__name__}: {first}"[:300]
                raise
            # Some OpenAI-compatible endpoints/models reject response_format
            # json_schema (e.g. HTTP 400). Retry ONCE without it before giving up.
            try:
                result = self._post(self.build_request(photos, json_schema=False, context=context))
            except Exception:
                # Both attempts failed: surface the original error, keep it recorded.
                self.last_error = f"{type(first).__name__}: {first}"[:300]
                raise first
            self.last_error = None
            return result


class FallbackExtractor:
    """Use the model when it works; on any error keep going with pixel checks only."""

    def __init__(self, primary: VisionExtractor):
        self.primary = primary
        self.name = primary.name
        self.last_error: Optional[str] = None

    def extract(self, photos: list[Photo]) -> Optional[dict]:
        try:
            self.last_error = None
            return self.primary.extract(photos)
        except Exception as e:  # noqa: BLE001 - network, HTTP or parse errors all degrade
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            return None


DOC_AI_DEFAULT_BASE = "https://api.sarvam.ai"


def _html_to_text(html: str) -> str:
    """Visible text from a Doc-AI HTML page: drops tags, styles, scripts and images."""
    s = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    s = re.sub(r"<img\b[^>]*>", " ", s, flags=re.I)
    s = re.sub(r"<(br|hr)\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</(p|div|li|tr|h[1-6]|figure|table)>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


class DocAiDigitizer:
    """Sarvam Doc-AI job API: submit a photo, poll it, fetch the rendered output.

    POST {base}/doc-ai/v1/job/digitise          multipart + Idempotency-Key -> {job_id}
    GET  {base}/doc-ai/v1/job/{id}/status       -> {status, usage, ...}
    GET  {base}/doc-ai/v1/job/{id}/download-url -> {method, url, headers}
    """

    TERMINAL = {"completed", "partially_completed", "failed", "rejected"}
    OK = {"completed", "partially_completed"}

    def __init__(self, api_key: str, base_url: str = DOC_AI_DEFAULT_BASE,
                 language: str = "hi-IN", output_format: str = "html",
                 content_type: str = "printed", auto_orient: bool = True,
                 timeout: float = 60.0, poll_interval: float = 1.5,
                 max_wait: float = 120.0, client: Any = None, sleep: Any = None):
        import httpx
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.language = language
        self.output_format = output_format
        self.content_type = content_type
        self.auto_orient = auto_orient
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.client = client or httpx.Client(timeout=timeout)
        self.sleep = sleep or time.sleep

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {"api-subscription-key": self.key}
        if extra:
            h.update(extra)
        return h

    def submit(self, photo: Photo) -> str:
        r = self.client.post(
            self.base + "/doc-ai/v1/job/digitise",
            headers=self._headers({"Idempotency-Key": uuid.uuid4().hex}),
            data={"language": self.language, "output_format": self.output_format,
                  "content_type": self.content_type,
                  "auto_orient": "true" if self.auto_orient else "false"},
            files={"file": (photo.name or "photo.jpg", photo.jpeg_bytes(), "image/jpeg")},
        )
        r.raise_for_status()
        return r.json()["job_id"]

    def wait(self, job_id: str) -> dict:
        deadline = time.monotonic() + self.max_wait
        while True:
            r = self.client.get(f"{self.base}/doc-ai/v1/job/{job_id}/status",
                                headers=self._headers())
            r.raise_for_status()
            status = r.json()
            if str(status.get("status", "")).lower() in self.TERMINAL:
                return status
            if time.monotonic() > deadline:
                raise TimeoutError(f"Doc-AI job {job_id} did not finish in {self.max_wait}s")
            self.sleep(self.poll_interval)

    def download(self, job_id: str) -> str:
        r = self.client.get(f"{self.base}/doc-ai/v1/job/{job_id}/download-url",
                            headers=self._headers())
        r.raise_for_status()
        spec = r.json()
        d = self.client.get(spec["url"], headers=spec.get("headers") or {})
        d.raise_for_status()
        return d.text

    def digitise(self, photo: Photo) -> str:
        """Run the whole job for one photo and return the rendered HTML."""
        job_id = self.submit(photo)
        status = self.wait(job_id)
        if str(status.get("status", "")).lower() not in self.OK:
            raise RuntimeError(f"Doc-AI job {job_id} ended as {status.get('status')}")
        return self.download(job_id)


class DocAiLLMExtractor:
    """Doc-AI digitise -> LLM extraction.

    Each photo is digitised by the Doc-AI job API; the text it reads (fabric/GSM
    labels, size charts, any printed matter) is handed to the chat model together
    with the photos, so the model gets both the OCR text and the image.
    """

    def __init__(self, digitizer: DocAiDigitizer, vlm: "ChatCompletionsVLM",
                 max_photos: int = 3, max_chars: int = 4000):
        self.digitizer = digitizer
        self.vlm = vlm
        self.max_photos = max_photos
        self.max_chars = max_chars
        self.name = "doc-ai+llm"
        self.last_error: Optional[str] = None

    def ocr_text(self, photos: list[Photo]) -> str:
        chunks: list[str] = []
        for p in photos[: self.max_photos]:
            try:
                text = _html_to_text(self.digitizer.digitise(p))
                if text:
                    chunks.append(f"[{p.name}] {text}")
            except Exception as e:  # noqa: BLE001 - OCR is only a hint; never block the answer
                chunks.append(f"[{p.name}] (Doc-AI failed: {type(e).__name__})")
        return "\n".join(chunks)[: self.max_chars]

    def extract(self, photos: list[Photo]) -> Optional[dict]:
        self.last_error = None
        context = self.ocr_text(photos)
        return self.vlm.extract(photos, context=context or None)


def default_extractor() -> VisionExtractor:
    """Pick the best configured extractor.

    Doc-AI digitise + LLM when both a Doc-AI key and a chat model are set;
    otherwise the chat model on its own; otherwise pixel checks only.
    """
    model, key = os.getenv("VISION_MODEL"), os.getenv("VISION_API_KEY")
    if not (model and key):
        return NoModelExtractor()
    base = os.getenv("VISION_API_BASE", "https://api.sarvam.ai/v1")
    schema = os.getenv("VISION_JSON_SCHEMA", "1") not in ("0", "false", "no")
    vlm = ChatCompletionsVLM(base, model, key, json_schema=schema)
    doc_key = os.getenv("DOC_AI_API_KEY") or os.getenv("SARVAM_API_KEY")
    doc_on = os.getenv("DOC_AI_ENABLED", "1") not in ("0", "false", "no")
    if doc_key and doc_on:
        try:
            max_photos = int(os.getenv("DOC_AI_MAX_PHOTOS", "3"))
        except ValueError:
            max_photos = 3
        digitizer = DocAiDigitizer(
            doc_key,
            base_url=os.getenv("DOC_AI_API_BASE", DOC_AI_DEFAULT_BASE),
            language=os.getenv("DOC_AI_LANGUAGE", "hi-IN"),
            content_type=os.getenv("DOC_AI_CONTENT_TYPE", "printed"),
        )
        return FallbackExtractor(DocAiLLMExtractor(digitizer, vlm, max_photos=max_photos))
    return FallbackExtractor(vlm)


# ------------------------------------------------------------------ result

ISSUE_TEXT = {
    "blurry": "dhundhli hai",
    "too_dark": "bahut andheri hai",
    "overexposed": "zyada roshni se safed ho gayi hai",
    "low_resolution": "chhoti (low resolution) hai",
}

SHOT_TEXT = {
    "back": "peeche ki photo",
    "close_up": "kapde ka close-up",
    "drape_or_worn": "pehne hue / drape ki photo",
    "size_chart": "size chart",
    "label": "fabric/GSM label",
}


@dataclass
class PhotoReport:
    backend: str
    photos: list[dict]
    suggested: dict
    issues: list[str]
    missing_shots: list[str]
    nudges: list[dict] = field(default_factory=list)
    model_error: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def analyze_photos(photos: list[Photo], extractor: Optional[VisionExtractor] = None,
                   seller_name: str = "Seller") -> PhotoReport:
    extractor = extractor or default_extractor()
    checks = [photo_checks(p) for p in photos]
    attrs = extractor.extract(photos)
    error = getattr(extractor, "last_error", None)

    # Pixel colour vote across photos, used when the model is absent or unsure.
    votes: dict[str, int] = {}
    for c in checks:
        if c["color"]:
            votes[c["color"]] = votes.get(c["color"], 0) + 1
    pixel_color = max(votes, key=votes.get) if votes else None

    a = attrs or {}
    shots = a.get("shots") or []
    suggested = {
        "image_count": len(photos),
        "color": a.get("color") or pixel_color,
        "subcategory": a.get("subcategory"),
        "fabric": a.get("fabric"),
        "pattern": a.get("pattern"),
        "title": a.get("suggested_title"),
        "has_size_chart": bool(a.get("has_size_chart") or "size_chart" in shots),
        "has_fabric_card": bool(a.get("has_fabric_label") or "label" in shots),
        "confidence": a.get("confidence"),
        "is_product_photo": a.get("is_product_photo", True),
    }
    issues = sorted({i for c in checks for i in c["issues"]})
    missing = [s for s in ("back", "close_up", "drape_or_worn") if attrs and s not in shots]
    if attrs and not suggested["has_size_chart"]:
        missing.append("size_chart")

    nudges = []
    bad = [c for c in checks if c["issues"]]
    if bad:
        parts = [f"photo {checks.index(c) + 1} {', '.join(ISSUE_TEXT[i] for i in c['issues'])}"
                 for c in bad[:3]]
        nudges.append({"type": "photo_quality", "text": (
            f"{seller_name} ji, " + "; ".join(parts) +
            ". Din ki roshni mein, saaf background par dobara photo lein; "
            "achhi photo se returns kam hote hain."),
            "actions": [{"id": "dismiss", "label": "Theek hai"}]})
    # During testing, one photo is enough; do not request additional angles.
    if len(photos) < 1:
        nudges.append({"type": "photo_shots", "text": (
            "Product ki kam se kam 1 saaf photo daalein."),
            "actions": [{"id": "dismiss", "label": "Theek hai"}]})
    if attrs and not suggested["is_product_photo"]:
        nudges.insert(0, {"type": "photo_quality", "text": (
            "Yeh photo product ki nahi lag rahi. Kripya product ki saaf photo daalein."),
            "actions": [{"id": "dismiss", "label": "Theek hai"}]})

    backend = extractor.name if attrs else (
        "pixel-checks (vision model unavailable)" if error else "pixel-checks")
    return PhotoReport(backend=backend, photos=checks, suggested=suggested, issues=issues,
                       missing_shots=missing, nudges=nudges, model_error=error)
