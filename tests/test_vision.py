import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFilter

from vyapar_ai import vision
from vyapar_ai.vision import (ChatCompletionsVLM, FallbackExtractor, Photo, PhotoError,
                              analyze_photos, load_photos, photo_checks)


def garment(color, size=(800, 1000)):
    img = Image.new("RGB", size, (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([200, 150, 600, 900], fill=color)
    for x in range(220, 580, 40):
        for y in range(170, 880, 40):
            d.ellipse([x, y, x + 14, y + 14], fill=(255, 215, 0))
    return img


def png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("rgb,name", [((200, 20, 40), "red"), ((240, 120, 170), "pink"),
                                      ((30, 60, 180), "blue"), ((30, 140, 60), "green")])
def test_dominant_color_ignores_white_background(rgb, name):
    assert photo_checks(Photo("p", garment(rgb)))["color"] == name


def test_quality_checks():
    sharp = photo_checks(Photo("ok", garment((200, 20, 40))))
    assert sharp["issues"] == []
    assert "blurry" in photo_checks(Photo("b", garment((200, 20, 40)).filter(ImageFilter.GaussianBlur(6))))["issues"]
    dark = photo_checks(Photo("d", Image.eval(garment((200, 20, 40)), lambda v: v // 6)))
    assert dark["issues"] == ["too_dark"]          # dark but sharp: not flagged blurry
    assert "low_resolution" in photo_checks(Photo("s", garment((200, 20, 40), (300, 375))))["issues"]


def test_load_photos_validation():
    assert len(load_photos([("a.png", png(garment((1, 2, 3))))])) == 1
    with pytest.raises(PhotoError, match="not a readable image"):
        load_photos([("a.txt", b"hello")])
    gif = io.BytesIO()
    garment((1, 2, 3)).save(gif, format="GIF")
    with pytest.raises(PhotoError, match="JPEG, PNG or WebP"):
        load_photos([("a.gif", gif.getvalue())])
    with pytest.raises(PhotoError, match="at most"):
        load_photos([("a.png", b"")] * (vision.MAX_FILES + 1))


MODEL_JSON = {"is_product_photo": True, "subcategory": "kurti", "fabric": "cotton",
              "pattern": "printed", "color": "pink", "has_size_chart": False,
              "has_fabric_label": True, "shots": ["front", "close_up", "label"],
              "suggested_title": "Pink printed cotton kurti", "confidence": 0.86}


def fake_vlm(content, status=200, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})
    return ChatCompletionsVLM("https://api.sarvam.ai/v1", "my-sarvam-vlm", "key",
                              client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_vlm_request_and_extraction():
    seen = []
    vlm = fake_vlm(json.dumps(MODEL_JSON), seen=seen)
    report = analyze_photos([Photo("a", garment((240, 120, 170)))] * 2, vlm, "Ramesh")
    body = seen[0]
    assert body["model"] == "my-sarvam-vlm"
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "text" and sum(p["type"] == "image_url" for p in parts) == 2
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert body["response_format"]["type"] == "json_schema"
    s = report.suggested
    assert (s["subcategory"], s["fabric"], s["pattern"], s["color"]) == ("kurti", "cotton", "printed", "pink")
    assert s["has_fabric_card"] is True and s["has_size_chart"] is False and s["image_count"] == 2
    assert report.backend == "vlm"
    assert "back" in report.missing_shots and "size_chart" in report.missing_shots
    assert not any(n["type"] == "photo_shots" for n in report.nudges)


def test_vlm_output_is_sanitised():
    bad = dict(MODEL_JSON, fabric="velvet", subcategory="rocket", confidence=7)
    vlm = fake_vlm("Here you go:\n```json\n" + json.dumps(bad) + "\n```")
    s = analyze_photos([Photo("a", garment((240, 120, 170)))], vlm).suggested
    assert s["fabric"] is None and s["subcategory"] is None and s["confidence"] == 1.0
    assert s["color"] == "pink"


def test_vlm_failure_falls_back_to_pixel_checks():
    ext = FallbackExtractor(fake_vlm("oops", status=500))
    report = analyze_photos([Photo("a", garment((30, 60, 180)))], ext)
    assert report.backend.startswith("pixel-checks")
    assert "HTTPStatusError" in report.model_error
    assert report.suggested["color"] == "blue" and report.suggested["subcategory"] is None


def test_analyze_endpoint_without_model(monkeypatch):
    import api.main as m
    monkeypatch.delenv("VISION_MODEL", raising=False)
    m.vision_extractor.cache_clear()
    client = TestClient(m.app)
    files = [("files", ("a.png", png(garment((200, 20, 40))), "image/png")),
             ("files", ("b.png", png(garment((200, 20, 40)).filter(ImageFilter.GaussianBlur(6))), "image/png"))]
    r = client.post("/api/photos/analyze", files=files, data={"seller_name": "Ramesh"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["backend"] == "pixel-checks"
    assert d["suggested"]["color"] == "red" and d["suggested"]["image_count"] == 2
    assert d["issues"] == ["blurry"]
    assert d["nudges"][0]["text"].startswith("Ramesh ji, photo 2 dhundhli hai")
    bad = client.post("/api/photos/analyze", files=[("files", ("x.txt", b"hi", "text/plain"))])
    assert bad.status_code == 422 and "not a readable image" in bad.json()["detail"]
    m.vision_extractor.cache_clear()


def room_photo(garment_rgb, size=(300, 450)):
    """Model in a room: off-white panelled wall, wooden table, skin, embroidered kurti."""
    w, h = size
    img = Image.new("RGB", size, (226, 222, 216))
    d = ImageDraw.Draw(img)
    for x in range(0, w, 60):
        d.rectangle([x + 6, 20, x + 54, h - 60], outline=(205, 200, 195), width=2)
    d.rectangle([0, h - 60, w, h], fill=(200, 190, 175))
    d.rectangle([5, int(h * .55), 70, h - 40], fill=(120, 80, 50))
    d.ellipse([w // 2 - 18, 40, w // 2 + 18, 85], fill=(205, 160, 130))
    d.polygon([(w // 2 - 45, 95), (w // 2 + 45, 95), (w // 2 + 75, h - 70), (w // 2 - 75, h - 70)],
              fill=garment_rgb)
    for x in range(w // 2 - 60, w // 2 + 60, 14):
        for y in range(110, h - 80, 18):
            d.ellipse([x, y, x + 4, y + 4], fill=(250, 240, 240))
    return img


@pytest.mark.parametrize("rgb,name", [((222, 48, 110), "pink"), ((30, 60, 180), "blue"),
                                      ((240, 240, 236), "white"), ((25, 25, 30), "black")])
def test_dominant_color_in_a_room_photo(rgb, name):
    # Regression: a pink kurti against an off-white wall used to come out "white".
    assert vision.dominant_color(room_photo(rgb)) == name


def test_doc_ai_digitise_then_llm():
    from vyapar_ai.vision import ChatCompletionsVLM, DocAiDigitizer, DocAiLLMExtractor
    seen = {"submit": None, "chat": None}
    html_page = ("<html><head><style>b{}</style></head><body>"
                 "<div class='page-body-container'><p class='paragraph'>100% Cotton, "
                 "GSM 180, Made in India</p></div></body></html>")

    def handler(request):
        url = str(request.url)
        if url.endswith("/doc-ai/v1/job/digitise"):
            seen["submit"] = request
            return httpx.Response(201, json={"job_id": "job-1", "status": "pending", "run_id": "r1"})
        if url.endswith("/doc-ai/v1/job/job-1/status"):
            return httpx.Response(200, json={"status": "completed", "usage": {"pages_total": 1}})
        if url.endswith("/doc-ai/v1/job/job-1/download-url"):
            return httpx.Response(200, json={"method": "GET", "url": "https://cdn.example/out.html",
                                             "headers": {}, "expires_at": "x"})
        if url == "https://cdn.example/out.html":
            return httpx.Response(200, text=html_page)
        if url.endswith("/chat/completions"):
            seen["chat"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(MODEL_JSON)}}]})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    digitizer = DocAiDigitizer("test-key", client=client, sleep=lambda s: None)
    vlm = ChatCompletionsVLM("https://api.sarvam.ai/v1", "my-sarvam-vlm", "key", client=client)
    report = analyze_photos([Photo("a", garment((240, 120, 170)))], DocAiLLMExtractor(digitizer, vlm), "Ramesh")

    body = seen["submit"].content
    assert b'name="language"' in body and b"hi-IN" in body
    assert b'name="output_format"' in body and b"html" in body
    assert b'name="file"' in body
    assert seen["submit"].headers.get("api-subscription-key") == "test-key"
    assert seen["submit"].headers.get("idempotency-key")
    parts = seen["chat"]["messages"][0]["content"]
    assert "GSM 180" in parts[0]["text"] and "Cotton" in parts[0]["text"]
    assert report.backend == "doc-ai+llm"
    assert report.suggested["subcategory"] == "kurti"


def test_doc_ai_failure_still_answers_via_pixel_checks():
    from vyapar_ai.vision import ChatCompletionsVLM, DocAiDigitizer, DocAiLLMExtractor

    def handler(request):
        if "/doc-ai/" in str(request.url):
            return httpx.Response(500, text="boom")
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ext = FallbackExtractor(DocAiLLMExtractor(DocAiDigitizer("k", client=client, sleep=lambda s: None),
                                              ChatCompletionsVLM("https://x/v1", "m", "k", client=client)))
    report = analyze_photos([Photo("a", garment((30, 60, 180)))], ext)
    assert report.backend.startswith("pixel-checks")
    assert report.suggested["color"] == "blue"


def test_non_product_photo_fills_nothing_and_warns():
    out = dict(MODEL_JSON, is_product_photo=False, suggested_title="unknown")
    report = analyze_photos([Photo("a", garment((240, 120, 170)))], fake_vlm(json.dumps(out)))
    s = report.suggested
    assert s["is_product_photo"] is False
    assert (s["subcategory"], s["fabric"], s["pattern"], s["color"], s["title"]) == (None,) * 5
    assert s["has_size_chart"] is False and s["has_fabric_card"] is False
    assert report.missing_shots == []
    assert report.nudges[0]["type"] == "not_product"


def test_placeholder_titles_are_dropped():
    for t in ("unknown", "Unknown", " N/A ", ""):
        assert vision._title(t) is None
    assert vision._title("Pink printed cotton kurti") == "Pink printed cotton kurti"
