from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_index_and_subcategories():
    assert client.get("/").status_code == 200
    assert "kurti" in client.get("/api/subcategories").json()["women_ethnic"]


def test_day0_endpoint():
    r = client.post("/api/day0", json={"title": "Pink printed cotton kurti", "subcategory": "kurti",
                                       "cogs": 150, "labor": 20, "packaging": 10,
                                       "target_profit": 40, "seller_name": "Ramesh"})
    assert r.status_code == 200
    d = r.json()
    assert d["recommendation"]["recommended_price"] >= d["recommendation"]["floor_price"]
    assert d["nudges"][0]["text"].startswith("Namaste Ramesh ji")


def test_day0_rejects_unknown_subcategory():
    r = client.post("/api/day0", json={"title": "x", "subcategory": "rocket", "cogs": 1})
    assert r.status_code == 422


def test_lifecycle_endpoint_defense_with_competitor_alert():
    r = client.post("/api/lifecycle", json={
        "seller_name": "Ramesh", "current_price": 289, "cogs": 150, "labor": 20, "packaging": 10,
        "target_profit": 20, "competitor_price": 269,
        "signals": {"day": 100, "reviews": 80, "rating": 4.2, "weekly_velocity": 15}})
    d = r.json()
    assert r.status_code == 200 and d["stage"] == "defense"
    assert d["bundles"] and any(n["type"] == "competitor_alert" for n in d["nudges"])
    assert d["suggested_price"] >= d["floor_p0"]


KURTI = {"title": "Pink printed cotton kurti", "subcategory": "kurti", "cogs": 150,
         "labor": 20, "packaging": 10, "target_profit": 40, "seller_name": "Ramesh"}


def test_nudge_actions_are_structured():
    d = client.post("/api/day0", json=KURTI).json()
    acts = d["nudges"][0]["actions"]
    assert [a["id"] for a in acts] == ["accept_price", "keep_price", "explain"]
    assert acts[0]["price"] == d["recommendation"]["recommended_price"]
    assert d["explanation"].startswith("Hisaab aise bana")


def test_price_check_verdicts():
    d = client.post("/api/day0", json=KURTI).json()
    rec = d["recommendation"]
    def verdict(p):
        return client.post("/api/price-check", json={"product": KURTI, "price": p}).json()
    assert verdict(int(rec["break_even"]) - 20)["verdict"] == "loss"
    assert verdict(int(rec["break_even"]) - 20)["unit_margin"] < 0
    assert verdict(int(rec["floor_price"]) - 2)["verdict"] == "below_target"
    ok = verdict(rec["recommended_price"])
    assert ok["verdict"] == "ok"
    assert abs(ok["unit_margin"] - rec["expected_unit_margin"]) < 0.01
    assert verdict(int(d["market"]["band_high"]) + 50)["verdict"] == "overpriced"


def test_create_and_list_listing():
    r = client.post("/api/listings", json={"product": KURTI, "price": 289})
    assert r.status_code == 201
    body = r.json()
    assert body["listing"]["price"] == 289 and body["message"].startswith("Ho gaya!")
    ids = [l["listing_id"] for l in client.get("/api/listings").json()["listings"]]
    assert body["listing"]["listing_id"] in ids


def test_health_routes_do_not_build_engine(monkeypatch):
    import api.main as m
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    m.engine.cache_clear()
    assert client.get("/health").json() == {"status": "ok"}
    h = client.get("/api/health").json()
    assert h == {"status": "ok", "engine_ready": False, "decision_backend": None}
    assert m.engine.cache_info().currsize == 0
    client.post("/api/day0", json=KURTI)
    h = client.get("/api/health").json()
    assert h["engine_ready"] is True and h["decision_backend"] == "rules"
