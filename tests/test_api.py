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
