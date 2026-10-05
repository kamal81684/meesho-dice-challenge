import pytest

from vyapar_ai.decision import (FallbackDecisionModel, JevDecisionModel, RuleDecisionModel,
                                ambiguity_multiplier, default_decision_model,
                                prioritise_levers)
from vyapar_ai.engine import PricingEngine, SellerProduct

typesafe_sdk = pytest.importorskip("typesafe_sdk")


class FakeJevClient:
    """Mimics TypeSafeClient.system_one with canned typed answers."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return typesafe_sdk.SystemOneResponse.model_validate({
            "model": "jev-latest", "usage": {"input_tokens": 10, "output_tokens": 0},
            "answers": {k: v for k, v in self.answers.items() if k in questions}})


class BrokenClient:
    def system_one(self, state, questions):
        raise ConnectionError("network down")


JEV_ANSWERS = {
    "listing_ambiguity": {"type": "score", "score": 4.0, "confidence": 0.82,
                          "legend": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"},
                          "probabilities": {3: 0.2, 4: 0.8}},
    "return_reason": {"type": "choice", "choice": "fabric_quality", "confidence": 0.77,
                      "probabilities": {"fabric_quality": 0.77, "size_mismatch": 0.13,
                                        "color_mismatch": 0.05, "damaged": 0.03,
                                        "changed_mind": 0.02}},
    "lifecycle_stage": {"type": "choice", "choice": "defense", "confidence": 0.9,
                        "probabilities": {"defense": 0.9, "scale": 0.1}},
}


def test_jev_questions_are_typed_and_sent_in_one_call():
    client = FakeJevClient(JEV_ANSWERS)
    bundle = JevDecisionModel(client=client).decide(
        {"title": "kurti"}, ["listing_ambiguity", "return_reason"])
    assert len(client.calls) == 1
    qs = client.calls[0][1]
    assert isinstance(qs["listing_ambiguity"], typesafe_sdk.Score)
    assert isinstance(qs["return_reason"], typesafe_sdk.Choice)
    assert bundle.backend == "jev"
    assert bundle.decisions["listing_ambiguity"].value == 5.0   # 0..4 legend -> 1..5
    assert bundle.decisions["return_reason"].value == "fabric_quality"


def test_fallback_when_jev_fails():
    model = FallbackDecisionModel(JevDecisionModel(client=BrokenClient()))
    bundle = model.decide({"title": "kurti", "description": ""}, ["listing_ambiguity"])
    assert bundle.backend.startswith("rules")
    assert "ConnectionError" in model.last_error


def test_rules_return_reason_from_hinglish_comments():
    d = RuleDecisionModel().decide({"return_comments": [
        "size chhota hai", "fit nahi hua, too tight", "kapda patla hai"]}, ["return_reason"])
    assert d.decisions["return_reason"].value == "size_mismatch"


def test_default_backend_without_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert default_decision_model().name == "rules"


def test_ambiguity_multiplier_and_lever_priority():
    assert ambiguity_multiplier(1) < 1 < ambiguity_multiplier(5)
    levers = [{"lever": "add_size_chart", "relative_drop": 0.18},
              {"lever": "add_fabric_card", "relative_drop": 0.12}]
    bundle = JevDecisionModel(client=FakeJevClient(JEV_ANSWERS)).decide({}, ["return_reason"])
    assert prioritise_levers(levers, bundle.decisions["return_reason"])[0]["lever"] == "add_fabric_card"


def test_engine_uses_jev_but_keeps_floor_guardrail():
    engine = PricingEngine(decision_model=JevDecisionModel(client=FakeJevClient(JEV_ANSWERS)))
    res = engine.day0(SellerProduct(
        title="Pink printed cotton kurti", subcategory="kurti", cogs=150, labor=20,
        packaging=10, target_profit=40, return_comments=["kapda bahut patla hai"]))
    assert res.decisions["backend"] == "jev"
    assert res.risk["predicted_return_rate"] > res.risk["model_return_rate"]  # ambiguous listing
    assert res.levers[0]["lever"] == "add_fabric_card"
    assert res.recommendation["recommended_price"] >= res.recommendation["floor_price"]


def test_lifecycle_endpoint_uses_confident_jev_stage(monkeypatch):
    from fastapi.testclient import TestClient
    import api.main as m
    jev_engine = PricingEngine(decision_model=JevDecisionModel(client=FakeJevClient(JEV_ANSWERS)))
    monkeypatch.setattr(m, "engine", lambda: jev_engine)
    r = TestClient(m.app).post("/api/lifecycle", json={
        "current_price": 299, "cogs": 150, "labor": 20, "packaging": 10,
        "recent_reviews": ["same kurti cheaper elsewhere"],
        "signals": {"day": 40, "reviews": 30, "rating": 4.3, "weekly_velocity": 20}})
    d = r.json()
    assert d["rule_stage"] == "scale"
    assert d["stage"] == "defense" and d["stage_source"] == "jev"
    assert d["suggested_price"] >= d["floor_p0"]
