import pytest

from vyapar_ai.engine import PricingEngine, SellerProduct


@pytest.fixture(scope="module")
def engine():
    return PricingEngine()


def test_kurti_day0(engine):
    res = engine.day0(SellerProduct(
        title="Pink printed cotton kurti", subcategory="kurti", cogs=150, labor=20,
        packaging=10, target_profit=40, fabric="cotton", pattern="printed", color="pink"))
    rec = res.recommendation
    assert rec["recommended_price"] >= rec["floor_price"]
    assert 230 < rec["recommended_price"] < 360
    assert res.price_spectrum["loss_zone_below"] < rec["floor_price"]
    assert res.nudges[0]["type"] == "launch_price"
    assert any(n["type"] == "lever_nudge" for n in res.nudges)


def test_silk_saree_day0(engine):
    res = engine.day0(SellerProduct(
        title="Purple woven banarasi silk saree", subcategory="banarasi_silk_saree",
        cogs=4200, labor=800, packaging=80, target_profit=1200, weight_kg=0.9,
        fabric="silk", pattern="woven", color="purple", c_damage=350))
    assert res.recommendation["recommended_price"] >= res.recommendation["floor_price"] > 6000
