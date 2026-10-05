import pytest

from vyapar_ai.pricing import (CostInputs, PriceBand, charm_round, floor_price,
                               recommend_launch_price, unit_margin)

KURTI = CostInputs(cogs=150, labor=20, packaging=10, target_profit=40,
                   r_rto=0.21, c_rto=90, tax_rate=0.07)
SAREE = CostInputs(cogs=4200, labor=800, packaging=80, target_profit=1200,
                   r_ret=0.28, c_damage=350, tax_rate=0.07)


def test_kurti_floor_matches_slide_3():
    assert floor_price(KURTI) == pytest.approx(256.88, abs=0.01)


def test_saree_floor_matches_slide_3():
    assert floor_price(SAREE) == pytest.approx(6858.06, abs=0.01)


def test_kurti_launch_price_matches_slide_3():
    rec = recommend_launch_price(KURTI, PriceBand(low=250, high=299, median=279))
    assert rec.recommended_price == 269
    assert rec.position == "within_band"
    assert rec.recommended_price >= rec.floor_price


def test_margin_at_floor_equals_target_profit():
    assert unit_margin(floor_price(KURTI), KURTI) == pytest.approx(40)


def test_never_below_floor_even_if_band_is_cheaper():
    rec = recommend_launch_price(KURTI, PriceBand(low=150, high=220, median=199))
    assert rec.recommended_price >= rec.floor_price
    assert rec.position == "above_band"
    assert rec.warnings


def test_charm_round():
    assert charm_round(264.6) == 269
    assert charm_round(269) == 269
    assert charm_round(269.5) == 279
    assert charm_round(7064) == 7099


def test_invalid_inputs():
    with pytest.raises(ValueError):
        floor_price(CostInputs(cogs=100, r_rto=1.5))
