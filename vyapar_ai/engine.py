"""Day-0 pipeline orchestration (Slide 3 flow / Slide 6 pipeline).

seller input -> embedding -> similar listings + competitor band -> risk model
-> P0 floor -> launch price + guardrails -> Hinglish nudges
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import nudges
from .catalog import category_of, generate_catalog
from .decision import (DecisionModel, ambiguity_multiplier, default_decision_model,
                       prioritise_levers)
from .pricing import CostInputs, PriceBand, break_even_price, recommend_launch_price
from .retrieval import SimilarityIndex
from .risk import RiskFeatures, RiskModel


def forward_shipping(weight_kg: float) -> float:
    """Approximate weight-slab forward shipping (₹). Slide 2: 0.49kg ₹40 -> 0.82kg ₹70."""
    if weight_kg <= 0.5:
        return 45.0
    if weight_kg <= 1.0:
        return 70.0
    return 70.0 + 40.0 * (weight_kg - 1.0)


@dataclass
class SellerProduct:
    title: str
    subcategory: str
    cogs: float
    labor: float = 0.0
    packaging: float = 0.0
    target_profit: float = 0.0
    platform_fees: float = 0.0
    fabric: str | None = None
    pattern: str | None = None
    color: str | None = None
    weight_kg: float = 0.4
    tax_rate: float = 0.07
    cod_share: float = 0.75
    has_size_chart: bool = False
    has_fabric_card: bool = False
    image_count: int = 3
    seller_name: str = "Seller"
    description: str = ""
    return_comments: list[str] = field(default_factory=list)  # buyer comments on similar/past listings
    c_rto: float | None = None       # override; default = two-way shipping
    c_damage: float | None = None    # override; default = reverse ship + repack + wear


@dataclass
class Day0Result:
    recommendation: dict
    market: dict
    risk: dict
    cost_breakdown: dict
    price_spectrum: dict
    levers: list[dict] = field(default_factory=list)
    nudges: list[dict] = field(default_factory=list)
    decisions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class PricingEngine:
    def __init__(self, catalog=None, seed: int = 7,
                 decision_model: DecisionModel | None = None):
        self.catalog = catalog or generate_catalog(3000, seed=seed)
        self.index = SimilarityIndex(self.catalog)
        self.risk = RiskModel().fit(self.catalog)
        self.decider = decision_model or default_decision_model()

    def day0(self, p: SellerProduct) -> Day0Result:
        category = category_of(p.subcategory)
        snap = self.index.market_snapshot(p.title, p.subcategory, category,
                                          p.fabric, p.pattern, p.color)
        feats = RiskFeatures(subcategory=p.subcategory, category=category,
                             rel_price=1.0, cod_share=p.cod_share,
                             has_size_chart=p.has_size_chart,
                             has_fabric_card=p.has_fabric_card,
                             image_count=p.image_count)
        r_ret_model, r_rto = self.risk.predict(feats)
        questions = ["listing_ambiguity"] + (["return_reason"] if p.return_comments else [])
        bundle = self.decider.decide({
            "title": p.title, "description": p.description, "subcategory": p.subcategory,
            "fabric": p.fabric, "pattern": p.pattern, "color": p.color,
            "has_size_chart": p.has_size_chart, "has_fabric_card": p.has_fabric_card,
            "image_count": p.image_count, "return_comments": p.return_comments[:50],
        }, questions)
        amb = bundle.decisions.get("listing_ambiguity")
        r_ret = min(0.9, r_ret_model * ambiguity_multiplier(amb.value)) if amb else r_ret_model
        ship = forward_shipping(p.weight_kg)
        c_rto = p.c_rto if p.c_rto is not None else 2 * ship
        c_damage = (p.c_damage if p.c_damage is not None
                    else ship + p.packaging + 0.03 * p.cogs)
        costs = CostInputs(cogs=p.cogs, labor=p.labor, packaging=p.packaging,
                           target_profit=p.target_profit, platform_fees=p.platform_fees,
                           r_rto=r_rto, c_rto=c_rto,
                           r_ret=r_ret, c_damage=c_damage, tax_rate=p.tax_rate)
        band = PriceBand(snap.band_low, snap.band_high, snap.median_price)
        rec = recommend_launch_price(costs, band)
        levers = prioritise_levers(self.risk.lever_impacts(feats),
                                   bundle.decisions.get("return_reason"))

        cards = [nudges.launch_card(p.seller_name, p.subcategory.replace("_", " "),
                                    rec.recommended_price, (band.low, band.high),
                                    rec.expected_unit_margin, r_ret)]
        cards += [nudges.lever_card(l, p.subcategory.replace("_", " ")) for l in levers[:2]]

        return Day0Result(
            recommendation=rec.to_dict(),
            market=snap.to_dict(),
            risk={"predicted_return_rate": round(r_ret, 4),
                  "model_return_rate": round(r_ret_model, 4),
                  "predicted_rto_rate": round(r_rto, 4),
                  "category_return_rate": round(snap.mean_return_rate, 4),
                  "category_rto_rate": round(snap.mean_rto_rate, 4)},
            cost_breakdown={
                "cogs": p.cogs, "labor": p.labor, "packaging": p.packaging,
                "platform_fees": p.platform_fees,
                "expected_rto_cost": round(r_rto * c_rto, 2),
                "expected_return_cost": round(r_ret * c_damage, 2),
                "target_profit": p.target_profit,
                "tax_rate": p.tax_rate,
                "c_rto": c_rto, "c_damage": round(c_damage, 2),
            },
            price_spectrum={
                "loss_zone_below": round(break_even_price(costs), 2),
                "floor_p0": rec.floor_price,
                "sweet_spot": rec.recommended_price,
                "overpriced_above": round(band.high, 2),
            },
            levers=levers,
            nudges=cards,
            decisions=bundle.to_dict(),
        )
