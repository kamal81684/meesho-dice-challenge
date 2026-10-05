"""FastAPI microservice for Vyapar-AI (Slide 6: "Real-time FastAPI microservice")."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from vyapar_ai import nudges
from vyapar_ai.catalog import TAXONOMY
from vyapar_ai.engine import PricingEngine, SellerProduct, forward_shipping
from vyapar_ai.lifecycle import Signals, determine_stage, stage_action
from vyapar_ai.pricing import CostInputs, break_even_price, floor_price, unit_margin

WEB = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="Vyapar-AI", version="0.1.0",
              description="Day-0 and lifecycle pricing for new-to-online sellers")


@lru_cache(maxsize=1)
def engine() -> PricingEngine:
    return PricingEngine()


class Day0Request(BaseModel):
    title: str
    subcategory: str
    cogs: float = Field(ge=0)
    labor: float = Field(0, ge=0)
    packaging: float = Field(0, ge=0)
    target_profit: float = Field(0, ge=0)
    platform_fees: float = Field(0, ge=0)
    fabric: str | None = None
    pattern: str | None = None
    color: str | None = None
    weight_kg: float = Field(0.4, gt=0)
    tax_rate: float = Field(0.07, ge=0, lt=1)
    cod_share: float = Field(0.75, ge=0, le=1)
    has_size_chart: bool = False
    has_fabric_card: bool = False
    image_count: int = Field(3, ge=0)
    seller_name: str = "Seller"
    c_rto: float | None = Field(None, ge=0)
    c_damage: float | None = Field(None, ge=0)


class LifecycleRequest(BaseModel):
    seller_name: str = "Seller"
    current_price: int = Field(gt=0)
    cogs: float = Field(ge=0)
    labor: float = 0
    packaging: float = 0
    platform_fees: float = 0
    target_profit: float = 0
    tax_rate: float = 0.07
    r_rto: float = Field(0.18, ge=0, le=1)
    r_ret: float = Field(0.15, ge=0, le=1)
    weight_kg: float = 0.4
    competitor_price: float | None = None
    signals: Signals


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/subcategories")
def subcategories():
    return {cat: list(subs) for cat, subs in TAXONOMY.items()}


@app.post("/api/day0")
def day0(req: Day0Request):
    valid = {s for subs in TAXONOMY.values() for s in subs}
    if req.subcategory not in valid:
        raise HTTPException(422, f"subcategory must be one of {sorted(valid)}")
    return engine().day0(SellerProduct(**req.model_dump())).to_dict()


@app.post("/api/lifecycle")
def lifecycle(req: LifecycleRequest):
    ship = forward_shipping(req.weight_kg)
    costs = CostInputs(cogs=req.cogs, labor=req.labor, packaging=req.packaging,
                       platform_fees=req.platform_fees, target_profit=req.target_profit,
                       r_rto=req.r_rto, c_rto=2 * ship, r_ret=req.r_ret,
                       c_damage=ship + req.packaging + 0.03 * req.cogs, tax_rate=req.tax_rate)
    p0, be = floor_price(costs), break_even_price(costs)
    stage = determine_stage(req.signals)
    bandit_price = None
    if stage.value == "scale":
        # Stateless endpoint: suggest the conservative +₹10 nudge; the
        # stateful LinUCB loop runs in the batch/simulation path.
        bandit_price = req.current_price + 10
    act = stage_action(stage, req.current_price, p0, be, ship, req.packaging, bandit_price)
    out = {"stage": stage.value, "floor_p0": round(p0, 2), "break_even": round(be, 2),
           "suggested_price": act.price, "action": act.action, "bundles": act.bundles,
           "unit_margin_at_suggested": round(unit_margin(act.price, costs), 2),
           "nudges": [nudges.stage_card(req.seller_name, stage.value, act.action, act.price)]}
    if req.competitor_price and req.competitor_price < req.current_price:
        drop = req.current_price - req.competitor_price
        match = max(int(p0 + 0.5), int(req.competitor_price) - 1)
        if match < req.current_price:
            uplift = (req.current_price / match) ** 4.3 - 1
            out["nudges"].append(nudges.competitor_alert(
                req.seller_name, "product", drop, match, uplift, unit_margin(match, costs)))
    return out
