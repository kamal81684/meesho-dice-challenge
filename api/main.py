"""FastAPI microservice for Vyapar-AI (Slide 6: "Real-time FastAPI microservice")."""
from __future__ import annotations

from typing import Optional

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from vyapar_ai import nudges
from vyapar_ai.catalog import TAXONOMY
from vyapar_ai.engine import PricingEngine, SellerProduct, forward_shipping
from vyapar_ai.lifecycle import Signals, Stage, determine_stage, stage_action
from vyapar_ai.pricing import CostInputs, break_even_price, floor_price, unit_margin

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DOCS = ROOT / "docs"

# Jev's stage call is used only when it is at least this confident;
# otherwise the deterministic state machine decides.
STAGE_CONFIDENCE_THRESHOLD = 0.7

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
    fabric: Optional[str] = None
    pattern: Optional[str] = None
    color: Optional[str] = None
    weight_kg: float = Field(0.4, gt=0)
    tax_rate: float = Field(0.07, ge=0, lt=1)
    cod_share: float = Field(0.75, ge=0, le=1)
    has_size_chart: bool = False
    has_fabric_card: bool = False
    image_count: int = Field(3, ge=0)
    seller_name: str = "Seller"
    description: str = ""
    return_comments: list[str] = Field(default_factory=list)
    c_rto: Optional[float] = Field(None, ge=0)
    c_damage: Optional[float] = Field(None, ge=0)


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
    competitor_price: Optional[float] = None
    recent_reviews: list[str] = Field(default_factory=list)
    seller_notes: str = ""
    signals: Signals


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")


@app.get("/architecture", include_in_schema=False)
def architecture():
    return FileResponse(DOCS / "architecture.html")


def _engine_ready() -> bool:
    return engine.cache_info().currsize > 0


@app.get("/health", include_in_schema=False)
def liveness():
    """Instant liveness check for load balancers (e.g. Render). Never builds the models."""
    return {"status": "ok"}


@app.get("/api/health")
def health():
    """Status without forcing the engine build (which takes ~5 s, longer on small CPUs)."""
    ready = _engine_ready()
    return {"status": "ok", "engine_ready": ready,
            "decision_backend": engine().decider.name if ready else None}


@app.get("/api/subcategories")
def subcategories():
    return {cat: list(subs) for cat, subs in TAXONOMY.items()}


def _run_day0(req: Day0Request) -> dict:
    valid = {s for subs in TAXONOMY.values() for s in subs}
    if req.subcategory not in valid:
        raise HTTPException(422, f"subcategory must be one of {sorted(valid)}")
    return engine().day0(SellerProduct(**req.model_dump())).to_dict()


@app.post("/api/day0")
def day0(req: Day0Request):
    return _run_day0(req)


class PriceRequest(BaseModel):
    product: Day0Request
    price: int = Field(gt=0)


def evaluate_price(d: dict, price: int) -> dict:
    """Margin and verdict for a seller-chosen price, using the Day-0 cost model."""
    c, rec, band_high = d["cost_breakdown"], d["recommendation"], d["market"]["band_high"]
    fixed = c["cogs"] + c["labor"] + c["packaging"] + c.get("platform_fees", 0)
    margin = (price * (1 - c["tax_rate"]) - fixed
              - c["expected_rto_cost"] - c["expected_return_cost"])
    if price < rec["break_even"]:
        verdict = "loss"
    elif price < rec["floor_price"]:
        verdict = "below_target"
    elif price > band_high:
        verdict = "overpriced"
    else:
        verdict = "ok"
    return {"price": price, "verdict": verdict, "unit_margin": round(margin, 2),
            "card": nudges.price_check_card(price, verdict, margin, rec["floor_price"],
                                            rec["break_even"], band_high,
                                            rec["recommended_price"])}


@app.post("/api/price-check")
def price_check(req: PriceRequest):
    return evaluate_price(_run_day0(req.product), req.price)


# In-memory store for the MVP; a real deployment writes to the catalog service.
LISTINGS: list[dict] = []


@app.post("/api/listings", status_code=201)
def create_listing(req: PriceRequest):
    check = evaluate_price(_run_day0(req.product), req.price)
    listing = {"listing_id": f"VY{len(LISTINGS) + 1:05d}", "title": req.product.title,
               "subcategory": req.product.subcategory, "price": req.price,
               "verdict": check["verdict"], "unit_margin": check["unit_margin"]}
    LISTINGS.append(listing)
    msg = (f"Ho gaya! '{req.product.title}' {nudges.rs(req.price)} par list ho gaya "
           f"(ID {listing['listing_id']}). Har order par lagbhag "
           f"{nudges.rs(check['unit_margin'])} munafa.")
    if check["verdict"] == "loss":
        msg += " Dhyaan dein: is daam par har order par nuksaan hoga."
    return {"listing": listing, "message": msg}


@app.get("/api/listings")
def list_listings():
    return {"listings": LISTINGS}


@app.post("/api/lifecycle")
def lifecycle(req: LifecycleRequest):
    ship = forward_shipping(req.weight_kg)
    costs = CostInputs(cogs=req.cogs, labor=req.labor, packaging=req.packaging,
                       platform_fees=req.platform_fees, target_profit=req.target_profit,
                       r_rto=req.r_rto, c_rto=2 * ship, r_ret=req.r_ret,
                       c_damage=ship + req.packaging + 0.03 * req.cogs, tax_rate=req.tax_rate)
    p0, be = floor_price(costs), break_even_price(costs)
    rule_stage = determine_stage(req.signals)
    stage, stage_source = rule_stage, "state_machine"
    bundle = engine().decider.decide(
        {"signals": req.signals.__dict__, "recent_reviews": req.recent_reviews[:50],
         "seller_notes": req.seller_notes, "return_comments": req.recent_reviews[:50]},
        ["lifecycle_stage"])
    d = bundle.decisions.get("lifecycle_stage")
    if (d is not None and bundle.backend == "jev"
            and d.confidence >= STAGE_CONFIDENCE_THRESHOLD
            and d.value in {s.value for s in Stage}):
        stage, stage_source = Stage(d.value), "jev"
    bandit_price = None
    if stage.value == "scale":
        # Stateless endpoint: suggest the conservative +₹10 nudge; the
        # stateful LinUCB loop runs in the batch/simulation path.
        bandit_price = req.current_price + 10
    act = stage_action(stage, req.current_price, p0, be, ship, req.packaging, bandit_price)
    out = {"stage": stage.value, "stage_source": stage_source,
           "rule_stage": rule_stage.value, "decisions": bundle.to_dict(),
"floor_p0": round(p0, 2), "break_even": round(be, 2),
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
