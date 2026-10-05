"""Day-0 pricing math (Slide 3).

    P0 = (COGS + Labor + Packaging + Fees + r_rto*C_rto + r_ret*C_damage + TargetProfit)
         / (1 - (GST rate + TCS/TDS rate))

P0 is the break-even-plus-target floor: the engine never recommends below it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict


@dataclass
class CostInputs:
    cogs: float
    labor: float = 0.0
    packaging: float = 0.0
    target_profit: float = 0.0
    platform_fees: float = 0.0  # fixed per-order fees charged to the seller (e.g. shipping)
    r_rto: float = 0.0        # predicted RTO (delivery refusal) probability
    c_rto: float = 0.0        # two-way reverse shipping penalty per RTO
    r_ret: float = 0.0        # predicted customer-return probability
    c_damage: float = 0.0     # re-packaging / damage cost per return
    tax_rate: float = 0.07    # GST + TCS/TDS withheld, as a fraction of price

    def validate(self) -> None:
        for name in ("cogs", "labor", "packaging", "platform_fees", "c_rto", "c_damage"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("r_rto", "r_ret"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if not 0 <= self.tax_rate < 1:
            raise ValueError("tax_rate must be within [0, 1)")


def expected_risk_cost(c: CostInputs) -> float:
    return c.r_rto * c.c_rto + c.r_ret * c.c_damage


def fixed_unit_cost(c: CostInputs) -> float:
    return c.cogs + c.labor + c.packaging + c.platform_fees


def floor_price(c: CostInputs) -> float:
    """The 0%-commission Day-0 floor P0."""
    c.validate()
    numerator = fixed_unit_cost(c) + expected_risk_cost(c) + c.target_profit
    return numerator / (1 - c.tax_rate)


def break_even_price(c: CostInputs) -> float:
    """P0 with zero target profit: below this every order loses money."""
    c.validate()
    return (fixed_unit_cost(c) + expected_risk_cost(c)) / (1 - c.tax_rate)


def unit_margin(price: float, c: CostInputs) -> float:
    """Expected net cash per shipped order at `price`, after taxes and risk."""
    return price * (1 - c.tax_rate) - fixed_unit_cost(c) - expected_risk_cost(c)


def charm_round(price: float) -> int:
    """Round *up* to a marketplace-style price (…9 below ₹1000, …99 above)."""
    if price < 1000:
        return int(math.ceil((price + 1) / 10) * 10 - 1)
    return int(math.ceil((price + 1) / 100) * 100 - 1)


@dataclass
class PriceBand:
    low: float
    high: float
    median: float


@dataclass
class Recommendation:
    floor_price: float
    break_even: float
    recommended_price: int
    safe_band: tuple[int, int]
    expected_unit_margin: float
    position: str           # "within_band" | "above_band" | "below_band"
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


LAUNCH_WEDGE = 0.03  # Slide 4: launch at P0 + 3%


def recommend_launch_price(c: CostInputs, band: PriceBand | None = None) -> Recommendation:
    """Combine the P0 floor with the competitor band into a Day-0 price.

    Guardrails (Slide 4/6): P >= P0 always; prefer P <= band.high so the
    listing stays competitive in search.
    """
    p0 = floor_price(c)
    be = break_even_price(c)
    target = charm_round(p0 * (1 + LAUNCH_WEDGE))
    warnings: list[str] = []
    position = "within_band"

    if band is not None:
        if target < band.low:
            # Room to price up to the bottom of the market band.
            target = charm_round(band.low)
        if target > band.high:
            if charm_round(p0) <= band.high:
                target = int(math.floor(band.high))
            else:
                position = "above_band"
                warnings.append(
                    f"Your floor ₹{p0:.0f} is above the competitor band "
                    f"(₹{band.low:.0f}–₹{band.high:.0f}). Reduce costs or "
                    "differentiate the listing (fabric/GSM card, better photos).")

    target = max(target, int(math.ceil(p0)))  # hard guardrail P >= P0
    lo = int(math.ceil(p0))
    hi = int(max(target, band.high if band else target))
    return Recommendation(
        floor_price=round(p0, 2),
        break_even=round(be, 2),
        recommended_price=target,
        safe_band=(lo, hi),
        expected_unit_margin=round(unit_margin(target, c), 2),
        position=position,
        warnings=warnings,
    )
