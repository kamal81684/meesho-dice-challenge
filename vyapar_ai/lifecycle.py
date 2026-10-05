"""Lifecycle state machine (Slide 4).

Launch & Discovery (days 1–21)  -> launch wedge at P0 + 3%
Scale & Growth   (days 22–90)   -> LinUCB upward nudges (+₹10 / +₹25)
Defense & Maturity (days 91–180)-> hold price, suggest 2/3-packs
Decay & Salvage  (181+ or stale)-> P0 minus logistics buffer to free up cash
"""
from __future__ import annotations

from typing import Optional

from dataclasses import dataclass
from enum import Enum


class Stage(str, Enum):
    LAUNCH = "launch"
    SCALE = "scale"
    DEFENSE = "defense"
    SALVAGE = "salvage"


TARGET_MARGIN = {  # Slide 4 "Target Unit Margin"
    Stage.LAUNCH: (0.03, 0.05),
    Stage.SCALE: (0.15, 0.20),
    Stage.DEFENSE: (0.10, 0.12),
    Stage.SALVAGE: (-0.05, 0.0),
}


@dataclass
class Signals:
    day: int
    reviews: int = 0
    rating: float = 0.0
    ctr: float = 0.0
    conversion: float = 0.0
    competitor_undercut: float = 0.0     # fraction by which the cheapest close rival undercuts us
    weekly_velocity: float = 0.0         # units/week
    inventory_age_days: int = 0
    conversion_trend: float = 0.0        # week-over-week change in conversion


def determine_stage(s: Signals) -> Stage:
    stale = s.weekly_velocity < 2 and s.inventory_age_days > 120
    if stale or (s.day > 180 and s.weekly_velocity < 2):
        return Stage.SALVAGE
    if s.day <= 21 or s.reviews == 0:
        return Stage.LAUNCH
    under_pressure = s.competitor_undercut >= 0.06 or s.conversion_trend <= -0.15
    if s.day > 90 or under_pressure:
        return Stage.DEFENSE
    return Stage.SCALE


@dataclass
class StageAction:
    stage: Stage
    price: int
    action: str
    bundles: Optional[list[dict]] = None


def bundle_offers(price: int, forward_ship: float, packaging: float) -> list[dict]:
    """Virtual multi-packs: pass part of the logistics saving on to the buyer."""
    offers = []
    for n in (2, 3):
        saving = (n - 1) * (forward_ship + packaging)
        offers.append({"pack": n, "price": int(n * price - 0.5 * saving),
                       "seller_logistics_saving": round(saving, 2)})
    return offers


def stage_action(stage: Stage, current_price: int, p0: float, break_even: float,
                 forward_ship: float, packaging: float, bandit_price: Optional[int] = None
                 ) -> StageAction:
    if stage == Stage.LAUNCH:
        return StageAction(stage, max(current_price, int(round(p0 * 1.03))),
                           "Launch wedge: seed the first 25 verified orders")
    if stage == Stage.SCALE:
        price = bandit_price if bandit_price is not None else current_price
        return StageAction(stage, max(price, int(p0 + 0.5)),
                           "Value harvesting: bandit-driven price nudge")
    if stage == Stage.DEFENSE:
        return StageAction(stage, current_price,
                           "Bundle defense: hold price, offer multi-packs",
                           bundle_offers(current_price, forward_ship, packaging))
    salvage = max(p0 - forward_ship, break_even * 0.95)
    return StageAction(stage, int(salvage), "Working-capital salvage: clear dead stock")
