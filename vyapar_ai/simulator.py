"""Stochastic marketplace simulator used to backtest pricing policies.

Per day: orders ~ Poisson(λ(price, reviews)); some orders are RTO'd, some
delivered orders are returned; cash flow is tallied per order outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bandit import LinUCB, context_vector
from .lifecycle import Signals, Stage, determine_stage
from .pricing import CostInputs, floor_price


@dataclass
class MarketParams:
    ref_price: float            # market median for the cluster
    pmax: float                 # top of competitor band (search drop-off above this)
    base_daily_orders: float = 4.0
    elasticity: float = 4.3
    true_rto: float = 0.18
    true_return: float = 0.15
    rating: float = 4.2


@dataclass
class DayLog:
    day: int
    price: int
    stage: str
    orders: int
    rto: int
    returns: int
    cash: float


@dataclass
class SimResult:
    logs: list[DayLog] = field(default_factory=list)

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum([d.cash for d in self.logs])

    def summary(self) -> dict:
        cum = self.cumulative
        orders = np.cumsum([d.orders for d in self.logs])
        positive = np.flatnonzero(cum > 0)
        first_100 = int(np.searchsorted(orders, 100))
        profit_first_100 = float(cum[min(first_100, len(cum) - 1)])
        # Profitability = cumulative cash stays positive from that day onward.
        neg = np.flatnonzero(cum <= 0)
        day_profitable = None if (len(neg) and neg[-1] == len(cum) - 1) else (
            int(self.logs[neg[-1] + 1].day) if len(neg) else int(self.logs[0].day))
        final_price = self.logs[-1].price
        top = [d.day for d in self.logs if d.price >= final_price]
        total_orders = int(orders[-1])
        return {
            "total_orders": total_orders,
            "cumulative_profit": round(float(cum[-1]), 2),
            "profit_first_100_orders": round(profit_first_100, 2),
            "worst_cumulative_drawdown": round(float(min(0.0, cum.min())), 2),
            "day_cash_positive_for_good": day_profitable,
            "first_day_cash_positive": int(self.logs[positive[0]].day) if len(positive) else None,
            "launch_price": self.logs[0].price,
            "final_price": final_price,
            "day_reached_final_price": top[0] if top else None,
            "realised_return_rate": round(sum(d.returns for d in self.logs)
                                          / max(1, sum(d.orders - d.rto for d in self.logs)), 4),
        }


class Market:
    def __init__(self, params: MarketParams, costs: CostInputs, seed: int = 0):
        self.p = params
        self.c = costs
        self.rng = np.random.default_rng(seed)
        self.reviews = 0

    def demand(self, price: float) -> float:
        discovery = 0.35 + 0.65 * min(1.0, self.reviews / 25)
        lam = self.p.base_daily_orders * (price / self.p.ref_price) ** (-self.p.elasticity)
        if price > self.p.pmax:
            lam *= 0.5   # falls out of the top search slots
        return lam * discovery

    def step(self, price: float, return_rate: float) -> tuple[int, int, int, float]:
        c = self.c
        n = int(self.rng.poisson(self.demand(price)))
        rto = int(self.rng.binomial(n, self.p.true_rto))
        delivered = n - rto
        ret = int(self.rng.binomial(delivered, return_rate))
        kept = delivered - ret
        self.reviews += int(self.rng.binomial(kept, 0.3))
        unit_cost = c.cogs + c.labor + c.packaging + c.platform_fees
        cash = (kept * (price * (1 - c.tax_rate) - unit_cost)
                - rto * (c.c_rto + c.packaging)    # product comes back unsold
                - ret * (c.c_damage + c.packaging + c.platform_fees))
        return n, rto, ret, float(cash)


def run_fixed_schedule(params: MarketParams, costs: CostInputs, schedule, return_rate,
                       days: int, seed: int = 0) -> SimResult:
    """`schedule(day) -> price`, `return_rate(day) -> rate`."""
    m = Market(params, costs, seed)
    res = SimResult()
    for day in range(1, days + 1):
        price = int(schedule(day))
        n, rto, ret, cash = m.step(price, return_rate(day))
        res.logs.append(DayLog(day, price, "manual", n, rto, ret, round(cash, 2)))
    return res


def run_vyapar(params: MarketParams, costs: CostInputs, launch_price: int,
               return_rate: float, days: int, seed: int = 0,
               cycle_days: int = 1) -> SimResult:
    """Lifecycle state machine + LinUCB during the Scale stage."""
    m = Market(params, costs, seed)
    p0 = floor_price(costs)
    bandit = LinUCB(n_features=6, seed=seed)
    res = SimResult()
    price = launch_price
    pending: tuple[int, np.ndarray] | None = None
    for day in range(1, days + 1):
        sig = Signals(day=day, reviews=m.reviews, rating=params.rating,
                      weekly_velocity=sum(d.orders for d in res.logs[-7:]),
                      inventory_age_days=0)
        stage = determine_stage(sig)
        if stage == Stage.SCALE and (day % cycle_days == 0):
            x = context_vector(params.rating, params.true_rto, sig.weekly_velocity,
                               min(1.0, m.reviews / 50), price / params.ref_price)
            arm = bandit.select(x, price, p0, params.pmax)
            price = int(price + bandit.arms[arm])
            pending = (arm, x)
        n, rto, ret, cash = m.step(price, return_rate)
        res.logs.append(DayLog(day, price, stage.value, n, rto, ret, round(cash, 2)))
        if pending is not None:
            # Reward on expected (not noisy realised) daily cash to keep learning stable.
            exp_cash = m.demand(price) * (
                (1 - params.true_rto) * (1 - return_rate) * (price * (1 - costs.tax_rate)
                - (costs.cogs + costs.labor + costs.packaging + costs.platform_fees))
                - params.true_rto * (costs.c_rto + costs.packaging)
                - (1 - params.true_rto) * return_rate
                * (costs.c_damage + costs.packaging + costs.platform_fees))
            bandit.update(pending[0], pending[1], exp_cash / 100.0)
            pending = None
    return res
