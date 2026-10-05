"""Surat cotton suit backtest (Slide 8): manual trial-and-error vs Vyapar-AI.

Both sellers face the same simulated market (same seed). The manual seller
prices by offline markup at ₹349, has no fabric card (15% returns) and only
moves to ₹399 on day 120. Vyapar-AI launches from the P0 floor, adds the
fabric/GSM card on Day 0 (8% returns) and lets LinUCB climb inside the band.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vyapar_ai.pricing import CostInputs, PriceBand, recommend_launch_price  # noqa: E402
from vyapar_ai.simulator import MarketParams, run_fixed_schedule, run_vyapar  # noqa: E402

DAYS = 150
COSTS = CostInputs(cogs=220, labor=10, packaging=10, platform_fees=35, target_profit=40,
                   r_rto=0.15, c_rto=110, r_ret=0.08, c_damage=120, tax_rate=0.07)
MARKET = MarketParams(ref_price=385, pmax=399, base_daily_orders=4.0, elasticity=4.3,
                      true_rto=0.15)


def main(seeds=range(20)):
    rec = recommend_launch_price(COSTS, PriceBand(370, 399, 385))
    manual_runs, vyapar_runs = [], []
    for seed in seeds:
        manual_runs.append(run_fixed_schedule(
            MARKET, COSTS, schedule=lambda d: 349 if d < 120 else 399,
            return_rate=lambda d: 0.15, days=DAYS, seed=seed).summary())
        vyapar_runs.append(run_vyapar(
            MARKET, COSTS, launch_price=rec.recommended_price, return_rate=0.08,
            days=DAYS, seed=seed).summary())

    def avg(runs, key):
        vals = [r[key] for r in runs if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    keys = ["launch_price", "final_price", "day_reached_final_price",
            "day_cash_positive_for_good", "profit_first_100_orders",
            "worst_cumulative_drawdown", "cumulative_profit", "total_orders",
            "realised_return_rate"]
    report = {
        "floor_p0": rec.floor_price,
        "recommended_launch_price": rec.recommended_price,
        "seeds": len(list(seeds)),
        "manual": {k: avg(manual_runs, k) for k in keys},
        "vyapar_ai": {k: avg(vyapar_runs, k) for k in keys},
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
