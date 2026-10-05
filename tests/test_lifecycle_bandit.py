import numpy as np

from vyapar_ai.bandit import LinUCB
from vyapar_ai.lifecycle import Signals, Stage, determine_stage, stage_action


def test_stage_transitions():
    assert determine_stage(Signals(day=5)) == Stage.LAUNCH
    assert determine_stage(Signals(day=40, reviews=30, rating=4.3, weekly_velocity=20)) == Stage.SCALE
    assert determine_stage(Signals(day=40, reviews=30, competitor_undercut=0.08,
                                   weekly_velocity=20)) == Stage.DEFENSE
    assert determine_stage(Signals(day=120, reviews=30, weekly_velocity=20)) == Stage.DEFENSE
    assert determine_stage(Signals(day=200, reviews=30, weekly_velocity=1)) == Stage.SALVAGE
    assert determine_stage(Signals(day=60, reviews=30, weekly_velocity=1,
                                   inventory_age_days=130)) == Stage.SALVAGE


def test_defense_offers_bundles_and_salvage_goes_below_p0():
    d = stage_action(Stage.DEFENSE, 299, p0=257, break_even=214, forward_ship=45, packaging=10)
    assert d.price == 299 and [b["pack"] for b in d.bundles] == [2, 3]
    assert d.bundles[0]["price"] < 2 * 299
    s = stage_action(Stage.SALVAGE, 299, p0=257, break_even=214, forward_ship=45, packaging=10)
    assert 214 * 0.95 <= s.price < 257


def test_bandit_respects_guardrails():
    b = LinUCB(n_features=2, arms=(-10, 0, 10, 25))
    x = np.array([1.0, 0.5])
    for _ in range(50):
        arm = b.select(x, price=300, p0=295, pmax=310)
        assert 295 <= 300 + b.arms[arm] <= 310
        b.update(arm, x, reward=1.0)


def test_bandit_learns_best_arm():
    rng = np.random.default_rng(0)
    b = LinUCB(n_features=1, arms=(-10, 0, 10), alpha=0.5)
    true = {-10: 0.2, 0: 0.5, 10: 0.9}
    x = np.array([1.0])
    picks = []
    for _ in range(300):
        arm = b.select(x, price=500, p0=0, pmax=1000)
        picks.append(b.arms[arm])
        b.update(arm, x, true[b.arms[arm]] + rng.normal(0, 0.1))
    assert picks[-50:].count(10) > 40


def test_backtest_vyapar_beats_manual():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from scripts.backtest import main
    r = main(seeds=range(5))
    assert r["vyapar_ai"]["profit_first_100_orders"] > 0 > r["manual"]["profit_first_100_orders"]
    assert r["vyapar_ai"]["day_reached_final_price"] < 60
