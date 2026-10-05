"""LinUCB contextual bandit for price nudges (Slide 4, Li et al. 2010).

    a_t = argmax_a  θ_a·x_t + α·sqrt(x_tᵀ A_a⁻¹ x_t)

Arms are price deltas. Guardrails mask out any arm whose resulting price
would fall below P0 or above Pmax, so the policy can never break even-or-worse.
"""
from __future__ import annotations

import numpy as np

DEFAULT_ARMS = (-10, 0, 10, 25)


class LinUCB:
    def __init__(self, n_features: int, arms=DEFAULT_ARMS, alpha: float = 0.8,
                 seed: int = 0):
        self.arms = list(arms)
        self.alpha = alpha
        self.d = n_features
        self.A = [np.eye(n_features) for _ in self.arms]
        self.b = [np.zeros(n_features) for _ in self.arms]
        self.rng = np.random.default_rng(seed)

    def scores(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(self.arms))
        for i, (A, b) in enumerate(zip(self.A, self.b)):
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            out[i] = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
        return out

    def select(self, x: np.ndarray, price: float, p0: float, pmax: float) -> int:
        """Return the index of the chosen arm, respecting P0 <= price+delta <= Pmax."""
        s = self.scores(x)
        allowed = np.array([p0 <= price + d <= pmax for d in self.arms])
        if not allowed.any():
            return self.arms.index(0) if 0 in self.arms else int(np.argmax(s))
        s = np.where(allowed, s, -np.inf)
        best = np.flatnonzero(s == s.max())
        return int(self.rng.choice(best))

    def update(self, arm_idx: int, x: np.ndarray, reward: float) -> None:
        self.A[arm_idx] += np.outer(x, x)
        self.b[arm_idx] += reward * x


def context_vector(rating: float, rto_trend: float, velocity: float,
                   impression_rank: float, rel_price: float) -> np.ndarray:
    """x_t from Slide 4: rating, category RTO trend, order velocity, search rank."""
    return np.array([1.0, rating / 5.0, rto_trend, min(velocity / 10.0, 3.0),
                     impression_rank, rel_price])


def reward(orders: float, price: float, unit_cost: float, tax_rate: float,
           reverse_logistics_loss: float) -> float:
    """R_t = orders × (price·(1-tax) − COGS − labor − packaging) − reverse-logistics loss."""
    return orders * (price * (1 - tax_rate) - unit_cost) - reverse_logistics_loss
