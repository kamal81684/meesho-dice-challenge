"""Risk & margin engine: predicts return and RTO probability (Slide 6, step 4).

LightGBM when installed, otherwise scikit-learn's HistGradientBoosting.
It also scores *levers*: how much adding a size chart, a fabric card or more
photos would cut the predicted return rate. These drive the nudges.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .catalog import TAXONOMY, Listing

SUBS = [s for subs in TAXONOMY.values() for s in subs]
APPAREL = {"women_ethnic", "men_fashion"}


@dataclass
class RiskFeatures:
    subcategory: str
    category: str
    rel_price: float = 1.0          # price / market median
    cod_share: float = 0.75
    has_size_chart: bool = False
    has_fabric_card: bool = False
    image_count: int = 3
    rating: float = 4.0             # prior for a new seller with zero reviews

    def vector(self) -> list[float]:
        onehot = [1.0 if self.subcategory == s else 0.0 for s in SUBS]
        return [self.rel_price, self.cod_share, float(self.has_size_chart),
                float(self.has_fabric_card), float(self.image_count), self.rating,
                1.0 if self.category in APPAREL else 0.0, *onehot]


def _make_regressor():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                 min_child_samples=20, verbose=-1)
    except ImportError:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)


class RiskModel:
    def __init__(self) -> None:
        self.ret_model = _make_regressor()
        self.rto_model = _make_regressor()
        self.fitted = False

    def fit(self, listings: list[Listing]) -> "RiskModel":
        medians: dict[str, float] = {}
        for s in SUBS:
            ps = [l.price for l in listings if l.subcategory == s]
            medians[s] = float(np.median(ps)) if ps else 1.0
        X = np.array([RiskFeatures(
            subcategory=l.subcategory, category=l.category,
            rel_price=l.price / medians[l.subcategory], cod_share=l.cod_share,
            has_size_chart=l.has_size_chart, has_fabric_card=l.has_fabric_card,
            image_count=l.image_count, rating=l.rating).vector() for l in listings])
        self.ret_model.fit(X, np.array([l.return_rate for l in listings]))
        self.rto_model.fit(X, np.array([l.rto_rate for l in listings]))
        self.fitted = True
        return self

    def predict(self, f: RiskFeatures) -> tuple[float, float]:
        x = np.array([f.vector()])
        with warnings.catch_warnings():
            # LightGBM names columns Column_0..N at fit time; plain arrays are expected here.
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            ret = float(np.clip(self.ret_model.predict(x)[0], 0.0, 0.9))
            rto = float(np.clip(self.rto_model.predict(x)[0], 0.0, 0.9))
        return ret, rto

    def lever_impacts(self, f: RiskFeatures) -> list[dict]:
        """Predicted return-rate change for each listing-quality lever."""
        base_ret, _ = self.predict(f)
        levers = []
        candidates = [
            ("add_size_chart", "Add a measurement/size chart image",
             {"has_size_chart": True}, not f.has_size_chart and f.category in APPAREL),
            ("add_fabric_card", "Add a fabric/GSM detail card",
             {"has_fabric_card": True}, not f.has_fabric_card),
            ("more_photos", "Upload at least 5 clear photos (drape, close-up, back)",
             {"image_count": max(5, f.image_count)}, f.image_count < 5),
        ]
        for key, label, change, applicable in candidates:
            if not applicable:
                continue
            g = RiskFeatures(**{**f.__dict__, **change})
            new_ret, _ = self.predict(g)
            if new_ret < base_ret - 0.005:
                levers.append({"lever": key, "label": label,
                               "return_rate_before": round(base_ret, 4),
                               "return_rate_after": round(new_ret, 4),
                               "relative_drop": round(1 - new_ret / base_ret, 3)
                               if base_ret else 0.0})
        return sorted(levers, key=lambda d: d["return_rate_after"])
