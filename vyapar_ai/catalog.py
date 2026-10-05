"""Synthetic marketplace catalog.

Stands in for Meesho's catalog tables and Valmo delivery/return logs. The
distributions follow the public figures cited on Slide 9 (COD-heavy demand,
25–40% fashion returns, ₹80–140 dead freight per failed delivery).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

# category -> subcategory -> (median price, base return rate, base rto rate, weight kg)
TAXONOMY: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "women_ethnic": {
        "kurti": (279, 0.20, 0.18, 0.35),
        "cotton_suit_set": (385, 0.14, 0.12, 0.55),
        # Generic printed/cotton saree: the catch-all when the saree type is
        # not clearly synthetic or Banarasi silk. Without it a plain saree had
        # no correct bucket and the vision model fell back to "kurti".
        "saree": (399, 0.16, 0.14, 0.55),
        "synthetic_saree": (449, 0.18, 0.16, 0.60),
        "banarasi_silk_saree": (7200, 0.24, 0.10, 0.90),
        "dupatta": (189, 0.10, 0.12, 0.20),
    },
    "men_fashion": {
        "tshirt": (249, 0.22, 0.17, 0.25),
        "casual_shirt": (349, 0.20, 0.16, 0.30),
    },
    "home": {
        "bedsheet": (399, 0.08, 0.10, 0.80),
        "cushion_cover": (229, 0.07, 0.09, 0.30),
    },
}

FABRICS = ["cotton", "rayon", "silk", "polyester", "georgette", "linen"]
PATTERNS = ["printed", "solid", "embroidered", "woven", "striped", "floral"]
COLORS = ["pink", "red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
CLUSTERS = ["surat", "varanasi", "jaipur", "tiruppur", "ludhiana", "kolkata"]


@dataclass
class Listing:
    listing_id: str
    title: str
    category: str
    subcategory: str
    fabric: str
    pattern: str
    color: str
    cluster: str
    price: float
    rating: float
    reviews: int
    weight_kg: float
    cod_share: float
    has_size_chart: bool
    has_fabric_card: bool
    image_count: int
    return_rate: float = 0.0
    rto_rate: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def true_risk(sub: str, price: float, cod_share: float, has_size_chart: bool,
              has_fabric_card: bool, image_count: int, rating: float,
              category: str) -> tuple[float, float]:
    """Ground-truth risk function the synthetic world is generated from.

    The risk model must learn this from data; the API never calls it directly.
    """
    median, base_ret, base_rto, _ = _sub_info(sub)
    apparel = category in ("women_ethnic", "men_fashion")
    rel_price = price / median
    ret = base_ret
    ret += 0.06 if (apparel and not has_size_chart) else 0.0
    ret += 0.05 if not has_fabric_card else 0.0
    ret -= 0.01 * min(image_count, 6)
    ret += 0.04 * (rel_price - 1)               # pricier than peers -> more regret
    ret -= 0.03 * (rating - 4.0)
    rto = base_rto + 0.18 * (cod_share - 0.7) + 0.05 * (rel_price - 1)
    return float(min(max(ret, 0.02), 0.6)), float(min(max(rto, 0.02), 0.6))


def _sub_info(sub: str) -> tuple[float, float, float, float]:
    for subs in TAXONOMY.values():
        if sub in subs:
            return subs[sub]
    raise KeyError(f"unknown subcategory: {sub}")


def category_of(sub: str) -> str:
    for cat, subs in TAXONOMY.items():
        if sub in subs:
            return cat
    raise KeyError(f"unknown subcategory: {sub}")


def generate_catalog(n: int = 3000, seed: int = 7) -> list[Listing]:
    rng = random.Random(seed)
    items: list[Listing] = []
    subs = [(c, s) for c, d in TAXONOMY.items() for s in d]
    for i in range(n):
        cat, sub = rng.choice(subs)
        median, _, _, weight = TAXONOMY[cat][sub]
        fabric = "silk" if "silk" in sub else rng.choice(FABRICS)
        pattern = rng.choice(PATTERNS)
        color = rng.choice(COLORS)
        cluster = "varanasi" if "banarasi" in sub else rng.choice(CLUSTERS)
        price = round(median * rng.lognormvariate(0, 0.15))
        rating = round(min(5.0, max(2.5, rng.gauss(4.0, 0.35))), 1)
        cod = min(0.95, max(0.4, rng.gauss(0.75, 0.08)))
        size_chart = rng.random() < 0.45
        fabric_card = rng.random() < 0.40
        images = rng.randint(1, 7)
        ret, rto = true_risk(sub, price, cod, size_chart, fabric_card, images, rating, cat)
        # Observed rates are noisy (finite orders).
        ret = min(0.7, max(0.0, ret + rng.gauss(0, 0.025)))
        rto = min(0.7, max(0.0, rto + rng.gauss(0, 0.025)))
        title = f"{color.title()} {pattern} {fabric} {sub.replace('_', ' ')}"
        items.append(Listing(
            listing_id=f"L{i:05d}", title=title, category=cat, subcategory=sub,
            fabric=fabric, pattern=pattern, color=color, cluster=cluster,
            price=float(price), rating=rating, reviews=rng.randint(0, 2000),
            weight_kg=round(weight * rng.uniform(0.8, 1.3), 2), cod_share=round(cod, 3),
            has_size_chart=size_chart, has_fabric_card=fabric_card, image_count=images,
            return_rate=round(ret, 4), rto_rate=round(rto, 4),
        ))
    return items
