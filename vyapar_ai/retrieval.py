"""Cluster gate: finds similar listings and summarises market intelligence (Slide 6, step 3).

MVP: hashed bag-of-words + attribute one-hots stand in for CLIP vectors, and
exact cosine search in numpy stands in for FAISS. Both sit behind small
interfaces so they can be swapped without touching the engine.
"""
from __future__ import annotations

from typing import Optional

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from .catalog import COLORS, FABRICS, PATTERNS, Listing

DIM_TEXT = 256


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class Embedder:
    """Maps a product (title + attributes) to a unit vector."""

    vocab_attrs = ([f"fabric={f}" for f in FABRICS] + [f"pattern={p}" for p in PATTERNS]
                   + [f"color={c}" for c in COLORS])

    def __init__(self) -> None:
        self._attr_index = {a: i for i, a in enumerate(self.vocab_attrs)}
        self.dim = DIM_TEXT + len(self.vocab_attrs)

    def embed(self, title: str, fabric: Optional[str] = None, pattern: Optional[str] = None,
              color: Optional[str] = None) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(title):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % DIM_TEXT] += 1.0
        for key, val in (("fabric", fabric), ("pattern", pattern), ("color", color)):
            idx = self._attr_index.get(f"{key}={val}")
            if idx is not None:
                v[DIM_TEXT + idx] += 2.0  # attributes weigh more than free text
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_listing(self, l: Listing) -> np.ndarray:
        return self.embed(l.title, l.fabric, l.pattern, l.color)


@dataclass
class MarketSnapshot:
    level: str                  # "subcategory" | "category" | "global"
    support: int                # number of similar listings used
    median_price: float
    band_low: float
    band_high: float
    avg_rating: float
    mean_return_rate: float
    mean_rto_rate: float
    neighbours: list[dict]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class SimilarityIndex:
    def __init__(self, listings: list[Listing], embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()
        self.listings = listings
        self.matrix = np.vstack([self.embedder.embed_listing(l) for l in listings])
        self._sub = np.array([l.subcategory for l in listings])
        self._cat = np.array([l.category for l in listings])

    def search(self, query: np.ndarray, mask: Optional[np.ndarray] = None,
               k: int = 20) -> list[tuple[int, float]]:
        # Every row is a unit vector, so this dot product is bounded by 1 and
        # cannot really overflow or divide by zero. Some BLAS backends (notably
        # Apple's Accelerate on macOS) still raise spurious floating-point flags
        # for a float32 matmul, which numpy reports as RuntimeWarnings - silence
        # them for this one call and drop any non-finite result defensively.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self.matrix @ query
        if not np.isfinite(sims).all():
            sims = np.nan_to_num(sims, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        k = min(k, int(np.isfinite(sims).sum()))
        if k == 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def market_snapshot(self, title: str, subcategory: str, category: str,
                        fabric: Optional[str] = None, pattern: Optional[str] = None,
                        color: Optional[str] = None, k: int = 20,
                        min_support: int = 8, min_sim: float = 0.3) -> MarketSnapshot:
        """Hierarchical fallback: subcategory -> category -> global.

        Sparse data widens the band (Slide 6, "Hierarchical Category Fallback").
        """
        q = self.embedder.embed(title, fabric, pattern, color)
        levels = [("subcategory", self._sub == subcategory),
                  ("category", self._cat == category),
                  ("global", None)]
        hits: list[tuple[int, float]] = []
        level = "global"
        for level, mask in levels:
            hits = [(i, s) for i, s in self.search(q, mask, k) if s >= min_sim]
            if len(hits) >= min_support:
                break
        if not hits:
            hits = self.search(q, None, k)
        prices = np.array([self.listings[i].price for i, _ in hits])
        widen = {"subcategory": 0.0, "category": 0.10, "global": 0.20}[level]
        lo, hi = np.percentile(prices, [20, 80])
        nb = [self.listings[i] for i, _ in hits]
        return MarketSnapshot(
            level=level,
            support=len(hits),
            median_price=float(np.median(prices)),
            band_low=float(lo * (1 - widen)),
            band_high=float(hi * (1 + widen)),
            avg_rating=float(np.mean([l.rating for l in nb])),
            mean_return_rate=float(np.mean([l.return_rate for l in nb])),
            mean_rto_rate=float(np.mean([l.rto_rate for l in nb])),
            neighbours=[{"listing_id": l.listing_id, "title": l.title, "price": l.price,
                         "rto_rate": l.rto_rate, "similarity": round(s, 3)}
                        for l, (_, s) in zip(nb[:5], hits[:5])],
        )
