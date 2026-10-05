"""Decision layer: fuzzy, text-heavy decisions with calibrated confidence.

Backends
--------
* ``JevDecisionModel``  : TypeSafe AI's System One model "Jev" (``typesafe_sdk``).
  Unstructured seller/listing state in, typed decisions plus probabilities out,
  all questions answered in one parallel query (~100 ms).
* ``RuleDecisionModel`` : deterministic keyword/heuristic fallback with the same
  output types, used offline, in tests, or when no ``TYPESAFE_API_KEY`` is set.

Questions asked
---------------
* ``listing_ambiguity`` (Score 1–5): can a buyer verify size, fabric and colour
  from this listing? Slide 6 lists "visual ambiguity score" as a risk feature.
  It scales the predicted return rate.
* ``return_reason`` (Choice): the dominant reason in buyer return comments.
  It decides which listing lever to nudge first.
* ``lifecycle_stage`` (Choice): reads signals *and* free text (reviews, seller
  notes). It is used only when its confidence clears a threshold; otherwise
  the deterministic state machine decides.

Price guardrails (P >= P0, P <= Pmax) are never delegated to a model.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

RETURN_REASONS = {
    "size_mismatch": "Wrong size or fit (too tight, too loose, too short or too long).",
    "fabric_quality": "Fabric or quality differs from the listing (thin, rough, see-through, cheap).",
    "color_mismatch": "Colour or print looks different from the photos.",
    "damaged": "Arrived torn, stained or damaged.",
    "changed_mind": "Buyer no longer wanted it or ordered by impulse (COD refusal).",
}

STAGES = {
    "launch": "New listing: few or no reviews, still seeding first orders.",
    "scale": "Healthy demand: good rating, conversion holding, restocking regularly.",
    "defense": "Mature or under pressure: competitors undercutting or conversion falling.",
    "salvage": "Dead stock: very low sales velocity and ageing inventory.",
}

AMBIGUITY_RUBRIC = [
    "Crystal clear: size chart, fabric/GSM, exact colour and several photos.",
    "Mostly clear: one minor detail missing.",
    "Some ambiguity: buyer must guess size or fabric.",
    "Ambiguous: little detail; size and fabric unclear.",
    "Very ambiguous: a bare title and one photo; high regret risk.",
]

LEVER_FOR_REASON = {
    "size_mismatch": "add_size_chart",
    "fabric_quality": "add_fabric_card",
    "color_mismatch": "more_photos",
}


@dataclass
class Decision:
    value: Any                          # label (choice) or expected score
    confidence: float
    probabilities: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"value": self.value, "confidence": round(self.confidence, 4),
                "probabilities": {str(k): round(v, 4) for k, v in self.probabilities.items()}}


@dataclass
class DecisionBundle:
    backend: str
    decisions: dict[str, Decision]

    def to_dict(self) -> dict:
        return {"backend": self.backend,
                "decisions": {k: v.to_dict() for k, v in self.decisions.items()}}


class DecisionModel(Protocol):
    name: str

    def decide(self, state: dict, questions: list[str]) -> DecisionBundle: ...


# ---------------------------------------------------------------- rules backend

_REASON_KEYWORDS = {
    "size_mismatch": r"size|fit|tight|loose|small|big|short|long|chhot|bada|badi|length",
    "fabric_quality": r"fabric|cloth|quality|thin|rough|transparent|see.?through|kapda|material|cheap",
    "color_mismatch": r"colou?r|shade|rang|faded|different from (the )?photo|print",
    "damaged": r"damag|torn|tear|stain|phat|hole|broken",
    "changed_mind": r"don.?t want|not needed|changed? (my )?mind|by mistake|refus|cancel",
}


class RuleDecisionModel:
    name = "rules"

    def decide(self, state: dict, questions: list[str]) -> DecisionBundle:
        out: dict[str, Decision] = {}
        if "listing_ambiguity" in questions:
            out["listing_ambiguity"] = self._ambiguity(state)
        if "return_reason" in questions:
            out["return_reason"] = self._return_reason(state.get("return_comments") or [])
        if "lifecycle_stage" in questions:
            from .lifecycle import Signals, determine_stage
            sig = Signals(**state["signals"])
            out["lifecycle_stage"] = Decision(determine_stage(sig).value, 0.9,
                                              {determine_stage(sig).value: 0.9})
        return DecisionBundle(self.name, out)

    @staticmethod
    def _ambiguity(s: dict) -> Decision:
        text = f"{s.get('title', '')} {s.get('description', '')}".lower()
        score = 3.0
        score -= 0.6 if s.get("has_size_chart") else 0.0
        score -= 0.6 if (s.get("has_fabric_card") or re.search(r"gsm|100% ?cotton|pure|fabric", text)) else 0.0
        score -= 0.5 if s.get("image_count", 0) >= 5 else 0.0
        score += 0.8 if len(s.get("description", "")) < 30 else -0.3
        score += 0.5 if s.get("image_count", 0) <= 1 else 0.0
        score = min(5.0, max(1.0, score))
        lo = int(score)
        frac = score - lo
        probs = {lo: 1 - frac} if frac == 0 else {lo: 1 - frac, lo + 1: frac}
        return Decision(round(score, 2), 0.6, probs)

    @staticmethod
    def _return_reason(comments: list[str]) -> Decision:
        counts = {k: 0.5 for k in RETURN_REASONS}           # Laplace smoothing
        for c in comments:
            for k, pat in _REASON_KEYWORDS.items():
                if re.search(pat, c.lower()):
                    counts[k] += 1
        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()}
        best = max(probs, key=probs.get)
        return Decision(best, probs[best], probs)


# ------------------------------------------------------------------ Jev backend

class JevDecisionModel:
    """TypeSafe System One ("Jev") backend. Needs ``TYPESAFE_API_KEY``."""

    name = "jev"

    def __init__(self, client: Any = None, model: str | None = None):
        if client is None:
            from typesafe_sdk import TypeSafeClient
            client = TypeSafeClient(model=model or os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-latest"))
        self.client = client

    @staticmethod
    def build_questions(names: list[str]) -> dict:
        from typesafe_sdk import Choice, Score
        q: dict = {}
        if "listing_ambiguity" in names:
            q["listing_ambiguity"] = Score(
                instructions=("How hard is it for an Indian marketplace buyer to verify the "
                              "size, fabric and exact colour of this product from the listing "
                              "alone? Higher means more ambiguity and a higher return risk."),
                criteria=AMBIGUITY_RUBRIC)
        if "return_reason" in names:
            q["return_reason"] = Choice(
                instructions=("Across the buyer return/review comments (which may be Hindi, "
                              "Hinglish or English), what is the dominant reason for returns?"),
                criteria=RETURN_REASONS)
        if "lifecycle_stage" in names:
            q["lifecycle_stage"] = Choice(
                instructions=("Given the sales signals, reviews and seller notes, which "
                              "lifecycle stage is this listing in?"),
                criteria=STAGES)
        return q

    def decide(self, state: dict, questions: list[str]) -> DecisionBundle:
        resp = self.client.system_one(state=state, questions=self.build_questions(questions))
        out: dict[str, Decision] = {}
        for name, ans in resp.choices.items():
            out[name] = Decision(ans.choice, ans.confidence, dict(ans.probabilities))
        for name, ans in resp.scores.items():
            out[name] = Decision(_rescale_score(ans.score, ans.legend), ans.confidence,
                                 dict(ans.probabilities))
        return DecisionBundle(self.name, out)


def _rescale_score(score: float, legend: dict) -> float:
    """Map Jev's expected score onto our 1–5 rubric whatever the key base."""
    keys = sorted(int(k) for k in legend) if legend else [1, 5]
    lo, hi = keys[0], keys[-1]
    if hi == lo:
        return 3.0
    return round(1 + 4 * (score - lo) / (hi - lo), 3)


class FallbackDecisionModel:
    """Try Jev; on any API/network error, answer with rules so pricing never blocks."""

    def __init__(self, primary: DecisionModel, fallback: DecisionModel | None = None):
        self.primary = primary
        self.fallback = fallback or RuleDecisionModel()
        self.name = primary.name
        self.last_error: str | None = None

    def decide(self, state: dict, questions: list[str]) -> DecisionBundle:
        try:
            return self.primary.decide(state, questions)
        except Exception as e:  # noqa: BLE001 - any SDK failure degrades gracefully
            self.last_error = f"{type(e).__name__}: {e}"
            bundle = self.fallback.decide(state, questions)
            bundle.backend = f"{self.fallback.name} (jev unavailable)"
            return bundle


def default_decision_model() -> DecisionModel:
    """Jev when ``TYPESAFE_API_KEY`` is set and the SDK is installed, else rules."""
    if os.getenv("TYPESAFE_API_KEY"):
        try:
            return FallbackDecisionModel(JevDecisionModel())
        except ImportError:
            pass
    return RuleDecisionModel()


# ------------------------------------------------------------- applying output

def ambiguity_multiplier(score: float) -> float:
    """1 (clear) -> 0.85×, 3 -> 1.0×, 5 (very ambiguous) -> 1.15× return rate."""
    return 1 + 0.075 * (score - 3)


def prioritise_levers(levers: list[dict], reason: Decision | None) -> list[dict]:
    """Put the lever that fixes the dominant return reason first, weighted by probability."""
    if reason is None or not reason.probabilities:
        return levers
    def weight(l: dict) -> float:
        p = sum(prob for r, prob in reason.probabilities.items()
                if LEVER_FOR_REASON.get(r) == l["lever"])
        return -(p * 10 + l["relative_drop"])
    return sorted(levers, key=weight)
