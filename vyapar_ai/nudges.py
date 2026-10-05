"""Hinglish WhatsApp-style action cards (Slide 5, right-hand side).

The MVP uses templates. The pilot would put Indic-BERT translation and the
WhatsApp Business API behind the same function signatures.
"""
from __future__ import annotations


def _rs(x: float) -> str:
    return f"₹{x:,.0f}"


def launch_card(name: str, product: str, price: int, band: tuple[float, float],
                margin: float, return_risk: float) -> dict:
    return {
        "type": "launch_price",
        "text": (f"Namaste {name} ji! 🙏 Aapke {product} ke liye recommended launch price "
                 f"{_rs(price)} hai. Market band {_rs(band[0])}–{_rs(band[1])} hai. "
                 f"Is price par har order par lagbhag {_rs(margin)} munafa bachega "
                 f"(return/RTO ka kharcha pehle se jod diya hai; return risk "
                 f"{return_risk:.0%})."),
        "actions": [f"Accept {_rs(price)}", "Keep my price", "Explain the calculation"],
    }


def lever_card(lever: dict, category_label: str) -> dict:
    drop = lever["relative_drop"]
    hints = {
        "add_size_chart": "returns size mismatch se hote hain. Ek measurement-chart photo add karein",
        "add_fabric_card": "returns fabric quality mismatch se hote hain. Fabric/GSM card add karein",
        "more_photos": "buyers ko kapda saaf nahi dikhta. Kam se kam 5 photos daalein",
    }
    return {
        "type": "lever_nudge",
        "lever": lever["lever"],
        "text": (f"Aapki category ({category_label}) mein kaafi {hints[lever['lever']]}; "
                 f"returns lagbhag {drop:.0%} tak gir sakte hain!"),
        "actions": [lever["label"], "Later"],
    }


def competitor_alert(name: str, product: str, competitor_drop: float, new_price: int,
                     order_uplift: float, margin: float) -> dict:
    return {
        "type": "competitor_alert",
        "text": (f"Namaste {name} ji! Competitor ne {product} ka daam {_rs(competitor_drop)} "
                 f"kam kiya hai. Agar aap {_rs(new_price)} karte hain, orders "
                 f"{order_uplift:.0%} badhenge aur har piece par {_rs(margin)} munafa bachega."),
        "actions": [f"Accept {_rs(new_price)} (1-click)", "Keep existing price"],
    }


def stage_card(name: str, stage: str, action: str, price: int) -> dict:
    lines = {
        "launch": "Abhi launch phase hai, pehle 25 orders aur reviews par focus karein.",
        "scale": "Reviews achhe aa rahe hain! Hum dheere-dheere daam badha rahe hain.",
        "defense": "Competition badh gaya hai. Daam same rakhein, 2/3 ka pack banayein.",
        "salvage": "Stock purana ho raha hai. Thoda kam daam par nikaal kar paisa free karein.",
    }
    return {"type": "stage_update", "stage": stage,
            "text": f"{name} ji, {lines.get(stage, '')} Suggested price: {_rs(price)}. ({action})",
            "actions": [f"Accept {_rs(price)}", "Keep existing price"]}
