"""Hinglish WhatsApp-style action cards (Slide 5, right-hand side).

The MVP uses templates. The pilot would put Indic-BERT translation and the
WhatsApp Business API behind the same function signatures.
"""
from __future__ import annotations


def rs(x: float) -> str:
    return f"₹{x:,.0f}"


_rs = rs


def _action(action_id: str, label: str, **extra) -> dict:
    """A button on the card. `id` tells the client (or WhatsApp webhook) what to do."""
    return {"id": action_id, "label": label, **extra}


def launch_card(name: str, product: str, price: int, band: tuple[float, float],
                margin: float, return_risk: float) -> dict:
    return {
        "type": "launch_price",
        "text": (f"Namaste {name} ji! 🙏 Aapke {product} ke liye recommended launch price "
                 f"{_rs(price)} hai. Market band {_rs(band[0])}–{_rs(band[1])} hai. "
                 f"Is price par har order par lagbhag {_rs(margin)} munafa bachega "
                 f"(return/RTO ka kharcha pehle se jod diya hai; return risk "
                 f"{return_risk:.0%})."),
        "actions": [_action("accept_price", f"Accept {_rs(price)}", price=price),
                    _action("keep_price", "Keep my price"),
                    _action("explain", "Explain the calculation")],
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
        "actions": [_action("apply_lever", lever["label"], lever=lever["lever"]),
                    _action("dismiss", "Later")],
    }


def competitor_alert(name: str, product: str, competitor_drop: float, new_price: int,
                     order_uplift: float, margin: float) -> dict:
    return {
        "type": "competitor_alert",
        "text": (f"Namaste {name} ji! Competitor ne {product} ka daam {_rs(competitor_drop)} "
                 f"kam kiya hai. Agar aap {_rs(new_price)} karte hain, orders "
                 f"{order_uplift:.0%} badhenge aur har piece par {_rs(margin)} munafa bachega."),
        "actions": [_action("accept_price", f"Accept {_rs(new_price)} (1-click)", price=new_price),
                    _action("keep_price", "Keep existing price")],
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
            "actions": [_action("accept_price", f"Accept {_rs(price)}", price=price),
                        _action("keep_price", "Keep existing price")]}


def explain_text(cost: dict, rec: dict, risk: dict) -> str:
    """Hinglish walk-through of the P0 floor, shown for "Explain the calculation"."""
    fees = cost.get("platform_fees", 0)
    fees_line = f" + fees {_rs(fees)}" if fees else ""
    return (f"Hisaab aise bana: kapda {_rs(cost['cogs'])} + labour {_rs(cost['labor'])} + "
            f"packing {_rs(cost['packaging'])}{fees_line}. RTO ka khatra "
            f"{risk['predicted_rto_rate']:.0%} × {_rs(cost['c_rto'])} = {_rs(cost['expected_rto_cost'])}, "
            f"return ka khatra {risk['predicted_return_rate']:.0%} × {_rs(cost['c_damage'])} = "
            f"{_rs(cost['expected_return_cost'])}. Aapka munafa {_rs(cost['target_profit'])}. "
            f"Sab jodkar {cost['tax_rate']:.0%} GST/TCS ke baad minimum daam (P₀) "
            f"{_rs(rec['floor_price'])} banta hai. Isse kam par aapka target munafa nahi bachega, "
            f"aur {_rs(rec['break_even'])} se kam par har order par nuksaan hoga.")


def price_check_card(price: int, verdict: str, margin: float, floor: float,
                     break_even: float, band_high: float, recommended: int) -> dict:
    lines = {
        "loss": (f"Dhyaan dein: {_rs(price)} par har order par lagbhag {_rs(-margin)} ka "
                 f"nuksaan hoga, kyunki break-even {_rs(break_even)} hai."),
        "below_target": (f"{_rs(price)} par nuksaan nahi hoga, par munafa sirf {_rs(margin)} "
                         f"bachega. Aapka target pane ke liye kam se kam {_rs(floor)} rakhein."),
        "overpriced": (f"{_rs(price)} par har order par {_rs(margin)} munafa hai, par yeh market "
                       f"band ({_rs(band_high)} tak) se upar hai, isliye search mein neeche "
                       f"dikhega aur orders kam aayenge."),
        "ok": f"{_rs(price)} theek hai! Har order par lagbhag {_rs(margin)} munafa bachega.",
    }
    actions = [_action("accept_price", f"List at {_rs(price)}", price=price)]
    if price != recommended:
        actions.append(_action("accept_price", f"Use {_rs(recommended)} instead",
                               price=recommended))
    return {"type": "price_check", "verdict": verdict, "text": lines[verdict],
            "actions": actions}
