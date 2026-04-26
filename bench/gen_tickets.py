"""Generate 100 unique refund tickets with deliberate variance.

Axes of variance:
  - issue type (10): damaged, defective, wrong item, late, missing parts, incompatible, quality, change of mind, warranty, sizing
  - product category (20): spanning electronics, kitchen, apparel, home, outdoor
  - amount bucket (5): <20, 20-75, 75-150, 150-300, >300
  - email length (4): short (1 sentence), medium (2-3), long (paragraph), very long (multi-paragraph)
  - tone (5): neutral, frustrated, calm, confused, angry
  - complexity (3): single issue, two issues, policy-ambiguous

Output: bench/tickets_100.json with unique content.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

PRODUCTS = [
    ("kitchen", "stand mixer"), ("kitchen", "espresso machine"), ("kitchen", "blender"),
    ("kitchen", "toaster"), ("kitchen", "cookware set"), ("kitchen", "ceramic bowl set"),
    ("electronics", "wireless headphones"), ("electronics", "smartwatch"),
    ("electronics", "tablet"), ("electronics", "gaming console"),
    ("electronics", "bluetooth speaker"), ("apparel", "running shoes"),
    ("apparel", "winter jacket"), ("apparel", "leather boots"),
    ("home", "vacuum cleaner"), ("home", "air purifier"), ("home", "lamp set"),
    ("home", "bedding set"), ("outdoor", "camping tent"), ("outdoor", "bicycle"),
]

ISSUES = {
    "damaged": {
        "policy_fit": "damage_yes",
        "short": "My {p} arrived damaged.",
        "med": "My {p} was delivered {days} days ago and has a visible dent on the side. Requesting a refund.",
        "long": "I received my {p} on {day_phrase} and when I opened the box the product was visibly damaged. There's a crack running along one side and the packaging looked tampered with. I'd like a refund please.",
        "xlong": "Hi support team, I'm writing about order {oid}. My {p} was delivered {days} days ago. Upon opening, I noticed {detail1}. I tried to use it anyway but {detail2}. The packaging was also in poor condition on arrival (the outer box had a large tear). Could you process a full refund? I still have all original packaging and photos of the damage. Thanks.",
    },
    "defective": {
        "policy_fit": "damage_yes",
        "short": "{p} stopped working after 2 days.",
        "med": "The {p} I ordered worked fine for the first {days} days but now won't turn on at all. Requesting refund.",
        "long": "I bought a {p} {days} days ago and it worked for a while but now it's completely unresponsive. I've tried resetting, different outlets, and the steps in the manual. Nothing works. This seems like a manufacturing defect. I'd like a refund.",
        "xlong": "I purchased a {p} (order {oid}) {days} days ago. Initially everything was fine but over the past few days I've noticed {detail1}. Today it stopped working entirely. I tried {detail2} and all the standard troubleshooting. This is clearly defective. I'd prefer a refund over a replacement because I need the money for an alternative.",
    },
    "wrong_item": {
        "policy_fit": "damage_no",
        "short": "You sent the wrong {p}.",
        "med": "I ordered a {p} but received a completely different item. Please refund.",
        "long": "My order was supposed to be a {p} but when it arrived {days} days ago it was a different product entirely. The packing slip shows the right item but the box contained something else. Please refund or send the correct item.",
        "xlong": "Order {oid}. I ordered a {p} and received something completely different. The SKU on the product doesn't match the SKU on my invoice. I haven't opened anything beyond the outer packaging. I'd like a refund and I don't want to deal with shipping a replacement because the last two orders had issues too. Please confirm.",
    },
    "late": {
        "policy_fit": "damage_no",
        "short": "My {p} arrived {days} days late.",
        "med": "The {p} was supposed to arrive by {day_phrase} but came {days} days late. I no longer need it.",
        "long": "My order was guaranteed for {day_phrase} but the {p} arrived {days} days after that. By the time it got here I had bought an alternative locally. I'd like to return it for a full refund.",
        "xlong": "I placed order {oid} specifically because of the delivery guarantee. The {p} was supposed to arrive {day_phrase} but actually arrived {days} days late. Because of this delay I had to buy {detail1} elsewhere for my {detail2}. I don't want the product anymore and I'd like a full refund including the expedited shipping fee I paid.",
    },
    "missing_parts": {
        "policy_fit": "damage_yes",
        "short": "{p} came with missing parts.",
        "med": "The {p} I received is missing critical components. Refund please.",
        "long": "I got my {p} {days} days ago and when I tried to assemble it I realized several parts are missing. The manual lists {detail1} but the box only contained some of them. Can't use it as-is.",
        "xlong": "Order {oid}, received {days} days ago. The {p} is missing {detail1}. Without these parts I literally cannot use the product. I checked the packaging twice, contacted a friend who owns the same model to confirm, and my box is incomplete. Either send the missing parts within 2 days or refund the full amount.",
    },
    "incompatible": {
        "policy_fit": "damage_no",
        "short": "{p} is incompatible with my setup.",
        "med": "The {p} doesn't work with my existing hardware. Want a refund.",
        "long": "I assumed the {p} would work with my current setup but turns out it requires {detail1} which I don't have. The product description was misleading about compatibility.",
        "xlong": "I bought the {p} assuming it would work with my existing setup. The product description implied broad compatibility but in practice it only works with {detail1} which I don't own and don't plan to buy. This wasn't clear from the listing. I feel the description was misleading and I'd like a refund.",
    },
    "quality": {
        "policy_fit": "damage_maybe",
        "short": "{p} quality is poor.",
        "med": "The {p} feels much cheaper than described. Requesting refund.",
        "long": "The {p} doesn't match the quality shown in product photos. Materials feel low-grade and the finish is uneven. Looks nothing like the listing.",
        "xlong": "I've been a customer for years and this is the first time I've been this disappointed. The {p} arrived {days} days ago and the quality is nowhere near what I expected. The {detail1} is visibly lower quality than in the product photos. I feel the listing was misleading. This is a quality complaint, not a damage complaint, but I want a refund.",
    },
    "change_of_mind": {
        "policy_fit": "damage_no",
        "short": "Changed my mind, want refund for {p}.",
        "med": "I no longer want the {p} I ordered {days} days ago. Refund please.",
        "long": "I ordered the {p} {days} days ago but have decided I don't need it anymore. The product is unopened and in original packaging. Please process a refund.",
        "xlong": "Ordered the {p} {days} days ago on order {oid}. Since then my situation changed and I don't need it anymore. The box is still sealed. I know your policy is strict about change of mind returns but I'm hoping you can make an exception given I'm a long-time customer with no prior returns.",
    },
    "warranty": {
        "policy_fit": "damage_maybe",
        "short": "{p} failed under warranty.",
        "med": "The {p} failed {days} days after purchase, still under warranty.",
        "long": "My {p} stopped working {days} days in. It's still under warranty and the manufacturer is slow to respond. Can you help with a refund since you're the seller?",
        "xlong": "Bought the {p} from you {days} days ago. It failed last week. The manufacturer warranty is theoretically active but they're not responding to my support tickets. As the seller, can you process a refund and handle the warranty claim yourselves? I've attached photos and the failure mode is clearly covered under warranty terms.",
    },
    "sizing": {
        "policy_fit": "damage_no",
        "short": "{p} doesn't fit.",
        "med": "The {p} I ordered doesn't fit. Refund please.",
        "long": "I ordered size {detail1} for the {p} but the actual fit is way off. Based on the size chart I should have been fine. Requesting refund.",
        "xlong": "Per the size chart on your site, I ordered size {detail1} for the {p}. When it arrived {days} days ago the fit was nowhere close. I compared measurements against your published size chart and the product is off by {detail2}. This feels like a manufacturing or listing error rather than my mistake. Refund please, I don't want to gamble on another size.",
    },
}

TONES = {
    "neutral": ["", ""],
    "frustrated": ["This has been frustrating. ", " Please resolve this quickly."],
    "calm": ["Hope you're well. ", " Appreciate your help."],
    "confused": ["I'm not sure how to handle this. ", " Let me know what to do."],
    "angry": ["This is unacceptable. ", " I expect a prompt resolution."],
}

DAYS_BY_ISSUE = {
    "damaged": [2, 5, 8, 12, 18, 25],
    "defective": [3, 7, 14, 22, 35],
    "wrong_item": [1, 3, 6, 10],
    "late": [3, 5, 7, 10, 14],
    "missing_parts": [2, 5, 9, 15],
    "incompatible": [4, 8, 15, 25],
    "quality": [5, 12, 20, 28],
    "change_of_mind": [8, 20, 35, 45, 60],
    "warranty": [45, 80, 150, 250],
    "sizing": [3, 6, 10, 15],
}

AMOUNT_BUCKETS = [
    (5, 20), (20, 75), (75, 150), (150, 300), (300, 600),
]

LENGTHS = ["short", "med", "long", "xlong"]

DETAILS = {
    "defective": [("intermittent flickering", "multiple factory resets"),
                  ("weird humming noise", "different outlets")],
    "damaged": [("a crack along the front", "it still wouldn't work properly"),
                ("the handle broken off", "it was unusable right away")],
    "missing_parts": [("screws and the assembly wrench", ""),
                      ("the power adapter", "")],
    "late": [("backup supplies", "home project")],
    "incompatible": [("USB-C", ""), ("a specific adapter", "")],
    "quality": [("stitching", ""), ("material", "")],
    "sizing": [("medium", "at least one full size"), ("large", "two inches off")],
    "wrong_item": [("", ""), ("", "")],
    "change_of_mind": [("", "")],
    "warranty": [("", "")],
}


def day_phrase(n: int) -> str:
    if n < 7:
        return f"{n} days ago"
    if n < 30:
        return f"about {n // 7} weeks ago"
    if n < 90:
        return f"roughly {n // 30} months ago"
    return f"about {n // 30} months ago"


def gen_tickets(n=100):
    issues = list(ISSUES.keys())
    tones = list(TONES.keys())
    tickets = []
    seen_text = set()
    tid = 1000
    attempts = 0
    while len(tickets) < n and attempts < n * 10:
        attempts += 1
        issue = issues[attempts % len(issues)]
        category, product = PRODUCTS[random.randrange(len(PRODUCTS))]
        length = LENGTHS[random.randrange(len(LENGTHS))]
        tone = tones[random.randrange(len(tones))]
        days = random.choice(DAYS_BY_ISSUE[issue])
        lo, hi = AMOUNT_BUCKETS[random.randrange(len(AMOUNT_BUCKETS))]
        amount = round(random.uniform(lo, hi), 2)
        oid = f"ORD-{random.randint(10000, 99999)}"
        det1, det2 = random.choice(DETAILS.get(issue, [("", "")]))
        template = ISSUES[issue][length]
        body = template.format(
            p=product, days=days, day_phrase=day_phrase(days),
            oid=oid, detail1=det1, detail2=det2,
        )
        pre, post = TONES[tone]
        text = (pre + body + post).strip()
        if text in seen_text:
            continue
        seen_text.add(text)
        tid += 1
        tickets.append({
            "ticket_id": f"T-{tid}",
            "email_text": text,
            "requested_refund": amount,
            "meta": {
                "issue": issue,
                "policy_fit": ISSUES[issue]["policy_fit"],
                "category": category,
                "product": product,
                "length_bucket": length,
                "tone": tone,
                "days_since_delivery": days,
                "amount_bucket": f"{lo}-{hi}",
                "email_char_len": len(text),
                "email_word_len": len(text.split()),
            },
        })
    return tickets


if __name__ == "__main__":
    tickets = gen_tickets(100)
    out = Path(__file__).parent / "tickets_100.json"
    out.write_text(json.dumps(tickets, indent=2))
    lengths = [t["meta"]["email_word_len"] for t in tickets]
    print(f"Wrote {len(tickets)} tickets to {out}")
    print(f"  word count min/median/max: {min(lengths)} / {sorted(lengths)[len(lengths)//2]} / {max(lengths)}")
    print(f"  issues: {sorted(set(t['meta']['issue'] for t in tickets))}")
    print(f"  amount range: ${min(t['requested_refund'] for t in tickets)} to ${max(t['requested_refund'] for t in tickets)}")
