"""Hand-written 4-shot LangGraph baseline.

Picks 4 demos by hand (2 eligible, 2 not eligible) covering policy edge cases.
Builds a triage prompt with those demos in-context.
Evaluates on the same 50-ticket test set used by compile_dspy.py
(tickets[50:]) with the same ground-truth labelling rule.
Compares to uncompiled DSPy and compiled DSPy.

This isolates: how much of compiled-DSPy's lift is "compile" vs "having
4 examples in the prompt".
"""
from __future__ import annotations

import json
import os
import sys
import time
import re
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

from langchain_google_genai import ChatGoogleGenerativeAI

# Three hand-picked demo sets. Each has 2 eligible + 2 not, but spans different
# issue types and tones, to bound demo-selection variance. Variant selected with
# --variant flag.
DEMO_SETS = {
    "v1": [
        {
            "ticket": "My toaster arrived with a dent on the side 3 days ago. Requesting a refund.",
            "reasoning": "Item is damaged and within the 30-day window. Damage clause applies.",
            "eligible": True,
        },
        {
            "ticket": "Ordered a blender 50 days ago and changed my mind. Want to return it.",
            "reasoning": "Change-of-mind outside the 30-day window. Damage policy does not cover this; not eligible.",
            "eligible": False,
        },
        {
            "ticket": "Espresso machine started leaking from the base 5 days after delivery.",
            "reasoning": "Defect within the 30-day window counts as damage under the policy. Eligible.",
            "eligible": True,
        },
        {
            "ticket": "Bought running shoes 8 days ago and they don't fit. Want a refund.",
            "reasoning": "Sizing mismatch is not damage. Policy covers damage only. Not eligible.",
            "eligible": False,
        },
    ],
    "v2": [
        {
            "ticket": "Ceramic bowl set arrived cracked yesterday. Want refund.",
            "reasoning": "Damage on arrival, within window. Eligible.",
            "eligible": True,
        },
        {
            "ticket": "Bought a tablet 3 weeks ago and the wrong color was sent. Refund please.",
            "reasoning": "Wrong-item shipment is a fulfilment error, not damage. Policy covers damage only; not eligible.",
            "eligible": False,
        },
        {
            "ticket": "My winter jacket has a torn lining; I noticed it 4 days after delivery.",
            "reasoning": "Manufacturing defect reported within 30 days. Eligible.",
            "eligible": True,
        },
        {
            "ticket": "Camping tent arrived 6 days late. I bought another one elsewhere.",
            "reasoning": "Late delivery is not a damage event. Policy does not cover this; not eligible.",
            "eligible": False,
        },
    ],
    "v3": [
        {
            "ticket": "Stand mixer paddle arrived broken inside the box, 2 days after delivery.",
            "reasoning": "Damaged on arrival, within window. Eligible.",
            "eligible": True,
        },
        {
            "ticket": "Wireless headphones, 12 days old, are missing the charging cable from the box.",
            "reasoning": "Missing critical component within the 30-day window counts as damaged-as-shipped. Eligible.",
            "eligible": True,
        },
        {
            "ticket": "Ordered a vacuum cleaner. Only realized it does not work with my hardwood floors.",
            "reasoning": "Incompatibility is buyer expectation, not damage. Policy covers damage only; not eligible.",
            "eligible": False,
        },
        {
            "ticket": "Bedding set fabric feels cheaper than expected. Would like a refund.",
            "reasoning": "Quality complaint without physical damage falls outside the damage policy. Not eligible.",
            "eligible": False,
        },
    ],
}


def label(meta):
    fit = meta.get("policy_fit")
    days = meta.get("days_since_delivery", 999)
    if fit == "damage_yes" and days < 30:
        return True
    if fit == "damage_maybe":
        return True
    return False


def build_prompt(ticket_text: str, demos) -> str:
    parts = [
        "You are a refund policy analyst. Policy: \"Damaged items refundable within 30 days.\"",
        "For each ticket, decide if the customer is eligible for a refund.",
        "Reply with exactly two lines:",
        "  reasoning: <one short sentence>",
        "  eligible: yes|no",
        "",
        "Examples:",
    ]
    for d in demos:
        parts.extend([
            f"TICKET: {d['ticket']}",
            f"reasoning: {d['reasoning']}",
            f"eligible: {'yes' if d['eligible'] else 'no'}",
            "",
        ])
    parts.extend([
        "Now answer for this ticket.",
        f"TICKET: {ticket_text}",
    ])
    return "\n".join(parts)


def parse_eligible(text: str) -> bool | None:
    m = re.search(r"eligible\s*:\s*(yes|no)", text, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower() == "yes"


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def evaluate(model: str, tickets, demos):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )
    rows = []
    correct = 0
    total_prompt = total_completion = total_calls = 0
    for t in tickets:
        truth = label(t["meta"])
        prompt = build_prompt(t["email_text"], demos)
        s = time.perf_counter()
        try:
            r = llm.invoke(prompt)
        except Exception as e:
            rows.append({"ticket_id": t["ticket_id"], "error": f"{type(e).__name__}: {e}"})
            continue
        dt = time.perf_counter() - s
        text = _text(r)
        pred = parse_eligible(text)
        ok = (pred == truth)
        if ok:
            correct += 1
        m = getattr(r, "usage_metadata", None) or {}
        prompt_t = int(m.get("input_tokens") or 0)
        comp_t = int(m.get("output_tokens") or 0)
        total_prompt += prompt_t
        total_completion += comp_t
        total_calls += 1
        rows.append({
            "ticket_id": t["ticket_id"],
            "latency_s": round(dt, 3),
            "prompt": prompt_t,
            "completion": comp_t,
            "predicted_eligible": pred,
            "true_eligible": truth,
            "correct": ok,
            "raw": text[:200],
        })
    n = len([r for r in rows if "error" not in r])
    return {
        "rows": rows,
        "accuracy": correct / max(n, 1),
        "n": n,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "total_calls": total_calls,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1", choices=list(DEMO_SETS.keys()))
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("set GEMINI_API_KEY", file=sys.stderr); sys.exit(1)
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
    demos = DEMO_SETS[args.variant]

    tickets = json.loads((HERE / "tickets_100.json").read_text())
    test = tickets[50:]
    print(f"Eval against {model} on n={len(test)} tickets, demo variant={args.variant}")
    print(f"True labels: eligible={sum(1 for t in test if label(t['meta']))}/{len(test)}")

    out = evaluate(model, test, demos)
    out["variant"] = args.variant
    print(f"\nHand-4shot LG ({args.variant}) accuracy: {out['accuracy']:.3f} ({sum(1 for r in out['rows'] if r.get('correct'))}/{out['n']})")
    print(f"Total tokens: prompt={out['total_prompt_tokens']} completion={out['total_completion_tokens']} total={out['total_tokens']}")
    print(f"Tokens/ticket: {out['total_tokens']/max(out['n'],1):.0f}")

    # Confusion
    rows = [r for r in out["rows"] if "error" not in r]
    tp = sum(1 for r in rows if r["true_eligible"] and r["predicted_eligible"])
    fn = sum(1 for r in rows if r["true_eligible"] and r["predicted_eligible"] is False)
    fp = sum(1 for r in rows if not r["true_eligible"] and r["predicted_eligible"])
    tn = sum(1 for r in rows if not r["true_eligible"] and r["predicted_eligible"] is False)
    nopar = sum(1 for r in rows if r["predicted_eligible"] is None)
    print(f"\nConfusion: TP={tp} FN={fn} FP={fp} TN={tn} (unparseable={nopar})")
    if tp + fp:
        print(f"  precision={tp/(tp+fp):.3f}")
    if tp + fn:
        print(f"  recall={tp/(tp+fn):.3f}")

    p = ROOT / "results" / f"hand_fewshot_lg_eval_{model}_{args.variant}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
