"""Capture per-framework reasoning-token breakdown on Gemini 2.5 Flash.

Bench callbacks above only sum prompt+completion+total. Gemini 2.5 emits
`output_token_details.reasoning` (langchain) / `completion_tokens_details.reasoning_tokens`
(litellm) which counts internal reasoning included inside completion.

Runs N tickets per framework, captures reasoning per LLM call, sums per ticket.
Writes results/probe_reasoning.json with per-ticket reasoning tokens.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

bucket = {"prompt": 0, "completion": 0, "reasoning": 0, "total": 0, "calls": 0}


def reset():
    for k in bucket:
        bucket[k] = 0


def _litellm_cb(kwargs, completion_response, start_time, end_time):
    try:
        u = completion_response.get("usage") or {}
        if hasattr(u, "model_dump"):
            u = u.model_dump()
        bucket["prompt"] += int(u.get("prompt_tokens") or 0)
        bucket["completion"] += int(u.get("completion_tokens") or 0)
        bucket["total"] += int(u.get("total_tokens") or 0)
        bucket["calls"] += 1
    except Exception:
        return
    try:
        details = u.get("completion_tokens_details") or {}
        if hasattr(details, "model_dump"):
            details = details.model_dump()
        bucket["reasoning"] += int(details.get("reasoning_tokens") or 0)
    except Exception:
        pass


def install_litellm_cb():
    import litellm
    litellm.success_callback = [_litellm_cb]


def install_langchain_cb():
    from langchain_google_genai import ChatGoogleGenerativeAI
    if getattr(ChatGoogleGenerativeAI, "_reasoning_patched", False):
        return
    orig = ChatGoogleGenerativeAI.invoke

    def wrapped(self, *a, **kw):
        r = orig(self, *a, **kw)
        m = getattr(r, "usage_metadata", None) or {}
        bucket["prompt"] += int(m.get("input_tokens") or 0)
        bucket["completion"] += int(m.get("output_tokens") or 0)
        bucket["total"] += int(m.get("total_tokens") or 0)
        details = m.get("output_token_details") or {}
        bucket["reasoning"] += int(details.get("reasoning") or 0)
        bucket["calls"] += 1
        return r

    ChatGoogleGenerativeAI.invoke = wrapped
    ChatGoogleGenerativeAI._reasoning_patched = True


CBS = {
    "langgraph": install_langchain_cb,
    "crewai": install_litellm_cb,
    "dspy": install_litellm_cb,
}


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("set GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)
    os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"

    tickets = json.loads((HERE / "tickets_100.json").read_text())[:30]

    out = {}
    for fw in ["langgraph", "crewai", "dspy"]:
        print(f"\n=== {fw} ===", flush=True)
        CBS[fw]()
        if fw == "langgraph":
            import impl_langgraph as mod
        elif fw == "crewai":
            import impl_crewai as mod
        elif fw == "dspy":
            import impl_dspy as mod

        rows = []
        for i, t in enumerate(tickets, 1):
            reset()
            try:
                mod.run(t)
            except Exception as e:
                rows.append({"ticket_id": t["ticket_id"], "error": f"{type(e).__name__}: {e}"})
                continue
            rows.append({
                "ticket_id": t["ticket_id"],
                "calls": bucket["calls"],
                "prompt": bucket["prompt"],
                "completion": bucket["completion"],
                "reasoning": bucket["reasoning"],
                "total": bucket["total"],
                "non_reasoning_completion": max(bucket["completion"] - bucket["reasoning"], 0),
            })
            if i % 10 == 0:
                print(f"  {fw} {i}/{len(tickets)}", flush=True)
        out[fw] = rows

    p = ROOT / "results" / "probe_reasoning.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")

    # Summary
    print("\n--- Summary (mean per ticket) ---")
    print(f"{'fw':<10} {'calls':>6} {'prompt':>8} {'compl':>8} {'reason':>8} {'reason%':>8}")
    for fw, rows in out.items():
        ok = [r for r in rows if "error" not in r]
        if not ok:
            print(f"{fw}: all errored")
            continue
        n = len(ok)
        c = sum(r["calls"] for r in ok) / n
        p_ = sum(r["prompt"] for r in ok) / n
        cm = sum(r["completion"] for r in ok) / n
        rt = sum(r["reasoning"] for r in ok) / n
        pct = (rt / cm * 100) if cm else 0
        print(f"{fw:<10} {c:>6.2f} {p_:>8.0f} {cm:>8.0f} {rt:>8.0f} {pct:>7.1f}%")


if __name__ == "__main__":
    main()
