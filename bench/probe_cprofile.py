"""cProfile each framework on a single representative ticket.

Captures top-N cumulative-time functions from each framework's hot path,
filtering out the network/HTTP layer (which is shared) so framework-specific
overhead is visible.
"""
from __future__ import annotations

import cProfile
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent

TICKET = {
    "ticket_id": "T-PROFILE",
    "email_text": "My toaster arrived with a dent on the side 3 days ago. Requesting a refund.",
    "requested_refund": 49.99,
}


def profile_one(fw_name: str, mod):
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    mod.run(TICKET)
    wall = time.perf_counter() - t0
    pr.disable()

    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    stats.print_stats(40)
    full = buf.getvalue()

    # Extract framework-only lines (drop httpx/google.api/urllib).
    fw_only = []
    for ln in full.splitlines():
        s = ln.strip()
        if not s:
            continue
        if any(x in s for x in ["langgraph", "crewai", "dspy", "litellm",
                                 "langchain_google_genai", "langchain_core",
                                 "pydantic"]):
            if any(x in s for x in ["site-packages", "function calls"]):
                fw_only.append(s)

    return {"wall_s": round(wall, 3), "full_pstats": full, "fw_lines": fw_only[:30]}


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("set GEMINI_API_KEY", file=sys.stderr); sys.exit(1)
    os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"

    out = {}
    for fw in ["langgraph", "crewai", "dspy"]:
        print(f"\n=== {fw} ===", flush=True)
        if fw == "langgraph":
            import impl_langgraph as mod
        elif fw == "crewai":
            import impl_crewai as mod
        elif fw == "dspy":
            import impl_dspy as mod
        out[fw] = profile_one(fw, mod)
        print(f"wall: {out[fw]['wall_s']}s")
        print("\n".join(out[fw]["fw_lines"][:15]))

    p = ROOT / "results" / "probe_cprofile.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
