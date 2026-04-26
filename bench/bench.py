"""End-to-end benchmark of LangGraph vs CrewAI vs DSPy.

Args:
  --model <name>       Gemini model (defaults to gemini-2.5-flash via env).
  --tickets <path>     JSON array of tickets (defaults to bench/tickets_100.json).
  --frameworks <list>  Comma list: langgraph,crewai,dspy (default all three).
  --out <path>         Output JSON (defaults to results/bench_<model>.json).

Measures wall-clock latency, token usage (via callbacks), LLM call count,
and captures per-ticket metadata for post-hoc analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path


class TicketTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TicketTimeout("per-ticket timeout exceeded")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

_token_bucket = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}


def _reset_bucket():
    for k in _token_bucket:
        _token_bucket[k] = 0


def _install_litellm_callback():
    import litellm

    def cb(kwargs, completion_response, start_time, end_time):
        try:
            usage = completion_response.get("usage") or {}
            _token_bucket["prompt"] += int(usage.get("prompt_tokens") or 0)
            _token_bucket["completion"] += int(usage.get("completion_tokens") or 0)
            _token_bucket["total"] += int(usage.get("total_tokens") or 0)
            _token_bucket["calls"] += 1
        except Exception:
            pass

    litellm.success_callback = [cb]


def _install_langchain_callback():
    from langchain_google_genai import ChatGoogleGenerativeAI

    if getattr(ChatGoogleGenerativeAI, "_bench_patched", False):
        return
    orig = ChatGoogleGenerativeAI.invoke

    def wrapped(self, *a, **kw):
        r = orig(self, *a, **kw)
        meta = getattr(r, "usage_metadata", None) or {}
        _token_bucket["prompt"] += int(meta.get("input_tokens") or 0)
        _token_bucket["completion"] += int(meta.get("output_tokens") or 0)
        _token_bucket["total"] += int(meta.get("total_tokens") or 0)
        _token_bucket["calls"] += 1
        return r

    ChatGoogleGenerativeAI.invoke = wrapped
    ChatGoogleGenerativeAI._bench_patched = True


FRAMEWORK_CBS = {
    "langgraph": _install_langchain_callback,
    "crewai": _install_litellm_callback,
    "dspy": _install_litellm_callback,
}


def run_framework(name, module, install_cb, tickets, log_every=5, ticket_timeout_s=90):
    install_cb()
    rows = []
    signal.signal(signal.SIGALRM, _alarm_handler)
    for i, t in enumerate(tickets, 1):
        _reset_bucket()
        start = time.perf_counter()
        row = {
            "ticket_id": t["ticket_id"],
            "meta": t.get("meta", {}),
        }
        try:
            signal.alarm(ticket_timeout_s)
            out = module.run(t)
            signal.alarm(0)
            elapsed = time.perf_counter() - start
            row.update({
                "latency_s": round(elapsed, 3),
                "total_tokens": _token_bucket["total"],
                "prompt_tokens": _token_bucket["prompt"],
                "completion_tokens": _token_bucket["completion"],
                "llm_calls": _token_bucket["calls"],
                "reply_len": len(out.get("draft_reply") or ""),
                "approved": out.get("approved"),
                "status": out.get("status"),
            })
        except TicketTimeout:
            signal.alarm(0)
            elapsed = time.perf_counter() - start
            row.update({
                "latency_s": round(elapsed, 3),
                "total_tokens": _token_bucket["total"],
                "prompt_tokens": _token_bucket["prompt"],
                "completion_tokens": _token_bucket["completion"],
                "llm_calls": _token_bucket["calls"],
                "error": f"TicketTimeout: exceeded {ticket_timeout_s}s",
                "timed_out": True,
            })
        except Exception as e:
            signal.alarm(0)
            elapsed = time.perf_counter() - start
            row.update({
                "latency_s": round(elapsed, 3),
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "traceback": traceback.format_exc()[:500],
            })
        rows.append(row)
        if i % log_every == 0:
            dt = row.get("latency_s")
            err = row.get("error", "")
            print(f"  [{name}] {i}/{len(tickets)} last={dt}s {err}", flush=True)
    return rows


def summarize(rows):
    ok = [r for r in rows if "error" not in r]
    timed_out = [r for r in rows if r.get("timed_out")]
    other_errors = [r for r in rows if "error" in r and not r.get("timed_out")]
    if not ok:
        return {"n": 0, "errors": len(rows), "timed_out": len(timed_out), "other_errors": len(other_errors)}
    lat = [r["latency_s"] for r in ok]
    tok = [r["total_tokens"] for r in ok]
    calls = [r["llm_calls"] for r in ok]
    prompt_tok = [r["prompt_tokens"] for r in ok]
    comp_tok = [r["completion_tokens"] for r in ok]
    lat_sorted = sorted(lat)

    def pct(data, p):
        idx = min(len(data) - 1, max(0, int(round(p * (len(data) - 1)))))
        return round(sorted(data)[idx], 3)

    return {
        "n": len(ok),
        "errors": len(rows) - len(ok),
        "timed_out": len(timed_out),
        "other_errors": len(other_errors),
        "mean_latency_s": round(statistics.mean(lat), 3),
        "median_latency_s": round(statistics.median(lat), 3),
        "p50_latency_s": pct(lat, 0.50),
        "p90_latency_s": pct(lat, 0.90),
        "p95_latency_s": pct(lat, 0.95),
        "p99_latency_s": pct(lat, 0.99),
        "stdev_latency_s": round(statistics.pstdev(lat), 3),
        "total_tokens": sum(tok),
        "total_prompt_tokens": sum(prompt_tok),
        "total_completion_tokens": sum(comp_tok),
        "mean_tokens_per_ticket": round(statistics.mean(tok), 1),
        "median_tokens_per_ticket": round(statistics.median(tok), 1),
        "stdev_tokens_per_ticket": round(statistics.pstdev(tok), 1),
        "total_llm_calls": sum(calls),
        "mean_calls_per_ticket": round(statistics.mean(calls), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--tickets", default=str(HERE / "tickets_100.json"))
    ap.add_argument("--frameworks", default="langgraph,crewai,dspy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Cap tickets for debug runs")
    args = ap.parse_args()

    os.environ["GEMINI_MODEL"] = args.model
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    tickets = json.loads(Path(args.tickets).read_text())
    if args.limit:
        tickets = tickets[: args.limit]
    print(f"Model: {args.model}  Tickets: {len(tickets)}")

    frameworks = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    results = []
    t_start_all = time.time()
    for fw in frameworks:
        print(f"\n=== {fw} ===", flush=True)
        if fw == "langgraph":
            import impl_langgraph as mod
        elif fw == "crewai":
            import impl_crewai as mod
        elif fw == "dspy":
            import impl_dspy as mod
        else:
            print(f"unknown framework {fw}")
            continue
        t0 = time.time()
        rows = run_framework(fw, mod, FRAMEWORK_CBS[fw], tickets)
        wall = round(time.time() - t0, 1)
        print(f"  [{fw}] wall={wall}s", flush=True)
        results.append({
            "framework": fw,
            "model": args.model,
            "wall_seconds": wall,
            "summary": summarize(rows),
            "rows": rows,
        })

    out_path = Path(args.out) if args.out else (
        HERE.parent / "results" / f"bench_{args.model.replace('/', '_')}.json"
    )
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nTotal wall: {round(time.time() - t_start_all, 1)}s")
    print(f"Written: {out_path}")
    for r in results:
        print(f"  {r['framework']}: {r['summary']}")


if __name__ == "__main__":
    main()
