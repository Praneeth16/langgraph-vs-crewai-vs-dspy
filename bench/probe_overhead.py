"""Measure the per-call prompt overhead each framework adds on top of
a minimal user request. Uses a short, fixed prompt and compares:
  - raw Gemini call (baseline)
  - LangGraph node calling the same LLM (no wrapper)
  - CrewAI Agent+Task+Crew executing a minimal prompt
  - DSPy Predict with a minimal Signature

Reports prompt_tokens, completion_tokens, total latency for a single call.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

PROMPT_QUESTION = "Is a mug refundable if it arrived damaged today under a 30-day damage policy? Reply with one short line."


def load_env():
    env = Path(__file__).parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def raw(model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=os.environ["GEMINI_API_KEY"], temperature=0)
    s = time.perf_counter()
    r = llm.invoke(PROMPT_QUESTION)
    dt = time.perf_counter() - s
    m = getattr(r, "usage_metadata", None) or {}
    return {
        "framework": "raw",
        "latency_s": round(dt, 3),
        "prompt_tokens": int(m.get("input_tokens") or 0),
        "completion_tokens": int(m.get("output_tokens") or 0),
        "total_tokens": int(m.get("total_tokens") or 0),
    }


def langgraph_one(model: str):
    """A LangGraph single-node graph that just calls the LLM."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langchain_google_genai import ChatGoogleGenerativeAI

    bucket = {"p": 0, "c": 0, "t": 0}

    class S(TypedDict, total=False):
        q: str
        a: str

    def node(state):
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=os.environ["GEMINI_API_KEY"], temperature=0)
        r = llm.invoke(state["q"])
        m = getattr(r, "usage_metadata", None) or {}
        bucket["p"] += int(m.get("input_tokens") or 0)
        bucket["c"] += int(m.get("output_tokens") or 0)
        bucket["t"] += int(m.get("total_tokens") or 0)
        c = r.content
        if isinstance(c, list):
            c = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
        return {"a": c}

    g = StateGraph(S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    graph = g.compile()
    s = time.perf_counter()
    graph.invoke({"q": PROMPT_QUESTION})
    dt = time.perf_counter() - s
    return {
        "framework": "langgraph",
        "latency_s": round(dt, 3),
        "prompt_tokens": bucket["p"],
        "completion_tokens": bucket["c"],
        "total_tokens": bucket["t"],
    }


def crewai_one(model: str):
    from crewai import Agent, Crew, LLM, Task, Process
    import litellm

    bucket = {"p": 0, "c": 0, "t": 0, "calls": 0}

    def cb(kwargs, completion_response, start_time, end_time):
        u = completion_response.get("usage") or {}
        bucket["p"] += int(u.get("prompt_tokens") or 0)
        bucket["c"] += int(u.get("completion_tokens") or 0)
        bucket["t"] += int(u.get("total_tokens") or 0)
        bucket["calls"] += 1

    litellm.success_callback = [cb]

    llm = LLM(model=f"gemini/{model}", api_key=os.environ["GEMINI_API_KEY"], temperature=0)
    a = Agent(
        role="Refund Policy Analyst",
        goal="Decide eligibility from ticket and policy",
        backstory="You know damage clauses and return windows.",
        llm=llm, verbose=False, allow_delegation=False,
    )
    t = Task(description=PROMPT_QUESTION, expected_output="One line.", agent=a)
    c = Crew(agents=[a], tasks=[t], process=Process.sequential, verbose=False)
    s = time.perf_counter()
    c.kickoff()
    dt = time.perf_counter() - s
    return {
        "framework": "crewai",
        "latency_s": round(dt, 3),
        "prompt_tokens": bucket["p"],
        "completion_tokens": bucket["c"],
        "total_tokens": bucket["t"],
        "llm_calls": bucket["calls"],
    }


def dspy_one(model: str):
    import dspy
    import litellm

    bucket = {"p": 0, "c": 0, "t": 0, "calls": 0}

    def cb(kwargs, completion_response, start_time, end_time):
        u = completion_response.get("usage") or {}
        bucket["p"] += int(u.get("prompt_tokens") or 0)
        bucket["c"] += int(u.get("completion_tokens") or 0)
        bucket["t"] += int(u.get("total_tokens") or 0)
        bucket["calls"] += 1

    litellm.success_callback = [cb]

    dspy.configure(lm=dspy.LM(f"gemini/{model}", api_key=os.environ["GEMINI_API_KEY"], temperature=0))

    class Q(dspy.Signature):
        """Answer the question in one short line."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    p = dspy.Predict(Q)
    s = time.perf_counter()
    p(question=PROMPT_QUESTION)
    dt = time.perf_counter() - s
    return {
        "framework": "dspy",
        "latency_s": round(dt, 3),
        "prompt_tokens": bucket["p"],
        "completion_tokens": bucket["c"],
        "total_tokens": bucket["t"],
        "llm_calls": bucket["calls"],
    }


def main():
    load_env()
    import json
    import sys
    results = {}
    for model in ["gemini-2.5-flash", "gemini-3.1-flash-lite-preview"]:
        print(f"\n### {model}", flush=True)
        rs = []
        for fn in [raw, langgraph_one, crewai_one, dspy_one]:
            try:
                r = fn(model)
                print(" ", r, flush=True)
                rs.append(r)
            except Exception as e:
                print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}", flush=True)
                rs.append({"framework": fn.__name__, "error": f"{type(e).__name__}: {e}"})
        results[model] = rs
    out = Path(__file__).parent.parent / "results" / "probe_overhead.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
