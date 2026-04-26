"""Capture the actual messages sent to the LLM by each framework on a fixed prompt.

Uses litellm callback for crewai/dspy. Uses langchain callback for langgraph.
Writes results/messages_<framework>.txt with the rendered prompt as the model sees it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

QUESTION = "Is a mug refundable if it arrived damaged today under a 30-day damage policy? Reply with one short line."


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


captured = {"messages": None}


def _litellm_messages_cb(kwargs, completion_response, start_time, end_time):
    captured["messages"] = kwargs.get("messages")


def crewai_inspect():
    from crewai import Agent, Crew, LLM, Task, Process
    import litellm
    litellm.success_callback = [_litellm_messages_cb]
    llm = LLM(model=f"gemini/gemini-2.5-flash", api_key=os.environ["GEMINI_API_KEY"], temperature=0)
    a = Agent(role="Refund Policy Analyst", goal="Decide eligibility from ticket and policy",
              backstory="You know damage clauses and return windows.", llm=llm, verbose=False)
    t = Task(description=QUESTION, expected_output="One line.", agent=a)
    c = Crew(agents=[a], tasks=[t], process=Process.sequential, verbose=False)
    c.kickoff()
    return captured["messages"]


def dspy_inspect():
    import dspy, litellm
    litellm.success_callback = [_litellm_messages_cb]
    dspy.configure(lm=dspy.LM("gemini/gemini-2.5-flash", api_key=os.environ["GEMINI_API_KEY"], temperature=0, cache=False))

    class Q(dspy.Signature):
        """Answer the question in one short line."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    p = dspy.Predict(Q)
    p(question=QUESTION)
    return captured["messages"]


def lg_inspect():
    """LangGraph node calls langchain ChatGoogleGenerativeAI; for prompt-only inspection
    we just construct the same call and inspect the underlying request."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ["GEMINI_API_KEY"], temperature=0)
    msg = [HumanMessage(content=QUESTION)]
    # Inspect what gets sent: langchain converts to dicts.
    return [{"role": "user", "content": QUESTION}]


def main():
    load_env()
    out = {}
    print("=== langgraph ===")
    captured["messages"] = None
    out["langgraph"] = lg_inspect()
    print(json.dumps(out["langgraph"], indent=2))

    print("\n=== crewai ===")
    captured["messages"] = None
    out["crewai"] = crewai_inspect()
    print(json.dumps(out["crewai"], indent=2)[:4000])

    print("\n=== dspy ===")
    captured["messages"] = None
    out["dspy"] = dspy_inspect()
    print(json.dumps(out["dspy"], indent=2)[:4000])

    p = RESULTS / "messages_inspected.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
