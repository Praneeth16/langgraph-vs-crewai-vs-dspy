"""LangGraph refund assistant. State-first, checkpointed, interruptible."""
from __future__ import annotations

import os
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class RefundState(TypedDict, total=False):
    ticket_id: str
    email_text: str
    policy_summary: str
    requested_refund: float
    triage: str
    draft_reply: str
    approved: bool
    sent: bool


def _llm():
    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )


def _text(msg) -> str:
    """Normalize Gemini response content: Gemini 2.x returns str, Gemini 3 returns list[dict]."""
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in c
        )
    return str(c)


def node_ingest(state: RefundState):
    return {}


def node_load_policy(state: RefundState):
    return {"policy_summary": "Damaged items refundable within 30 days."}


def node_triage(state: RefundState):
    msg = (
        "You are a refund policy analyst. Given the ticket and policy, "
        "decide eligibility. Reply with one short line: "
        "'eligible: yes|no. reason: ...'\n\n"
        f"TICKET: {state['email_text']}\nPOLICY: {state['policy_summary']}"
    )
    out = _text(_llm().invoke(msg))
    return {"triage": out.strip()}


def node_draft(state: RefundState):
    msg = (
        "You are a support writer. Using the triage, write a concise "
        "customer email (3 sentences max). No sign-off.\n\n"
        f"TICKET: {state['email_text']}\nTRIAGE: {state['triage']}"
    )
    out = _text(_llm().invoke(msg))
    return {"draft_reply": out.strip()}


def node_approval(state: RefundState):
    if state["requested_refund"] <= 100:
        return Command(goto="send", update={"approved": True})
    approved = interrupt({
        "question": "Approve refund reply?",
        "ticket_id": state["ticket_id"],
        "amount": state["requested_refund"],
        "draft_reply": state["draft_reply"],
    })
    return Command(
        update={"approved": bool(approved)},
        goto="send" if approved else END,
    )


def node_send(state: RefundState):
    return {"sent": True}


def build_graph():
    g = StateGraph(RefundState)
    for name, fn in [
        ("ingest", node_ingest),
        ("load_policy", node_load_policy),
        ("triage", node_triage),
        ("draft", node_draft),
        ("approval", node_approval),
        ("send", node_send),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "load_policy")
    g.add_edge("load_policy", "triage")
    g.add_edge("triage", "draft")
    g.add_edge("draft", "approval")
    g.add_edge("send", END)
    return g.compile(checkpointer=InMemorySaver())


def run(ticket):
    graph = build_graph()
    config = {"configurable": {"thread_id": ticket["ticket_id"]}}
    result = graph.invoke(ticket, config=config)
    if result.get("sent"):
        return result
    result = graph.invoke(Command(resume=True), config=config)
    return result
