"""DSPy refund assistant. Program-first, compile-ready."""
from __future__ import annotations

import os

import dspy


def configure():
    dspy.configure(lm=dspy.LM(
        f"gemini/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}",
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
        cache=False,
    ))


class Triage(dspy.Signature):
    """Decide refund eligibility given ticket and policy."""
    email_text: str = dspy.InputField()
    policy: str = dspy.InputField()
    eligible: bool = dspy.OutputField()
    rationale: str = dspy.OutputField()


class Reply(dspy.Signature):
    """Draft a concise customer support reply (3 sentences max, no sign-off)."""
    email_text: str = dspy.InputField()
    rationale: str = dspy.InputField()
    reply: str = dspy.OutputField()


class RefundProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.triage = dspy.ChainOfThought(Triage)
        self.write = dspy.ChainOfThought(Reply)

    def forward(self, email_text: str, policy: str, requested_refund: float):
        t = self.triage(email_text=email_text, policy=policy)
        if not t.eligible:
            return dspy.Prediction(
                reply=None, rationale=t.rationale, approved=False,
                status="held_for_manual_followup",
            )
        r = self.write(email_text=email_text, rationale=t.rationale)
        approved = requested_refund <= 100
        return dspy.Prediction(
            reply=r.reply,
            rationale=t.rationale,
            approved=approved,
            status="sent" if approved else "held_for_review",
        )


_program = None


def get_program():
    global _program
    if _program is None:
        configure()
        _program = RefundProgram()
    return _program


def run(ticket):
    prog = get_program()
    pred = prog(
        email_text=ticket["email_text"],
        policy="Damaged items refundable within 30 days.",
        requested_refund=ticket["requested_refund"],
    )
    return {
        "ticket_id": ticket["ticket_id"],
        "triage": pred.rationale,
        "draft_reply": pred.reply,
        "approved": pred.approved,
        "status": pred.status,
    }
