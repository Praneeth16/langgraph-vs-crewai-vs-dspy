"""CrewAI refund assistant. Role-first, OSS-compatible (no @human_feedback enterprise decorator)."""
from __future__ import annotations

import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel


class RefundState(BaseModel):
    ticket_id: str = ""
    email_text: str = ""
    policy_summary: str = "Damaged items refundable within 30 days."
    requested_refund: float = 0.0
    triage: str = ""
    draft_reply: str = ""
    approved: bool | None = None
    status: str = ""


def _llm():
    return LLM(
        model=f"gemini/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}",
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )


def build_agents():
    llm = _llm()
    policy = Agent(
        role="Refund Policy Analyst",
        goal="Decide eligibility from ticket and policy",
        backstory="You know damage clauses and return windows.",
        llm=llm,
        verbose=False,
    )
    writer = Agent(
        role="Customer Support Writer",
        goal="Draft a concise policy-aligned reply",
        backstory="You write short support emails.",
        llm=llm,
        verbose=False,
    )
    return policy, writer


class RefundFlow(Flow[RefundState]):
    @start()
    def ingest(self):
        pass

    @listen(ingest)
    def run_crew(self):
        policy, writer = build_agents()
        triage_task = Task(
            description=(
                "Ticket: {email}\nPolicy: {policy}\n"
                "Reply with one short line: 'eligible: yes|no. reason: ...'"
            ),
            expected_output="One line verdict.",
            agent=policy,
        )
        draft_task = Task(
            description=(
                "Using the triage and ticket, write a concise customer email "
                "(3 sentences max, no sign-off).\nTicket: {email}"
            ),
            expected_output="A short email body.",
            agent=writer,
            context=[triage_task],
        )
        crew = Crew(
            agents=[policy, writer],
            tasks=[triage_task, draft_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff(inputs={
            "email": self.state.email_text,
            "policy": self.state.policy_summary,
        })
        self.state.triage = str(triage_task.output)
        self.state.draft_reply = str(result)

    @listen(run_crew)
    def finalize(self):
        if self.state.requested_refund > 100:
            self.state.approved = False
            self.state.status = "held_for_review"
        else:
            self.state.approved = True
            self.state.status = "sent"


def run(ticket):
    flow = RefundFlow()
    flow.state.ticket_id = ticket["ticket_id"]
    flow.state.email_text = ticket["email_text"]
    flow.state.requested_refund = ticket["requested_refund"]
    flow.kickoff()
    return flow.state.model_dump()
