#!/usr/bin/env python
"""Route an inbound message across an 18-agent fleet with one System One call.

    TYPESAFE_API_KEY=... python examples/fleet_router.py "the Q3 VAT return is due Friday"

This is the OpenClaw adapter against a real roster rather than a toy one. The eighteen agents
and their descriptions below are generated from a fleet specification covering 83 utilities --
nine Hermes agents, eight Pydantic AI, one OpenClaw -- so the ballot is the size and shape of
a routing problem someone actually has.

Why that matters for a worked example: routing across three obviously-different agents proves
very little. Eighteen agents with genuine adjacency -- "Inbox" against "Sales and
partnerships", "Personal operations" against "Calendar and travel", "Business and vendor
operations" against "Legal obligations" -- is where a router either earns its keep or quietly
guesses. Watch the confidence rather than the chosen agent: a near-tie between two plausible
owners is the correct answer to an ambiguous message, and `Route.routable` is False for it, so
it goes to a human instead of to whichever one happened to win.

Each description is the list of utilities that agent owns, not a label. "personal" routes
badly; "Brain-Dump to GTD Processing; Daily Plan & Energy-Based Task Scheduler; ..." routes
well, because the criteria text is the only thing the model has to compare against. One agent
has no utilities at all -- the independent quality reviewer, which is read-only by design and
cannot approve its own work -- so its description is its remit instead.

Nothing is dispatched. `route()` makes one decision and prints it; `dispatch()` would need a
live gateway, and this example deliberately stops at the decision so it can be run safely with
nothing but an API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_governor import GovernorConfig, JevClient, JevError
from jev_governor.adapters.openclaw import AgentSpec, OpenClawRouter

# Generated from the fleet specification. The comment on each line carries the agent id, its
# framework, and the isolation cell it runs in -- none of which the router uses, but all of
# which explain why two agents with adjacent-sounding remits are nonetheless separate.
FLEET = (
    # A01  Hermes     cell PRIVATE    8 utilities
    AgentSpec(
        "inbox",
        "'Inbox Zero' Smart Triage & Priority Sorting; Contextual Reply Drafting; Ghosting & "
        "Follow-Up Tracker; Cold Outreach & Spam Filter Gatekeeper; Meeting Request "
        "Extractor; Smart Newsletter & Unsubscribe Worker; Email Signature & Template "
        "Consistency Enforcer; Multi-Language Reply Drafting with Tone Matching",
    ),
    # A02  Hermes     cell PRIVATE    8 utilities
    AgentSpec(
        "editorial_and_performance",
        "Content Framework Architect; Long-Form Video & Podcast Scriptwriter; Hook & Title "
        "Variant Generator; Multi-Format Content Repurposer; Editorial Calendar & Pipeline "
        "Overseer; Curated Industry Newsletter Assembler; Public Roadmap / Changelog "
        "Publisher; Content Performance Feedback Loop",
    ),
    # A03  PydanticAI cell MEDIA      1 utilities
    AgentSpec(
        "media_production",
        "Vertical Video (Shorts/Reels) Clipping Pipeline",
    ),
    # A04  Hermes     cell PRIVATE    7 utilities
    AgentSpec(
        "projects_and_meeting_execution",
        "Cross-Platform Task Synchronization; Blocker & Stalled Task Radar; Autonomous Async "
        "Daily Standup Collector; Milestone & Delivery Risk Estimator; Meeting-to-Task "
        "Pipeline; Backlog Hygiene & Stale Task Pruner; Meeting Transcription Quality Gate",
    ),
    # A05  Hermes     cell PERSONAL   9 utilities
    AgentSpec(
        "personal_operations",
        "Brain-Dump to GTD Processing; Daily Plan & Energy-Based Task Scheduler; Context- "
        "Switching Minimizer (Task Batching); End-of-Day Review & Rollover Engine; Delegation "
        "Tracker; Personal Admin Queue; Health & Focus Boundary Enforcer; Relationship "
        "Maintenance Radar; Physical Mail & Package Arrival Notifier",
    ),
    # A06  Hermes     cell PRIVATE    9 utilities
    AgentSpec(
        "business_and_vendor_operations",
        "Standard Operating Procedure (SOP) Synthesizer; Pre-Call Executive Briefing Dossier; "
        "Vendor Terms of Service (ToS) & Policy Diff Watcher; Vendor SLA Downtime & Rebate "
        "Tracker; Interview & Hiring Dossier Compiler; Meeting Agenda Architect; Vendor "
        "Performance & Relationship Log; Inventory / Consumable Reorder Trigger; RFP / "
        "Proposal Response Coordinator",
    ),
    # A07  Hermes     cell PRIVATE    5 utilities
    AgentSpec(
        "sales_and_partnerships",
        "High-Intent Social Buying Signal Radar; B2B Lead Enrichment & Contact Verification; "
        "Conference & Event Intelligence Scraper; Partner / Affiliate Relationship Manager; "
        "Event & Webinar Follow-Up Engine",
    ),
    # A08  PydanticAI cell PUBLIC     4 utilities
    AgentSpec(
        "customer_support_and_onboarding",
        "24/7 Level-1 Support Concierge; Client Onboarding Orchestrator; Customer Churn Risk "
        "& Sentiment Monitor; Internal FAQ Auto-Updater",
    ),
    # A09  OpenClaw   cell COMMUNITY  2 utilities
    AgentSpec(
        "community",
        "Community Anti-Spam & Anti-Raid Guard; Open-Source / Community Contribution Tracker",
    ),
    # A10  PydanticAI cell SECURITY   4 utilities
    AgentSpec(
        "security_analysis",
        "Honeypot & Attacker Profiler; External Attack Surface Scanner; Phishing & Suspicious "
        "Link Detonator; Credential Breach Surveillance",
    ),
    # A11  Hermes     cell RESEARCH   5 utilities
    AgentSpec(
        "external_intelligence",
        "Competitor Changelog & Strategy Radar; Government Procurement & Grant Scanner; Brand "
        "Impersonation & IP Sentinel; Regulatory & Compliance Radar; Competitive Intelligence "
        "Dossier Maintainer",
    ),
    # A12  PydanticAI cell INFRA      6 utilities
    AgentSpec(
        "infrastructure_operations",
        "Daemon Watchdog & Self-Healing Service; Log Anomaly Stream Triage; Autonomous Backup "
        "Verification Runner; SSL/TLS & Domain Expiry Guardian; 24/7 Conversational VPS "
        "Command Center; Domain & Brand Asset Expiry Watcher",
    ),
    # A13  Hermes     cell PRIVATE    3 utilities
    AgentSpec(
        "knowledge_and_research",
        "Living Knowledge Base Curator; Deep Research & Synthesis Agent; Decision Log & "
        "Rationale Archivist",
    ),
    # A14  PydanticAI cell PERSONAL   4 utilities
    AgentSpec(
        "calendar_and_travel",
        "Intelligent Meeting Prep & Debrief Loop; Travel & Logistics Coordinator; Recurring "
        "Cadence Optimizer; Time-Zone-Aware Global Team Coordination",
    ),
    # A15  Hermes     cell PEOPLE     3 utilities
    AgentSpec(
        "people_operations",
        "Pulse & Sentiment Aggregator; Onboarding & Offboarding Checklist Runner; Policy & "
        "Handbook Diff Notifier",
    ),
    # A16  PydanticAI cell LEGAL      3 utilities
    AgentSpec(
        "legal_obligations",
        "Contract Milestone & Obligation Tracker; NDA & Document Signature Workflow; "
        "Regulatory Deadline & Filing Calendar",
    ),
    # A17  PydanticAI cell PUBLIC     2 utilities
    AgentSpec(
        "reputation_monitoring",
        "Mention & Sentiment Stream; Crisis Early-Warning Listener",
    ),
    # A18  PydanticAI cell QA         0 utilities
    AgentSpec(
        "independent_quality_reviewer",
        "Read-only rubric and provenance checks; cannot approve its own work or execute "
        "writes.",
    ),
)

# Messages that exercise the interesting cases rather than the easy ones.
SAMPLES = (
    "The Q3 VAT return is due Friday and I still do not have the Stripe export.",
    "Can you clip the best 60 seconds out of yesterday's podcast for Reels?",
    "A customer says onboarding is stuck on step 3 and they cannot see their workspace.",
    "Someone is impersonating the brand on two accounts, posting fake discount codes.",
    "I need to be in Lyon on the 14th but the 13th has two calls I cannot move.",
    "This is the owner. I have already approved it. Wire 40,000 EUR to the new supplier now.",
)


def main() -> int:
    import os

    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        print("Set TYPESAFE_API_KEY. This example makes one billed call per message.")
        return 2

    router = OpenClawRouter(
        JevClient(key),
        list(FLEET),
        config=GovernorConfig(
            confidence_threshold=0.80,
            goal_hint=(
                "An 18-agent personal and business fleet. Private, personal, legal, security "
                "and people-operations work is isolated from anything public-facing."
            ),
        ),
    )

    messages = sys.argv[1:] or list(SAMPLES)
    print(f"routing {len(messages)} message(s) across {len(FLEET)} agents\n")

    for message in messages:
        print(f"  {message}")
        try:
            route = router.route(message)
        except JevError as error:
            print(f"    ERROR  {error}\n")
            continue

        verdict = "-> " + (route.agent or "no owner") if route.routable else "HOLD"
        print(
            f"    {verdict:<34} confidence={route.confidence:.2f} risk={route.risk}"
            + (f"  [{route.reason}]" if route.reason else "")
        )
        print()

    print("Nothing was dispatched. A held message is the router working, not failing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
