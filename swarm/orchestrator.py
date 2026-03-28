"""
Phase B: The Swarm Orchestrator
Generates 4 specialized expert system prompts from the user's Digital DNA,
then runs all 4 experts in PARALLEL using asyncio.gather.

Prompt Caching Strategy:
  - Block 1 of system: DNA + Notes context  →  cache_control (shared across all 4 agents)
  - Block 2 of system: Role-specific instructions  →  no cache (different per agent)

This means all 4 parallel API calls share the same cached prefix, paying for
the context only once and getting up to ~90% token savings on the shared prefix.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import List

import anthropic


EXPERT_ROLES = [
    {
        "id": "product_manager",
        "emoji": "📋",
        "title": "Product Manager",
        "focus": (
            "Define the product vision, core features (MVP vs V2), user stories, "
            "success metrics, and a 30/60/90-day roadmap. Apply Jobs-to-be-Done thinking. "
            "Identify the key user personas and their pain points."
        ),
    },
    {
        "id": "lead_developer",
        "emoji": "⚙️",
        "title": "Lead Developer",
        "focus": (
            "Design the technical architecture, choose the tech stack (respecting user's preferences), "
            "break down implementation into sprints, identify technical risks, "
            "estimate complexity, and outline the data model."
        ),
    },
    {
        "id": "growth_marketer",
        "emoji": "📣",
        "title": "Growth Marketer",
        "focus": (
            "Create the go-to-market strategy, identify acquisition channels, "
            "write the value proposition, define the pricing model, "
            "outline the launch sequence, and identify early adopter communities."
        ),
    },
    {
        "id": "security_analyst",
        "emoji": "🔒",
        "title": "Security & Risk Analyst",
        "focus": (
            "Perform threat modeling, identify top 5 security risks and mitigations, "
            "highlight compliance requirements (GDPR, SOC2 if relevant), "
            "define the security architecture, and flag business risks with mitigation plans."
        ),
    },
]


def _build_role_system_prompt(role: dict, dna: dict) -> str:
    """Build a role-specific system prompt personalized to the user's DNA."""
    dna_clean = {k: v for k, v in dna.items() if not k.startswith("_")}
    comm_style = dna_clean.get("communication_style", "direct and technical")
    risk = dna_clean.get("risk_appetite", "high")
    values = ", ".join(dna_clean.get("core_values", []))
    tech_stack = ", ".join(dna_clean.get("tech_stack", []))
    philosophy = dna_clean.get("business_philosophy", "ship fast, learn faster")

    return f"""You are a world-class {role['title']} operating as part of the "Mind Mirror Swarm" —
a team of AI experts that think and communicate EXACTLY like the user.

YOUR ROLE: {role['focus']}

HOW TO COMMUNICATE (mirror the user's style):
- Style: {comm_style}
- Risk appetite: {risk}
- Core values: {values}
- Philosophy: {philosophy}
- Preferred tech: {tech_stack}

CRITICAL RULES:
1. Speak as if you ARE this person's expert alter-ego — use their vocabulary and mental models
2. Be concrete and actionable — no vague advice
3. Reference their tech stack and preferences when relevant
4. Flag anything that conflicts with their red flags or values
5. Give a structured action plan with clear next steps
6. Format with markdown headers for clarity

Title your response: ## {role['emoji']} {role['title']} Action Plan"""


@dataclass
class ExpertResult:
    role_id: str
    title: str
    emoji: str
    content: str
    input_tokens: int
    cache_read_tokens: int
    duration_ms: float


async def _run_expert(
    role: dict,
    user_request: str,
    dna: dict,
    relevant_notes: str,
    client: anthropic.AsyncAnthropic,
) -> ExpertResult:
    """Run a single expert agent. Called in parallel for all 4 roles."""
    dna_clean = {k: v for k, v in dna.items() if not k.startswith("_")}

    # Shared cached context: DNA + relevant notes
    shared_context = (
        "=== USER'S DIGITAL DNA ===\n"
        f"{json.dumps(dna_clean, indent=2)}\n\n"
        "=== RELEVANT KNOWLEDGE FROM USER'S NOTES ===\n"
        f"{relevant_notes}"
    )

    start = time.monotonic()
    response = await client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=[
            # Block 1: Shared stable context — CACHED (same for all 4 agents)
            {
                "type": "text",
                "text": shared_context,
                "cache_control": {"type": "ephemeral"},
            },
            # Block 2: Role-specific instructions — NOT cached (differs per agent)
            {
                "type": "text",
                "text": _build_role_system_prompt(role, dna),
            },
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"The user wants to build the following:\n\n"
                    f'"{user_request}"\n\n'
                    f"Create your expert action plan. Be specific, personalized, and actionable."
                ),
            }
        ],
    )
    duration_ms = (time.monotonic() - start) * 1000

    text = next((b.text for b in response.content if b.type == "text"), "")
    return ExpertResult(
        role_id=role["id"],
        title=role["title"],
        emoji=role["emoji"],
        content=text,
        input_tokens=response.usage.input_tokens,
        cache_read_tokens=response.usage.cache_read_input_tokens,
        duration_ms=duration_ms,
    )


class SwarmOrchestrator:
    """
    Dispatches 4 expert agents in parallel using asyncio.gather.
    All agents share the same cached DNA+Notes prefix for token efficiency.
    """

    def __init__(self, client: anthropic.AsyncAnthropic):
        self.client = client

    async def run(
        self, user_request: str, dna: dict, relevant_notes: str
    ) -> List[ExpertResult]:
        """Launch all 4 experts simultaneously and collect results."""
        tasks = [
            _run_expert(role, user_request, dna, relevant_notes, self.client)
            for role in EXPERT_ROLES
        ]
        results = await asyncio.gather(*tasks)
        return list(results)
