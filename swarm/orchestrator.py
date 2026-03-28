"""
Phase B: The Swarm Orchestrator — JIT Context Edition
=======================================================
Each expert agent receives:
  1. A minimal INDEX of the knowledge base (~300 tokens) — NOT the full notes
  2. A `read_section` tool to fetch specific sections on demand
  3. The user's Digital DNA (cached, shared)

The agent decides which 1-3 sections it actually needs, fetches them,
then produces its action plan. Total context per agent: ~700-900 tokens
instead of 3,000-5,000 with full RAG injection.

Prompt Caching Strategy:
  System Block 1: DNA + Knowledge Index  →  cache_control (shared across all 4 agents)
  System Block 2: Role-specific prompt   →  no cache (differs per agent)

All 4 agents still run in PARALLEL via asyncio.gather.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import List

import anthropic

from rag.knowledge_base import KnowledgeBase, TOOL_READ_SECTION


MAX_TOOL_CALLS = 3  # Per agent — prevents runaway loops


EXPERT_ROLES = [
    {
        "id": "product_manager",
        "emoji": "📋",
        "title": "Product Manager",
        "focus": (
            "Define the product vision, MVP feature set, user stories, "
            "success metrics, and a 30/60/90-day roadmap. "
            "Apply Jobs-to-be-Done. Identify the key user personas and their pain points."
        ),
        "suggested_sections": ["About Me", "Goals", "Business Philosophy", "What I Look For in Products"],
    },
    {
        "id": "lead_developer",
        "emoji": "⚙️",
        "title": "Lead Developer",
        "focus": (
            "Design the technical architecture, choose the tech stack (respect user preferences), "
            "break implementation into sprints, identify technical risks, "
            "estimate complexity, and outline the data model."
        ),
        "suggested_sections": ["Tech Stack & Preferences", "Mental Models I Use", "Lessons Learned"],
    },
    {
        "id": "growth_marketer",
        "emoji": "📣",
        "title": "Growth Marketer",
        "focus": (
            "Create the go-to-market strategy, identify acquisition channels, "
            "write the value proposition, define pricing, "
            "outline the launch sequence, and identify early adopter communities."
        ),
        "suggested_sections": ["Business Philosophy", "Core Values", "What I Look For in Products"],
    },
    {
        "id": "security_analyst",
        "emoji": "🔒",
        "title": "Security & Risk Analyst",
        "focus": (
            "Perform threat modeling, identify top 5 security risks with mitigations, "
            "flag compliance requirements, define security architecture, "
            "and list business risks with mitigation plans."
        ),
        "suggested_sections": ["Red Flags I Watch For", "Risk Appetite", "Lessons Learned"],
    },
]


def _build_role_system_prompt(role: dict, dna: dict) -> str:
    """Build a concise role-specific system prompt using DNA."""
    dna_clean = {k: v for k, v in dna.items() if not k.startswith("_")}
    comm_style = dna_clean.get("communication_style", "direct and technical")
    risk = dna_clean.get("risk_appetite", "high")
    values = ", ".join(dna_clean.get("core_values", []))
    tech_stack = ", ".join(dna_clean.get("tech_stack", []))
    philosophy = dna_clean.get("business_philosophy", "ship fast, learn faster")
    suggested = ", ".join(f'"{s}"' for s in role.get("suggested_sections", []))

    return f"""You are a world-class {role['title']} — part of the "Mind Mirror Swarm".

YOUR ROLE: {role['focus']}

COMMUNICATE LIKE THIS USER:
- Style: {comm_style}
- Values: {values}
- Philosophy: {philosophy}
- Tech preferences: {tech_stack}
- Risk appetite: {risk}

WORKFLOW:
1. Use the `read_section` tool to fetch 1-3 sections you need (suggested: {suggested})
2. Only read sections DIRECTLY relevant to your role
3. Then write your expert action plan

RULES:
- Be concrete and actionable — no vague advice
- Reference their tech stack when recommending solutions
- Flag anything conflicting with their values or red flags
- Use markdown headers for structure

Title your response: ## {role['emoji']} {role['title']} Action Plan"""


@dataclass
class ToolCall:
    section_name: str
    result: str


@dataclass
class ExpertResult:
    role_id: str
    title: str
    emoji: str
    content: str
    tool_calls: List[ToolCall]
    input_tokens: int
    cache_read_tokens: int
    cache_created_tokens: int
    total_api_calls: int
    duration_ms: float


async def _run_expert_jit(
    role: dict,
    user_request: str,
    dna: dict,
    kb: KnowledgeBase,
    client: anthropic.AsyncAnthropic,
) -> ExpertResult:
    """
    Run one expert agent with JIT context loading.

    Loop:
      1. First call: agent sees index + role prompt, decides what to read
      2. If tool_use: execute read_section, send result back
      3. Repeat up to MAX_TOOL_CALLS times
      4. Agent produces final plan
    """
    dna_clean = {k: v for k, v in dna.items() if not k.startswith("_")}

    # ── Shared cached prefix (same for ALL 4 agents) ──────────────────────────
    # Only send the index, NOT the full notes. This is the JIT key insight.
    shared_context = (
        "=== USER'S DIGITAL DNA ===\n"
        f"{json.dumps(dna_clean, indent=2)}\n\n"
        f"{kb.get_index()}"
    )

    system_blocks = [
        # Block 1: Shared stable content — CACHED, read from cache on agents 2-4
        {
            "type": "text",
            "text": shared_context,
            "cache_control": {"type": "ephemeral"},
        },
        # Block 2: Role-specific — NOT cached (different per agent, short)
        {
            "type": "text",
            "text": _build_role_system_prompt(role, dna),
        },
    ]

    messages = [
        {
            "role": "user",
            "content": (
                f'The user wants to build: "{user_request}"\n\n'
                "Use `read_section` to fetch relevant context, then write your action plan."
            ),
        }
    ]

    start = time.monotonic()
    tool_calls_made: List[ToolCall] = []
    total_input_tokens = 0
    total_cache_read = 0
    total_cache_created = 0
    api_call_count = 0

    # ── JIT tool-use loop ─────────────────────────────────────────────────────
    for _ in range(MAX_TOOL_CALLS + 1):  # +1 for final plan call
        api_call_count += 1
        response = await client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            system=system_blocks,
            tools=[TOOL_READ_SECTION],
            tool_choice={"type": "auto"},
            messages=messages,
        )

        total_input_tokens += response.usage.input_tokens
        total_cache_read += response.usage.cache_read_input_tokens
        total_cache_created += response.usage.cache_creation_input_tokens

        # Done — no more tool calls
        if response.stop_reason == "end_turn":
            text = next((b.text for b in response.content if b.type == "text"), "")
            break

        # Handle tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "read_section":
                    section_name = block.input.get("section_name", "")
                    result = kb.read_section(section_name)
                    tool_calls_made.append(ToolCall(section_name=section_name, result=result))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            # Append assistant response + tool results to conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason — take whatever text is there
            text = next((b.text for b in response.content if b.type == "text"), "")
            break
    else:
        # Hit loop limit — extract whatever text is present
        text = next((b.text for b in response.content if b.type == "text"), "")

    duration_ms = (time.monotonic() - start) * 1000

    return ExpertResult(
        role_id=role["id"],
        title=role["title"],
        emoji=role["emoji"],
        content=text,
        tool_calls=tool_calls_made,
        input_tokens=total_input_tokens,
        cache_read_tokens=total_cache_read,
        cache_created_tokens=total_cache_created,
        total_api_calls=api_call_count,
        duration_ms=duration_ms,
    )


class SwarmOrchestrator:
    """
    Dispatches 4 expert agents in parallel with JIT context loading.
    Each agent fetches only the knowledge sections it needs.
    """

    def __init__(self, client: anthropic.AsyncAnthropic):
        self.client = client

    async def run(
        self,
        user_request: str,
        dna: dict,
        kb: KnowledgeBase,
    ) -> List[ExpertResult]:
        """Launch all 4 experts simultaneously and collect results."""
        tasks = [
            _run_expert_jit(role, user_request, dna, kb, self.client)
            for role in EXPERT_ROLES
        ]
        results = await asyncio.gather(*tasks)
        return list(results)
