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
from dataclasses import dataclass
from typing import Any, List

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
    client: Any,
    model: str,
    provider: str,
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

    system_prompt = (
        f"{shared_context}\n\n"
        f"{_build_role_system_prompt(role, dna)}"
    )

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
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                tools=[TOOL_READ_SECTION],
                messages=messages,
            )

            usage = getattr(response, "usage", None)
            total_input_tokens += usage.input_tokens if usage else 0

            blocks = getattr(response, "content", None) or []
            assistant_blocks = []
            text_parts = []
            tool_uses = []

            for block in blocks:
                block_type = getattr(block, "type", "")
                if block_type == "text":
                    text_value = getattr(block, "text", "") or ""
                    if text_value:
                        text_parts.append(text_value)
                    assistant_blocks.append({"type": "text", "text": text_value})
                elif block_type == "tool_use":
                    tool_uses.append(block)
                    assistant_blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            if not tool_uses:
                text = "\n".join(part for part in text_parts if part).strip()
                break

            messages.append({"role": "assistant", "content": assistant_blocks})

            for tc in tool_uses:
                if tc.name != "read_section":
                    continue
                args = tc.input or {}
                section_name = (args.get("section_name", "") or "").strip()
                if not section_name:
                    used_sections = [t.section_name for t in tool_calls_made if t.section_name]
                    suggested_sections = role.get("suggested_sections", [])
                    section_name = next((s for s in suggested_sections if s not in used_sections), "")
                if not section_name:
                    section_name = "About Me"

                result = kb.read_section(section_name)
                tool_calls_made.append(ToolCall(section_name=section_name, result=result))
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": result,
                            }
                        ],
                    }
                )
        else:
            tool_schema = {
                "type": "function",
                "function": {
                    "name": TOOL_READ_SECTION["name"],
                    "description": TOOL_READ_SECTION["description"],
                    "parameters": TOOL_READ_SECTION["input_schema"],
                },
            }
            openai_messages = [{"role": "system", "content": system_prompt}] + messages
            response = await client.chat.completions.create(
                model=model,
                max_tokens=2048,
                tools=[tool_schema],
                tool_choice="auto",
                messages=openai_messages,
            )

            usage = getattr(response, "usage", None)
            total_input_tokens += usage.prompt_tokens if usage else 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details:
                total_cache_read += getattr(prompt_details, "cached_tokens", 0) or 0
                total_cache_created += getattr(prompt_details, "cache_write_tokens", 0) or 0

            choices = getattr(response, "choices", None) or []
            if not choices or getattr(choices[0], "message", None) is None:
                text = ""
                break
            assistant_msg = choices[0].message
            tool_calls = assistant_msg.tool_calls or []

            # Done — no more tool calls
            if not tool_calls:
                text = assistant_msg.content or ""
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                if tc.type != "function" or tc.function.name != "read_section":
                    continue
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                section_name = (args.get("section_name", "") or "").strip()
                if not section_name:
                    used_sections = [t.section_name for t in tool_calls_made if t.section_name]
                    suggested_sections = role.get("suggested_sections", [])
                    section_name = next((s for s in suggested_sections if s not in used_sections), "")
                if not section_name:
                    section_name = "About Me"
                result = kb.read_section(section_name)
                tool_calls_made.append(ToolCall(section_name=section_name, result=result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": "read_section",
                        "content": result,
                    }
                )
    else:
        # Hit loop limit without final assistant content.
        text = ""

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

    def __init__(self, client: Any, model: str, provider: str = "openrouter"):
        self.client = client
        self.model = model
        self.provider = provider

    async def run(
        self,
        user_request: str,
        dna: dict,
        kb: KnowledgeBase,
    ) -> List[ExpertResult]:
        """Launch all 4 experts simultaneously and collect results."""
        tasks = [
            _run_expert_jit(role, user_request, dna, kb, self.client, self.model, self.provider)
            for role in EXPERT_ROLES
        ]
        results = await asyncio.gather(*tasks)
        return list(results)
