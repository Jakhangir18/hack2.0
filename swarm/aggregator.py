"""
Phase C: The Aggregator
Receives 4 expert action plans and synthesizes them into one coherent,
coordinated final report — with streaming output for real-time display.
"""

import json
from typing import List

import anthropic

from .orchestrator import ExpertResult


AGGREGATOR_SYSTEM = """You are "The Aggregator" — the final synthesizer in the Mind Mirror Swarm.

You receive action plans from 4 specialized experts (PM, Dev, Marketing, Security) and your job is
to weave them into ONE unified, coherent action plan that:

1. Eliminates redundancy and contradictions between expert recommendations
2. Creates a sequenced execution plan (what to do FIRST vs later)
3. Highlights synergies between different expert recommendations
4. Produces a clear, motivating "This Week's First Steps" section
5. Matches the user's communication style — direct, technical, action-oriented

Output format:
# 🧠 Mind Mirror: Your Personalized Action Plan

## Executive Summary (3 sentences max)

## 🎯 The Opportunity
[What this could become]

## 🏗️ Unified Architecture
[The integrated technical + product vision]

## 📅 Execution Timeline
### Week 1-2: Foundation
### Month 1: MVP
### Month 2-3: Growth

## ⚡ This Week's First Steps (Priority Order)
[Concrete, numbered, actionable tasks]

## ⚠️ Top Risks & Mitigations
[From Security Analyst, condensed]

## 💡 Key Insights from the Swarm
[The most surprising/valuable insights across all 4 experts]

Be concrete. Be direct. Make it feel like the user wrote this plan themselves."""


async def aggregate_and_stream(
    user_request: str,
    dna: dict,
    expert_results: List[ExpertResult],
    client: anthropic.AsyncAnthropic,
) -> tuple[str, dict]:
    """
    Synthesize 4 expert plans into one final report with streaming output.
    Returns (full_text, usage_stats).
    """
    dna_clean = {k: v for k, v in dna.items() if not k.startswith("_")}

    # Build the expert plans block
    expert_plans_text = "\n\n".join(
        f"{'='*60}\n{r.emoji} {r.title.upper()} PLAN:\n{'='*60}\n{r.content}"
        for r in expert_results
    )

    user_message = (
        f"Original user request: \"{user_request}\"\n\n"
        f"User's Digital DNA summary:\n"
        f"- Values: {', '.join(dna_clean.get('core_values', []))}\n"
        f"- Philosophy: {dna_clean.get('business_philosophy', '')}\n"
        f"- Tech stack: {', '.join(dna_clean.get('tech_stack', []))}\n\n"
        f"Here are the 4 expert action plans:\n\n"
        f"{expert_plans_text}\n\n"
        "Synthesize these into the final unified action plan."
    )

    full_text = []
    usage_stats = {}

    async with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": json.dumps(dna_clean, indent=2),
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": AGGREGATOR_SYSTEM,
            },
        ],
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text_chunk in stream.text_stream:
            print(text_chunk, end="", flush=True)
            full_text.append(text_chunk)

        final = await stream.get_final_message()
        usage_stats = {
            "input_tokens": final.usage.input_tokens,
            "output_tokens": final.usage.output_tokens,
            "cache_read": final.usage.cache_read_input_tokens,
            "cache_created": final.usage.cache_creation_input_tokens,
        }

    return "".join(full_text), usage_stats
