"""
Phase C: The Aggregator
Receives 4 expert action plans and synthesizes them into one coherent,
coordinated final report — with streaming output for real-time display.
"""

import json
from typing import Any, List

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
    client: Any,
    model: str,
    provider: str = "openrouter",
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

    if provider == "anthropic":
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=(
                f"USER DNA:\n{json.dumps(dna_clean, indent=2)}\n\n{AGGREGATOR_SYSTEM}"
            ),
            messages=[{"role": "user", "content": user_message}],
        )

        text = "\n".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()
        for i in range(0, len(text), 120):
            text_chunk = text[i:i + 120]
            print(text_chunk, end="", flush=True)
            full_text.append(text_chunk)

        usage = getattr(response, "usage", None)
        usage_stats = {
            "input_tokens": usage.input_tokens if usage else "—",
            "output_tokens": usage.output_tokens if usage else "—",
            "cache_read": 0,
            "cache_created": 0,
        }
    else:
        stream = await client.chat.completions.create(
            model=model,
            max_tokens=4096,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"USER DNA:\n{json.dumps(dna_clean, indent=2)}\n\n{AGGREGATOR_SYSTEM}"
                    ),
                },
                {"role": "user", "content": user_message},
            ],
        )

        final_usage = None
        async for chunk in stream:
            if chunk.usage is not None:
                final_usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text_chunk = delta.content or ""
            if text_chunk:
                print(text_chunk, end="", flush=True)
                full_text.append(text_chunk)

        if final_usage is not None:
            prompt_details = getattr(final_usage, "prompt_tokens_details", None)
            usage_stats = {
                "input_tokens": final_usage.prompt_tokens,
                "output_tokens": final_usage.completion_tokens,
                "cache_read": getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0,
                "cache_created": getattr(prompt_details, "cache_write_tokens", 0) if prompt_details else 0,
            }
        else:
            usage_stats = {
                "input_tokens": "—",
                "output_tokens": "—",
                "cache_read": "—",
                "cache_created": "—",
            }

    return "".join(full_text), usage_stats
