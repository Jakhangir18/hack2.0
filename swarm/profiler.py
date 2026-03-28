"""
Phase A: The Profiler
Analyzes user notes and extracts a structured "Digital DNA" — a personality
config that drives personalization across all downstream agents.
"""

import json
import re
from typing import Any


PROFILER_SYSTEM = """You are "The Profiler" — an expert psychologist, personal analyst, and talent assessor.

Your mission: analyze a person's personal notes, journals, and writing to extract their "Digital DNA" —
a precise, structured personality configuration that captures their true essence.

This Digital DNA is a configuration object that will be injected into specialized AI agents to make
every response feel like it was written BY this person, FOR this person.

Be specific and concrete. Avoid generic descriptions. Extract real patterns from the text.

Return ONLY a valid JSON object with this exact structure:
{
  "name": "...",
  "communication_style": "...",
  "tech_stack": ["...", "..."],
  "core_values": ["...", "..."],
  "business_philosophy": "...",
  "decision_style": "...",
  "risk_appetite": "...",
  "expertise_areas": ["...", "..."],
  "mental_models": ["...", "..."],
  "red_flags": ["...", "..."],
  "goals": ["...", "..."],
  "personality_summary": "..."
}"""

REQUIRED_DNA_KEYS = {
    "name",
    "communication_style",
    "tech_stack",
    "core_values",
    "business_philosophy",
    "decision_style",
    "risk_appetite",
    "expertise_areas",
    "mental_models",
    "red_flags",
    "goals",
    "personality_summary",
}


def _extract_name(notes: str) -> str:
    match = re.search(r"I am\s+([A-Za-z]+)", notes)
    return match.group(1) if match else "User"


def _extract_bullet_values(notes: str, heading: str, max_items: int = 3) -> list[str]:
    match = re.search(rf"##\s+{re.escape(heading)}\n([\s\S]*?)(\n##\s+|$)", notes)
    if not match:
        return []

    values = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if line.startswith("- "):
            values.append(line[2:].strip())
    return values[:max_items]


def _extract_first_line(notes: str, heading: str, fallback: str) -> str:
    match = re.search(rf"##\s+{re.escape(heading)}\n([\s\S]*?)(\n##\s+|$)", notes)
    if not match:
        return fallback
    for line in match.group(1).splitlines():
        line = line.strip()
        if line and not line.startswith("- "):
            return line
    return fallback


def _fallback_dna(notes: str) -> dict:
    name = _extract_name(notes)
    return {
        "name": name,
        "communication_style": _extract_first_line(notes, "Communication Style", "Direct and concise"),
        "tech_stack": _extract_bullet_values(notes, "Tech Stack & Preferences", 4) or ["Python", "TypeScript"],
        "core_values": _extract_bullet_values(notes, "Core Values", 4) or ["Speed", "Ownership"],
        "business_philosophy": _extract_first_line(notes, "Business Philosophy", "Ship and learn quickly"),
        "decision_style": "Evidence-driven with fast iteration",
        "risk_appetite": _extract_first_line(notes, "Risk Appetite", "Medium"),
        "expertise_areas": ["Product", "Execution", "AI workflows"],
        "mental_models": _extract_bullet_values(notes, "Mental Models I Use", 3) or ["First principles", "Pareto"],
        "red_flags": _extract_bullet_values(notes, "Red Flags I Watch For", 3) or ["Scope creep"],
        "goals": _extract_bullet_values(notes, "Current Projects & Goals", 3) or ["Launch and validate quickly"],
        "personality_summary": f"{name} is pragmatic, goal-oriented, and favors actionable plans.",
    }


async def create_digital_dna(
    notes: str,
    client: Any,
    model: str,
    provider: str = "openrouter",
) -> dict:
    """
    Phase A: Analyze user notes and produce their Digital DNA.

    Uses adaptive thinking for deep analysis and caches the notes input
    so subsequent calls (if any) benefit from prompt caching.
    """
    if provider == "anthropic":
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=PROFILER_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Here are my personal notes and knowledge base:\n\n"
                        f"{notes}\n\n"
                        "Extract my Digital DNA. Return ONLY the JSON object, no extra text."
                    ),
                }
            ],
        )
        text = "\n".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ) or "{}"
    else:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PROFILER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Here are my personal notes and knowledge base:\n\n"
                        f"{notes}\n\n"
                        "Extract my Digital DNA. Return ONLY the JSON object, no extra text."
                    ),
                },
            ],
        )

        choices = getattr(response, "choices", None) or []
        if choices and getattr(choices[0], "message", None) is not None:
            text = choices[0].message.content or "{}"
        else:
            text = "{}"

    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        text = "{}"

    # Parse JSON — strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)

    try:
        dna = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract JSON object with regex
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                dna = json.loads(match.group())
            except json.JSONDecodeError:
                dna = {}
        else:
            dna = {}

    missing_keys = REQUIRED_DNA_KEYS - set(dna.keys())
    if missing_keys:
        fallback = _fallback_dna(notes)
        # Keep model-generated fields when present and backfill the rest.
        dna = {**fallback, **dna}

    # Attach cache usage stats for display
    usage = getattr(response, "usage", None)
    if provider == "anthropic":
        cache_read = 0
        cache_created = 0
        input_tokens = usage.input_tokens if usage else 0
    else:
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        cache_read = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
        cache_created = getattr(prompt_details, "cache_write_tokens", 0) if prompt_details else 0
        input_tokens = usage.prompt_tokens if usage else 0
    dna["_cache_stats"] = {
        "cache_created": cache_created,
        "cache_read": cache_read,
        "input_tokens": input_tokens,
    }

    return dna
