"""
Phase A: The Profiler
Analyzes user notes and extracts a structured "Digital DNA" — a personality
config that drives personalization across all downstream agents.
"""

import json
import re
import anthropic


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


async def create_digital_dna(notes: str, client: anthropic.AsyncAnthropic) -> dict:
    """
    Phase A: Analyze user notes and produce their Digital DNA.

    Uses adaptive thinking for deep analysis and caches the notes input
    so subsequent calls (if any) benefit from prompt caching.
    """
    response = await client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=PROFILER_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here are my personal notes and knowledge base:\n\n"
                            f"{notes}\n\n"
                            "Extract my Digital DNA. Return ONLY the JSON object, no extra text."
                        ),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    )

    # Extract the text block (thinking blocks come first)
    text = next((b.text for b in response.content if b.type == "text"), "{}")

    # Parse JSON — strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)

    try:
        dna = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract JSON object with regex
        match = re.search(r"\{[\s\S]*\}", text)
        dna = json.loads(match.group()) if match else {}

    # Attach cache usage stats for display
    dna["_cache_stats"] = {
        "cache_created": response.usage.cache_creation_input_tokens,
        "cache_read": response.usage.cache_read_input_tokens,
        "input_tokens": response.usage.input_tokens,
    }

    return dna
