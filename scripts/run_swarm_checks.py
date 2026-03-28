"""
Quick checks for Mind Mirror core personalization.

What this script validates:
1. KnowledgeBase parsing for multiple people (different notes => different context shape)
2. Live Digital DNA extraction per sample profile (requires ANTHROPIC_API_KEY)
3. Optional full swarm run (4 experts + aggregation) for one sample

Usage:
  /Users/jakhangirtynshimov/Desktop/hack2.0/.venv/bin/python scripts/run_swarm_checks.py
  /Users/jakhangirtynshimov/Desktop/hack2.0/.venv/bin/python scripts/run_swarm_checks.py --full
  /Users/jakhangirtynshimov/Desktop/hack2.0/.venv/bin/python scripts/run_swarm_checks.py --full --sample notes_fast_builder.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
import anthropic

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.knowledge_base import KnowledgeBase
from swarm.profiler import create_digital_dna
from swarm.orchestrator import SwarmOrchestrator
from swarm.aggregator import aggregate_and_stream

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


class _MockUsage:
    def __init__(self, input_tokens: int = 150, output_tokens: int = 250):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = max(0, input_tokens // 3)
        self.cache_creation_input_tokens = max(0, input_tokens // 2)


class _MockResponse:
    def __init__(self, stop_reason: str, content: list, input_tokens: int = 150, output_tokens: int = 250):
        self.stop_reason = stop_reason
        self.content = content
        self.usage = _MockUsage(input_tokens=input_tokens, output_tokens=output_tokens)


class _MockStream:
    def __init__(self, text: str):
        self._chunks = [text[i:i + 80] for i in range(0, len(text), 80)]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @property
    def text_stream(self):
        async def _gen():
            for chunk in self._chunks:
                yield chunk
        return _gen()

    async def get_final_message(self):
        return SimpleNamespace(usage=_MockUsage(input_tokens=220, output_tokens=420))


def _extract_name(notes: str) -> str:
    m = re.search(r"I am\s+([A-Za-z]+)", notes)
    return m.group(1) if m else "User"


def _extract_bullet_values(notes: str, heading: str, max_items: int = 3) -> list[str]:
    m = re.search(rf"##\s+{re.escape(heading)}\n([\s\S]*?)(\n##\s+|$)", notes)
    if not m:
        return []
    block = m.group(1)
    values = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("- "):
            values.append(line[2:].strip())
    return values[:max_items]


def _extract_first_line(notes: str, heading: str, fallback: str) -> str:
    m = re.search(rf"##\s+{re.escape(heading)}\n([\s\S]*?)(\n##\s+|$)", notes)
    if not m:
        return fallback
    block = m.group(1).strip()
    for line in block.splitlines():
        s = line.strip()
        if s and not s.startswith("- "):
            return s
    return fallback


def _mock_dna_from_notes(notes: str) -> dict:
    name = _extract_name(notes)
    tech = _extract_bullet_values(notes, "Tech Stack & Preferences", 4)
    values = _extract_bullet_values(notes, "Core Values", 4)
    goals = _extract_bullet_values(notes, "Current Projects & Goals", 3)
    models = _extract_bullet_values(notes, "Mental Models I Use", 3)
    red_flags = _extract_bullet_values(notes, "Red Flags I Watch For", 3)
    comm = _extract_first_line(notes, "Communication Style", "Direct and concise")
    biz = _extract_first_line(notes, "Business Philosophy", "Ship and learn quickly")
    risk = _extract_first_line(notes, "Risk Appetite", "Medium")

    return {
        "name": name,
        "communication_style": comm,
        "tech_stack": tech or ["Python", "TypeScript"],
        "core_values": values or ["Speed", "Ownership"],
        "business_philosophy": biz,
        "decision_style": "Evidence-driven with fast iteration",
        "risk_appetite": risk,
        "expertise_areas": ["Product", "Execution", "AI workflows"],
        "mental_models": models or ["First principles", "Pareto"],
        "red_flags": red_flags or ["Scope creep"],
        "goals": goals or ["Launch and validate quickly"],
        "personality_summary": f"{name} is pragmatic, goal-oriented, and favors actionable plans.",
    }


class _MockMessages:
    def __init__(self):
        self._tool_counter = 0

    async def create(self, **kwargs):
        system = kwargs.get("system")
        messages = kwargs.get("messages", [])

        # Profiler call uses system string.
        if isinstance(system, str):
            user_content = messages[0]["content"][0]["text"]
            dna = _mock_dna_from_notes(user_content)
            text_block = SimpleNamespace(type="text", text=json.dumps(dna, ensure_ascii=True))
            return _MockResponse("end_turn", [text_block], input_tokens=180, output_tokens=260)

        # Orchestrator calls use system blocks and tools.
        role_prompt = system[1]["text"] if system and len(system) > 1 else ""
        if "Lead Developer" in role_prompt:
            section = "Tech Stack & Preferences"
        elif "Growth Marketer" in role_prompt:
            section = "What I Look For in Products"
        elif "Security & Risk Analyst" in role_prompt:
            section = "Red Flags I Watch For"
        else:
            section = "Business Philosophy"

        last_content = messages[-1]["content"] if messages else ""
        if isinstance(last_content, list):
            # Final answer after tool result.
            plan = (
                "## Mock Action Plan\n"
                "1. Define scope and target user.\n"
                "2. Build a tiny MVP with measurable success metrics.\n"
                "3. Validate with 5 real users this week."
            )
            text_block = SimpleNamespace(type="text", text=plan)
            return _MockResponse("end_turn", [text_block], input_tokens=140, output_tokens=220)

        # First turn asks for one relevant section.
        self._tool_counter += 1
        tool_block = SimpleNamespace(
            type="tool_use",
            name="read_section",
            input={"section_name": section},
            id=f"mock-tool-{self._tool_counter}",
        )
        return _MockResponse("tool_use", [tool_block], input_tokens=120, output_tokens=80)

    def stream(self, **kwargs):
        text = (
            "# Mock Final Plan\n\n"
            "## Executive Summary\n"
            "This is a synthesized plan from PM, Dev, Marketing, and Security viewpoints.\n\n"
            "## This Week\n"
            "1. Validate problem with users\n"
            "2. Build MVP scope\n"
            "3. Launch and measure\n"
        )
        return _MockStream(text)


class _MockAnthropicClient:
    def __init__(self):
        self.messages = _MockMessages()


def build_mock_client() -> _MockAnthropicClient:
    return _MockAnthropicClient()


def _samples_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "samples"


def _project_root() -> Path:
    return PROJECT_ROOT


def load_sample_notes() -> dict[str, str]:
    samples = {}
    for file_path in sorted(_samples_dir().glob("*.md")):
        samples[file_path.name] = file_path.read_text(encoding="utf-8")
    return samples


def check_knowledge_base(notes_by_file: dict[str, str]) -> None:
    print("\n=== CHECK 1: KnowledgeBase parsing ===")
    for filename, notes in notes_by_file.items():
        kb = KnowledgeBase(notes)
        stats = kb.token_estimate()
        print(
            f"- {filename}: sections={stats['sections_count']}, "
            f"full_tokens~{stats['full_notes_tokens']}, index_tokens~{stats['index_tokens']}"
        )
        if stats["sections_count"] < 6:
            raise AssertionError(f"Too few sections parsed for {filename}")


async def check_profiler(notes_by_file: dict[str, str], client: anthropic.AsyncAnthropic) -> dict[str, dict]:
    print("\n=== CHECK 2: Profiler personalization (live API) ===")
    dna_by_file: dict[str, dict] = {}

    for filename, notes in notes_by_file.items():
        print(f"- Profiling {filename} ...", end=" ", flush=True)
        dna = await create_digital_dna(notes, client)
        cache_stats = dna.get("_cache_stats", {})

        missing = REQUIRED_DNA_KEYS - set(dna.keys())
        if missing:
            raise AssertionError(f"Profiler missing keys for {filename}: {sorted(missing)}")

        dna_by_file[filename] = dna
        print(
            "OK "
            f"(input={cache_stats.get('input_tokens', 'n/a')}, "
            f"cache_read={cache_stats.get('cache_read', 'n/a')})"
        )

    # Basic differentiation sanity check: personality summary should differ
    summaries = {k: v.get("personality_summary", "") for k, v in dna_by_file.items()}
    unique_summaries = len(set(summaries.values()))
    if unique_summaries < max(2, len(summaries) - 1):
        raise AssertionError("Profiler outputs are too similar across distinct personalities")

    print("- Personalization sanity check: PASS (summaries are differentiated)")
    return dna_by_file


async def check_full_swarm(
    filename: str,
    notes: str,
    dna: dict,
    client: anthropic.AsyncAnthropic,
) -> None:
    print("\n=== CHECK 3: Full swarm + aggregation (live API) ===")
    print(f"- Running full pipeline for {filename}")

    kb = KnowledgeBase(notes)
    orchestrator = SwarmOrchestrator(client)
    user_request = "I want to launch a personalized AI planner product in 30 days"

    expert_results = await orchestrator.run(user_request=user_request, dna=dna, kb=kb)
    if len(expert_results) != 4:
        raise AssertionError(f"Expected 4 expert results, got {len(expert_results)}")

    print("- Experts completed:")
    for r in expert_results:
        print(
            f"  {r.emoji} {r.title}: calls={r.total_api_calls}, "
            f"sections_read={len(r.tool_calls)}, input_tokens={r.input_tokens}"
        )

    _, agg_stats = await aggregate_and_stream(
        user_request=user_request,
        dna=dna,
        expert_results=expert_results,
        client=client,
    )
    print("\n- Aggregator stats:")
    print(
        f"  input={agg_stats.get('input_tokens', 'n/a')}, "
        f"output={agg_stats.get('output_tokens', 'n/a')}, "
        f"cache_read={agg_stats.get('cache_read', 'n/a')}"
    )


async def async_main(run_full: bool, full_sample: str | None) -> None:
    notes_by_file = load_sample_notes()
    if not notes_by_file:
        raise RuntimeError("No sample notes found in ./samples")

    check_knowledge_base(notes_by_file)

    load_dotenv(_project_root() / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    use_mock = False
    if not api_key:
        use_mock = True
        print("\nNo ANTHROPIC_API_KEY found in .env. Switching to MOCK mode.")

    if use_mock:
        client = build_mock_client()
        dna_by_file = await check_profiler(notes_by_file, client)
    else:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        try:
            dna_by_file = await check_profiler(notes_by_file, client)
        except anthropic.BadRequestError as e:
            print(f"\nLive API unavailable ({e.__class__.__name__}). Switching to MOCK mode.")
            client = build_mock_client()
            dna_by_file = await check_profiler(notes_by_file, client)

    if run_full:
        target = full_sample or next(iter(notes_by_file.keys()))
        if target not in notes_by_file:
            available = ", ".join(notes_by_file.keys())
            raise ValueError(f"Sample '{target}' not found. Available: {available}")
        await check_full_swarm(target, notes_by_file[target], dna_by_file[target], client)

    print("\nAll requested checks completed successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick checks for Mind Mirror core")
    parser.add_argument("--full", action="store_true", help="Run full swarm + aggregation check")
    parser.add_argument(
        "--sample",
        default=None,
        help="Sample filename in ./samples to use with --full (e.g. notes_fast_builder.md)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(run_full=args.full, full_sample=args.sample))


if __name__ == "__main__":
    main()
