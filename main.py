"""
Mind Mirror: Active Swarm Core
================================
A personalized AI brainstorming system combining:
  Phase A — The Profiler:     user notes → Digital DNA
  Phase B — The Swarm:        4 parallel experts with JIT context loading
  Phase C — The Aggregator:   unified streaming report

JIT Context Strategy:
  Agents receive a ~300-token INDEX instead of full notes.
  They call `read_section` to fetch only what they need.
  Result: ~75% fewer tokens vs full context injection.

Usage:
  python main.py [--notes path/to/notes.md] [--request "your idea"]
"""

import asyncio
import json
import os
import sys
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
import anthropic

from swarm.profiler import create_digital_dna
from swarm.orchestrator import SwarmOrchestrator
from swarm.aggregator import aggregate_and_stream
from rag.knowledge_base import KnowledgeBase

load_dotenv()

# ── Terminal colors ────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
RED     = "\033[91m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"

def c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"

def banner():
    print(c(CYAN, """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🧠  M I N D   M I R R O R  :  A C T I V E   S W A R M    ║
║                                                              ║
║   Profiler → JIT Swarm Orchestrator → Aggregator             ║
║   Powered by Claude Opus 4.6 + Prompt Caching + JIT RAG      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""))

def phase_header(num: int, title: str, subtitle: str = ""):
    icons = {1: "🔬", 2: "🚀", 3: "🔗"}
    icon = icons.get(num, "•")
    print(f"\n{c(BOLD + MAGENTA, f'─── Phase {num}: {icon} {title} ───')}")
    if subtitle:
        print(c(DIM, f"    {subtitle}"))
    print()

def stat_line(label: str, value: str, color: str = WHITE):
    print(f"  {c(DIM, label):38s} {c(color, value)}")

def separator():
    print(c(DIM, "─" * 66))


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(notes_path: str, user_request: str | None):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(c(RED, "✗ ANTHROPIC_API_KEY not set. Copy .env.example → .env and add your key."))
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # ── Load notes + build knowledge base ─────────────────────────────────────
    notes_file = Path(notes_path)
    if not notes_file.exists():
        print(c(RED, f"✗ Notes file not found: {notes_path}"))
        sys.exit(1)

    user_notes = notes_file.read_text(encoding="utf-8")
    kb = KnowledgeBase(user_notes)
    stats = kb.token_estimate()

    print(c(DIM, f"  Loaded: {notes_file.name}  |  "
                 f"{stats['sections_count']} sections  |  "
                 f"~{stats['full_notes_tokens']:,} tokens full  →  "
                 f"~{stats['index_tokens']:,} tokens index"))
    print(c(GREEN, f"  JIT index saves ~{stats['full_notes_tokens'] - stats['index_tokens']:,} "
                   f"tokens per agent vs full injection"))

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE A: The Profiler → Digital DNA
    # ══════════════════════════════════════════════════════════════════════════
    phase_header(1, "The Profiler", "Analyzing your notes → building Digital DNA...")

    t0 = time.monotonic()
    dna = await create_digital_dna(user_notes, client)
    t1 = time.monotonic()

    cache_stats = dna.pop("_cache_stats", {})
    print(c(GREEN, f"  ✓ Digital DNA extracted in {(t1-t0):.1f}s"))
    print()
    print(c(BOLD, "  Your Digital DNA:"))
    for key, val in dna.items():
        if key == "personality_summary":
            print(f"    {c(CYAN, key)}: {c(DIM, str(val)[:120])}")
        elif isinstance(val, list):
            print(f"    {c(CYAN, key)}: {c(WHITE, ', '.join(str(v) for v in val[:5]))}")
        else:
            print(f"    {c(CYAN, key)}: {c(WHITE, str(val)[:90])}")

    print()
    separator()
    stat_line("Profiler input tokens:", str(cache_stats.get("input_tokens", "—")), YELLOW)
    stat_line("Cache created:", str(cache_stats.get("cache_created", "—")), BLUE)
    separator()

    # ── Get user request ──────────────────────────────────────────────────────
    if not user_request:
        print()
        print(c(BOLD + WHITE, "  💭  What idea do you want to explore?"))
        user_request = input(c(CYAN, "  › ")).strip()
        if not user_request:
            print(c(RED, "  No request provided. Exiting."))
            sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B: The Swarm Orchestrator (JIT) → 4 parallel experts
    # ══════════════════════════════════════════════════════════════════════════
    phase_header(
        2, "The Swarm Orchestrator — JIT Mode",
        "Launching 4 experts in parallel. Each agent fetches only what it needs."
    )
    print(c(DIM, "  Experts: 📋 PM  ⚙️ Lead Dev  📣 Marketer  🔒 Security"))
    print(c(DIM, "  Context per agent: ~300-token INDEX + on-demand section reads"))
    print(c(DIM, "  Shared cached prefix: DNA + Index (written once, read 3× from cache)"))
    print()

    t2 = time.monotonic()
    orchestrator = SwarmOrchestrator(client)
    expert_results = await orchestrator.run(user_request, dna, kb)
    t3 = time.monotonic()

    print(c(GREEN, f"  ✓ All 4 experts completed in {(t3-t2):.1f}s (parallel)"))
    print()

    # ── Per-expert stats ──────────────────────────────────────────────────────
    separator()
    total_input = sum(r.input_tokens for r in expert_results)
    total_cache_read = sum(r.cache_read_tokens for r in expert_results)
    total_cache_created = sum(r.cache_created_tokens for r in expert_results)
    total_api_calls = sum(r.total_api_calls for r in expert_results)

    for r in expert_results:
        sections_read = [tc.section_name for tc in r.tool_calls]
        sections_str = ", ".join(f'"{s}"' for s in sections_read) if sections_read else "none"
        cache_hit = r.cache_read_tokens > 0
        hit_str = c(GREEN, "✓ cache hit") if cache_hit else c(YELLOW, "  cache miss")

        print(f"  {r.emoji} {c(BOLD, r.title)}")
        stat_line("  API calls:", f"{r.total_api_calls}  ({len(r.tool_calls)} section reads)", DIM)
        stat_line("  Sections read:", sections_str, CYAN)
        stat_line("  Input tokens:", f"{r.input_tokens:,}  |  {hit_str}", WHITE)
        stat_line("  Duration:", f"{r.duration_ms:.0f}ms", DIM)
        print()

    separator()
    stat_line("Total input tokens (4 agents):", f"{total_input:,}", YELLOW)
    stat_line("Total cache_read tokens:", f"{total_cache_read:,}", GREEN)
    stat_line("Total API calls:", str(total_api_calls), DIM)

    # Compare vs hypothetical full-context injection
    hypothetical_full = stats['full_notes_tokens'] * 4
    if hypothetical_full > 0:
        savings = (1 - total_input / hypothetical_full) * 100
        stat_line(
            "vs full-context injection:",
            f"~{max(0, savings):.0f}% fewer tokens",
            GREEN
        )
    separator()

    # ── Expert plan previews ──────────────────────────────────────────────────
    print()
    print(c(BOLD + WHITE, "  Expert Plans Preview:"))
    for r in expert_results:
        print()
        print(c(BOLD + MAGENTA, f"  {'─'*58}"))
        preview = r.content[:350].replace("\n", "\n  ")
        print(c(DIM, f"  {preview}..."))

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE C: The Aggregator → Unified streaming report
    # ══════════════════════════════════════════════════════════════════════════
    phase_header(
        3, "The Aggregator",
        "Synthesizing 4 expert plans → unified report (streaming)..."
    )
    print()

    t4 = time.monotonic()
    final_report, agg_stats = await aggregate_and_stream(
        user_request, dna, expert_results, client
    )
    t5 = time.monotonic()

    print()
    print()
    separator()
    print(c(GREEN, f"  ✓ Aggregation complete in {(t5-t4):.1f}s"))
    stat_line("Aggregator input tokens:", str(agg_stats.get("input_tokens", "—")), YELLOW)
    stat_line("Aggregator output tokens:", str(agg_stats.get("output_tokens", "—")), WHITE)
    stat_line("Cache read:", str(agg_stats.get("cache_read", "—")), GREEN)
    separator()

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = t5 - t0
    parallel_saved = (t3 - t2) * 3  # saved by running 4 agents instead of sequential
    print()
    print(c(BOLD + CYAN, f"  🏁 Total wall-clock time: {total_time:.1f}s"))
    print(c(DIM, f"     Parallelism saved ~{parallel_saved:.0f}s vs sequential execution"))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Mind Mirror: Active Swarm Core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python main.py --request 'I want to build an AI note-taking app'"
    )
    parser.add_argument(
        "--notes",
        default="demo_notes.md",
        help="Path to your personal notes (default: demo_notes.md)"
    )
    parser.add_argument(
        "--request",
        default=None,
        help="Your idea or request (if omitted, will prompt interactively)"
    )
    args = parser.parse_args()

    banner()
    asyncio.run(run_pipeline(args.notes, args.request))


if __name__ == "__main__":
    main()
