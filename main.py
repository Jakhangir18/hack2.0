"""
Mind Mirror: Active Swarm Core
================================
A personalized AI brainstorming system that combines:
  - Digital DNA profiling (The Profiler)
  - Parallel expert swarm (The Orchestrator)
  - Unified synthesis with streaming (The Aggregator)

Usage:
  python main.py [--notes path/to/notes.md] [--request "your idea here"]
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
from rag.retriever import NoteRetriever

load_dotenv()

# ── Terminal colors (no external deps) ────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
BLUE   = "\033[94m"
WHITE  = "\033[97m"

def c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"

def banner():
    print(c(CYAN, """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🧠  M I N D   M I R R O R  :  A C T I V E   S W A R M    ║
║                                                              ║
║   Profiler → Swarm Orchestrator → Aggregator                 ║
║   Powered by Claude Opus 4.6 + Prompt Caching                ║
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
    print(f"  {c(DIM, label):30s} {c(color, value)}")

def separator():
    print(c(DIM, "─" * 66))

# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_pipeline(notes_path: str, user_request: str | None):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(c(RED, "✗ ANTHROPIC_API_KEY not set. Copy .env.example → .env and add your key."))
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # ── Load notes ────────────────────────────────────────────────────────────
    notes_file = Path(notes_path)
    if not notes_file.exists():
        print(c(RED, f"✗ Notes file not found: {notes_path}"))
        sys.exit(1)

    user_notes = notes_file.read_text(encoding="utf-8")
    print(c(DIM, f"  Loaded notes: {len(user_notes):,} chars from {notes_file.name}"))

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
    print(c(BOLD, "  Digital DNA Profile:"))
    for key, val in dna.items():
        if key == "personality_summary":
            print(f"    {c(CYAN, key)}: {c(DIM, str(val)[:120])}")
        elif isinstance(val, list):
            print(f"    {c(CYAN, key)}: {c(WHITE, ', '.join(str(v) for v in val[:4]))}")
        else:
            print(f"    {c(CYAN, key)}: {c(WHITE, str(val)[:80])}")

    print()
    separator()
    stat_line("Profiler input tokens:", str(cache_stats.get("input_tokens", "—")), YELLOW)
    stat_line("Cache created:", str(cache_stats.get("cache_created", "—")), BLUE)
    separator()

    # ══════════════════════════════════════════════════════════════════════════
    # RAG: Retrieve relevant notes
    # ══════════════════════════════════════════════════════════════════════════
    if not user_request:
        print()
        print(c(BOLD + WHITE, "  💭  What idea do you want to explore? "))
        user_request = input(c(CYAN, "  › ")).strip()
        if not user_request:
            print(c(RED, "  No request provided. Exiting."))
            sys.exit(0)

    retriever = NoteRetriever(user_notes)
    relevant_notes = retriever.get_relevant(user_request, top_k=5)
    print()
    print(c(DIM, f"  RAG: Retrieved {len(relevant_notes):,} chars of relevant context"))

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B: The Swarm Orchestrator → 4 parallel experts
    # ══════════════════════════════════════════════════════════════════════════
    phase_header(
        2, "The Swarm Orchestrator",
        "Launching 4 expert agents in parallel (asyncio.gather)..."
    )
    print(c(DIM, "  Experts: 📋 PM  ⚙️ Lead Dev  📣 Marketer  🔒 Security Analyst"))
    print(c(DIM, "  Shared cached context: Digital DNA + Relevant Notes"))
    print()

    t2 = time.monotonic()
    orchestrator = SwarmOrchestrator(client)
    expert_results = await orchestrator.run(user_request, dna, relevant_notes)
    t3 = time.monotonic()

    print(c(GREEN, f"  ✓ All 4 experts completed in {(t3-t2):.1f}s (parallel)"))
    print()

    total_cache_read = sum(r.cache_read_tokens for r in expert_results)
    total_input = sum(r.input_tokens for r in expert_results)

    separator()
    for r in expert_results:
        hit = "✓ cache hit" if r.cache_read_tokens > 0 else "  cache miss"
        color = GREEN if r.cache_read_tokens > 0 else YELLOW
        stat_line(
            f"{r.emoji} {r.title}:",
            f"{r.duration_ms:.0f}ms  |  input: {r.input_tokens:,}  |  {c(color, hit)}",
            WHITE
        )
    separator()
    stat_line("Total input tokens (4 agents):", f"{total_input:,}", YELLOW)
    stat_line("Total cache_read tokens:", f"{total_cache_read:,}", GREEN)
    if total_input > 0:
        savings_pct = (total_cache_read / (total_input + total_cache_read)) * 100
        stat_line("Estimated token savings:", f"{savings_pct:.0f}%", GREEN)
    separator()

    # Show individual expert plans
    print()
    print(c(BOLD + WHITE, "  Individual Expert Plans:"))
    print()
    for r in expert_results:
        print(c(BOLD + MAGENTA, f"  {'─'*60}"))
        # Show first 400 chars of each plan
        preview = r.content[:400].replace("\n", "\n  ")
        print(c(DIM, f"  {preview}..."))
        print()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE C: The Aggregator → Unified report (streaming)
    # ══════════════════════════════════════════════════════════════════════════
    phase_header(
        3, "The Aggregator",
        "Synthesizing 4 expert plans into one unified report (streaming)..."
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
    print()
    print(c(BOLD + CYAN, f"  🏁 Total pipeline time: {total_time:.1f}s"))
    print(c(DIM, f"     (Sequential equivalent: ~{total_time + (t3-t2)*3:.0f}s — "
                 f"{((t3-t2)*3/total_time*100):.0f}% faster via parallelism)"))
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
        help="Path to your personal notes file (default: demo_notes.md)"
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
