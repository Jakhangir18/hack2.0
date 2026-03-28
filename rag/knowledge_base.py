"""
JIT (Just-in-Time) Knowledge Base
===================================
Instead of dumping all notes into every agent's context, we:
  1. Give agents a lightweight INDEX of available sections (~200-300 tokens)
  2. Define a `read_section` tool
  3. Each agent selectively fetches only the 1-3 sections it actually needs

Token savings vs full context injection:
  Old: 4 agents × 3,000 tokens = 12,000 tokens
  New: 4 agents × (300 index + ~2×200 sections) = 4 × 700 = 2,800 tokens
  Savings: ~77%
"""

import re
from typing import Optional


# Tool definition passed to Claude API
TOOL_READ_SECTION = {
    "name": "read_section",
    "description": (
        "Read a specific section from the user's personal knowledge base. "
        "Use this to get detailed information about a topic before writing your plan. "
        "Only fetch sections directly relevant to your expert role. "
        "You may call this tool up to 3 times."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "section_name": {
                "type": "string",
                "description": (
                    "The exact section name from the index "
                    "(e.g., 'Tech Stack & Preferences', 'Business Philosophy')"
                ),
            }
        },
        "required": ["section_name"],
    },
}


class KnowledgeBase:
    """
    Parses personal notes into named sections.
    Provides a minimal index for JIT context loading.
    """

    def __init__(self, notes: str):
        self.sections: dict[str, str] = {}
        self._parse(notes)

    def _parse(self, notes: str) -> None:
        """Split notes into sections by markdown headers."""
        # Split on ## or ### headers
        parts = re.split(r'\n(?=#{1,3} )', notes.strip())
        for part in parts:
            part = part.strip()
            if not part:
                continue
            lines = part.split('\n', 1)
            heading = lines[0].lstrip('#').strip()
            content = lines[1].strip() if len(lines) > 1 else ""
            if heading and content:
                self.sections[heading] = content

    def get_index(self) -> str:
        """
        Returns a minimal index listing all section names with a 1-line preview.
        This is what agents receive INSTEAD of the full notes.
        Typically 200-400 tokens.
        """
        lines = ["Available sections in user's knowledge base:"]
        for heading, content in self.sections.items():
            # First non-empty line as preview, max 80 chars
            preview = next(
                (l.strip() for l in content.split('\n') if l.strip()),
                ""
            )[:80]
            lines.append(f"  • {heading}: {preview}")
        return "\n".join(lines)

    def read_section(self, section_name: str) -> str:
        """
        Returns full content of a requested section.
        Uses fuzzy matching so agents don't need exact names.
        """
        # Exact match first
        if section_name in self.sections:
            return f"## {section_name}\n\n{self.sections[section_name]}"

        # Case-insensitive partial match
        section_lower = section_name.lower()
        for heading, content in self.sections.items():
            if section_lower in heading.lower() or heading.lower() in section_lower:
                return f"## {heading}\n\n{content}"

        available = list(self.sections.keys())
        return (
            f"Section '{section_name}' not found.\n"
            f"Available sections: {', '.join(available)}"
        )

    def token_estimate(self) -> dict:
        """Rough token estimates for comparison."""
        full_text = "\n\n".join(
            f"## {h}\n{c}" for h, c in self.sections.items()
        )
        index = self.get_index()
        # ~4 chars per token rough estimate
        return {
            "full_notes_tokens": len(full_text) // 4,
            "index_tokens": len(index) // 4,
            "sections_count": len(self.sections),
        }
