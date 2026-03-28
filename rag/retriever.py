import re
from typing import List, Tuple


class NoteRetriever:
    """Simple keyword-based RAG retriever over personal notes."""

    def __init__(self, notes: str):
        self.notes = notes
        self.chunks = self._chunk_notes(notes)

    def _chunk_notes(self, notes: str) -> List[str]:
        """Split notes into logical chunks by headers and paragraphs."""
        # Split on markdown headers
        sections = re.split(r'\n(?=#{1,3} )', notes)
        chunks = []
        for section in sections:
            # Further split large sections by double newlines
            sub = [s.strip() for s in section.split('\n\n') if s.strip()]
            chunks.extend(sub)
        return [c for c in chunks if len(c) > 20]

    def _score_chunk(self, chunk: str, query_words: set) -> float:
        """Score a chunk based on keyword overlap with query."""
        chunk_lower = chunk.lower()
        chunk_words = set(re.findall(r'\b\w+\b', chunk_lower))
        overlap = len(query_words & chunk_words)
        # Boost score for header matches
        header_match = 1.5 if chunk.startswith('#') else 1.0
        return overlap * header_match

    def get_relevant(self, query: str, top_k: int = 5) -> str:
        """Return the most relevant notes chunks for a given query."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))

        scored: List[Tuple[float, str]] = [
            (self._score_chunk(chunk, query_words), chunk)
            for chunk in self.chunks
        ]
        scored.sort(reverse=True, key=lambda x: x[0])

        top_chunks = [chunk for score, chunk in scored[:top_k] if score > 0]

        if not top_chunks:
            # Fallback: return first 3 chunks as general context
            top_chunks = self.chunks[:3]

        return "\n\n---\n\n".join(top_chunks)
