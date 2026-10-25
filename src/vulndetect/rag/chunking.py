"""Sentence-aware chunking of CWE documents (paper §3) with a per-chunk CWE prefix."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

# Preload nltk's dependencies so its import hook does not block them (venv lies under the CWD).
import click  # noqa: F401
import joblib  # noqa: F401
import regex  # noqa: F401
import tqdm  # noqa: F401

from vulndetect.rag.catalog import CweEntry
from vulndetect.rag.corpus import chunk_prefix, render_body

Tokenizer = Callable[[str], list]


@dataclass(frozen=True)
class Chunk:
    """One indexed unit: identity prefix + a sentence-aligned slice of the body."""

    cwe_id: str
    cwe_name: str
    chunk_index: int
    text: str
    n_tokens: int


def default_tokenizer() -> Tokenizer:
    """Return llama-index-core's default token measure (tiktoken cl100k_base)."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base").encode


class CweChunker:
    """Split CWE entries into prefixed, sentence-aligned chunks within a token budget."""

    def __init__(
        self, chunk_size: int, chunk_overlap: int, tokenizer: Tokenizer | None = None
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenize = tokenizer if tokenizer is not None else default_tokenizer()
        self._splitters: dict[int, object] = {}

    def _splitter(self, effective_size: int):
        if effective_size not in self._splitters:
            from llama_index.core.node_parser import SentenceSplitter

            self._splitters[effective_size] = SentenceSplitter(
                chunk_size=effective_size,
                chunk_overlap=self.chunk_overlap,
                tokenizer=self._tokenize,
            )
        return self._splitters[effective_size]

    def chunk_entry(self, entry: CweEntry) -> list[Chunk]:
        """Split one CWE entry into prefixed chunks that fit the chunk size."""
        body = render_body(entry)
        if not body.strip():
            return []
        prefix = chunk_prefix(entry)
        effective_size = self.chunk_size - len(self._tokenize(prefix))
        if effective_size <= self.chunk_overlap:
            raise ValueError(
                f"{entry.cwe_id}: chunk_size {self.chunk_size} leaves no room after the "
                f"prefix ({prefix!r}) — effective size {effective_size} must exceed "
                f"chunk_overlap {self.chunk_overlap}"
            )
        chunks: list[Chunk] = []
        for piece in self._splitter(effective_size).split_text(body):
            if not piece.strip():
                continue
            text = prefix + piece
            chunks.append(
                Chunk(
                    cwe_id=entry.cwe_id,
                    cwe_name=entry.name,
                    chunk_index=len(chunks),
                    text=text,
                    n_tokens=len(self._tokenize(text)),
                )
            )
        return chunks

    def chunk_catalog(self, entries: Iterable[CweEntry]) -> list[Chunk]:
        """Chunk all given CWE entries into one flat list."""
        out: list[Chunk] = []
        for entry in entries:
            out.extend(self.chunk_entry(entry))
        return out
