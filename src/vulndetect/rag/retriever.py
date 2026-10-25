"""Top-N vector retrieval plus optional cross-encoder re-ranking over the CWE index."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

from vulndetect.rag.embedding import Embedder, Reranker
from vulndetect.rag.index import load_manifest, open_collection


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved CWE chunk with its vector distance and optional re-rank score."""

    text: str
    cwe_id: str
    cwe_name: str
    chunk_index: int
    vector_distance: float  # cosine distance, smaller is better
    rerank_score: float | None  # larger is better; None without re-ranker


def distinct_cwe_count(chunks: list[RetrievedChunk]) -> int:
    """Count the distinct CWEs among the retrieved chunks."""
    return len({c.cwe_id for c in chunks})


class CweRetriever:
    """Retrieve the top-k CWE text chunks for a code snippet."""

    def __init__(
        self,
        collection,
        embedder: Embedder,
        reranker: Reranker | None = None,
        manifest: dict | None = None,
    ) -> None:
        self.collection = collection
        self.embedder = embedder
        self.reranker = reranker
        self.manifest = manifest
        prefixes = ((manifest or {}).get("embedding", {}) or {}).get("prefixes") or {}
        self.query_prefix: str = prefixes.get("query", "")

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str | Path,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> CweRetriever:
        """Open a built index, rebuilding the embedder from its manifest if none is given."""
        manifest = load_manifest(persist_dir)
        if embedder is None:
            from vulndetect.rag.embedding import SentenceTransformerEmbedder

            emb = manifest["embedding"]
            embedder = SentenceTransformerEmbedder(emb["model"], revision=emb["revision"])
        return cls(open_collection(persist_dir), embedder, reranker, manifest)

    def retrieve(self, code: str, *, top_n: int, top_k: int) -> list[RetrievedChunk]:
        """Fetch ``top_n`` chunks by cosine similarity for ``code``, re-rank, return ``top_k``."""
        if not code or not code.strip():
            raise ValueError("retrieve() got an empty query")
        if top_n < 1 or top_k < 1:
            raise ValueError(f"top_n/top_k must be >= 1 (got {top_n}/{top_k})")
        if top_k > top_n:
            raise ValueError(f"top_k ({top_k}) must not exceed top_n ({top_n})")

        query_embedding = self.embedder.embed([self.query_prefix + code])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=["documents", "metadatas", "distances"],
        )
        candidates = [
            RetrievedChunk(
                text=document,
                cwe_id=metadata["cwe_id"],
                cwe_name=metadata["cwe_name"],
                chunk_index=int(metadata["chunk_index"]),
                vector_distance=float(distance),
                rerank_score=None,
            )
            for document, metadata, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
            )
        ]
        candidates.sort(key=lambda c: (c.vector_distance, c.cwe_id, c.chunk_index))

        if self.reranker is None:
            return candidates[:top_k]

        scores = self.reranker.scores(code, [c.text for c in candidates])
        reranked = [
            replace(candidate, rerank_score=score)
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda c: (-c.rerank_score, c.vector_distance, c.cwe_id, c.chunk_index))
        return reranked[:top_k]

    def random_chunks(self, k: int, rng: random.Random) -> list[RetrievedChunk]:
        """Draw ``k`` chunks uniformly from the whole index (random-context control)."""
        if not hasattr(self, "_all_ids"):
            got = self.collection.get(include=[])
            self._all_ids = sorted(got["ids"])
        ids = rng.sample(self._all_ids, k)
        got = self.collection.get(ids=ids, include=["documents", "metadatas"])
        rows = zip(got["documents"], got["metadatas"], strict=True)
        by_id = dict(zip(got["ids"], rows, strict=True))
        return [
            RetrievedChunk(
                text=by_id[i][0],
                cwe_id=by_id[i][1]["cwe_id"],
                cwe_name=by_id[i][1]["cwe_name"],
                chunk_index=int(by_id[i][1]["chunk_index"]),
                vector_distance=float("nan"),
                rerank_score=None,
            )
            for i in ids
        ]

    def query_is_truncated(self, code: str) -> bool:
        """Return True if the embedder will truncate this query."""
        limit = self.embedder.max_tokens
        return limit is not None and self.embedder.count_tokens(code) > limit
