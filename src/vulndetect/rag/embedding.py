"""Embedder and re-ranker protocols plus sentence-transformers implementations."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Interface of a text embedding model used for the index and queries."""

    name: str
    revision: str | None
    max_tokens: int | None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count the model tokens of a text without truncation."""
        ...


class Reranker(Protocol):
    """Interface of a query-passage relevance scorer."""

    name: str
    revision: str | None

    def scores(self, query: str, texts: list[str]) -> list[float]:
        """Score the relevance of each text to the query."""
        ...


class SentenceTransformerEmbedder:
    """bge-small-en-v1.5 (or any sentence-transformers model), L2-normalized."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, revision=revision, device=device)
        self.max_tokens: int | None = int(self._model.max_seq_length)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into L2-normalized vectors."""
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def count_tokens(self, text: str) -> int:
        """Count the tokenizer's tokens for a text without truncation."""
        return len(self._model.tokenizer(text, truncation=False, verbose=False)["input_ids"])


class CrossEncoderReranker:
    """ms-marco-MiniLM-L6-v2 cross-encoder relevance scores (larger = better)."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, revision=revision, device=device)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        """Score each (query, text) pair with the cross-encoder."""
        predictions = self._model.predict(
            [(query, text) for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(p) for p in predictions]
