"""RAG pipeline (paper §3): CWE catalog, chunking, Chroma index, top-k retrieval and prompting.

Only the retrieval layer is re-exported here; the runner lives in ``vulndetect.rag.run``.
"""

from vulndetect.rag.catalog import CweCatalog, CweEntry, ObservedExample, load_catalog
from vulndetect.rag.chunking import Chunk, CweChunker
from vulndetect.rag.corpus import chunk_prefix, render_body
from vulndetect.rag.embedding import (
    CrossEncoderReranker,
    Embedder,
    Reranker,
    SentenceTransformerEmbedder,
)
from vulndetect.rag.index import build_index, load_manifest, open_collection
from vulndetect.rag.retriever import CweRetriever, RetrievedChunk, distinct_cwe_count

__all__ = [
    "Chunk",
    "CrossEncoderReranker",
    "CweCatalog",
    "CweChunker",
    "CweEntry",
    "CweRetriever",
    "Embedder",
    "ObservedExample",
    "Reranker",
    "RetrievedChunk",
    "SentenceTransformerEmbedder",
    "build_index",
    "chunk_prefix",
    "distinct_cwe_count",
    "load_catalog",
    "load_manifest",
    "open_collection",
    "render_body",
]
