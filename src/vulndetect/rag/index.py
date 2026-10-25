"""Build and load the persistent Chroma index (cosine distance) over CWE chunks, with a manifest."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from vulndetect.rag.catalog import load_catalog
from vulndetect.rag.chunking import CweChunker, Tokenizer
from vulndetect.rag.embedding import Embedder

logger = logging.getLogger(__name__)

COLLECTION_NAME = "cwe"
MANIFEST_NAME = "manifest.json"
DISTANCE = "cosine"

EMBEDDING_PREFIXES = {"passage": "passage: ", "query": "query: "}

CORPUS_FIELDS = (
    "Name",
    "Abstraction",
    "Description",
    "Extended_Description",
    "Common_Consequences",
    "Potential_Mitigations",
    "Demonstrative_Examples",
    "Observed_Examples",
    "Summary",
    "Has_Member",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def chroma_client(persist_dir: str | Path):
    """Open a persistent Chroma client at the given directory with telemetry off."""
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
    )


def create_collection(client):
    """Create the CWE collection with the cosine distance metric."""
    return client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": DISTANCE})


def load_manifest(persist_dir: str | Path) -> dict:
    """Load the index manifest from the persist directory."""
    return json.loads((Path(persist_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def write_manifest(persist_dir: str | Path, manifest: dict) -> None:
    """Write the index manifest into the persist directory."""
    path = Path(persist_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def open_collection(persist_dir: str | Path):
    """Open the existing CWE collection in the persist directory."""
    return chroma_client(persist_dir).get_collection(COLLECTION_NAME)


def build_index(
    catalog_xml: str | Path,
    persist_dir: str | Path,
    embedder: Embedder,
    *,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: Tokenizer | None = None,
    tokenizer_name: str = "cl100k_base",
    seed: int | None = None,
    batch_size: int = 64,
    overwrite: bool = False,
    config_meta: dict | None = None,
    prefixes: dict | None = None,
) -> dict:
    """Chunk and embed the CWE catalog into a persistent Chroma collection; return the manifest."""
    catalog_xml = Path(catalog_xml)
    persist_dir = Path(persist_dir)
    if (persist_dir / MANIFEST_NAME).exists() and not overwrite:
        raise FileExistsError(
            f"{persist_dir / MANIFEST_NAME} exists — pass overwrite=True (CLI: --overwrite) "
            "to rebuild; the old index will be replaced"
        )

    catalog = load_catalog(catalog_xml)
    logger.info(
        "catalog v%s (%s): %d entries = %d weaknesses + %d categories (%d deprecated filtered)",
        catalog.version,
        catalog.date,
        len(catalog.entries),
        catalog.n_weaknesses,
        catalog.n_categories,
        catalog.n_deprecated_filtered,
    )
    chunker = CweChunker(chunk_size, chunk_overlap, tokenizer)
    chunks = chunker.chunk_catalog(catalog.entries)
    if not chunks:
        raise ValueError(f"{catalog_xml}: catalog produced no chunks")
    logger.info("chunked into %d chunks", len(chunks))

    n_over = 0
    if embedder.max_tokens is not None:
        n_over = sum(1 for c in chunks if embedder.count_tokens(c.text) > embedder.max_tokens)

    client = chroma_client(persist_dir)
    if overwrite:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001
            pass
    collection = create_collection(client)

    passage_prefix = (prefixes or {}).get("passage", "")
    dimension: int | None = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embedder.embed([passage_prefix + c.text for c in batch])
        if dimension is None:
            dimension = len(embeddings[0])
        collection.add(
            ids=[f"{c.cwe_id}:{c.chunk_index:04d}" for c in batch],
            embeddings=embeddings,
            documents=[c.text for c in batch],
            metadatas=[
                {"cwe_id": c.cwe_id, "cwe_name": c.cwe_name, "chunk_index": c.chunk_index}
                for c in batch
            ],
        )
    logger.info("indexed %d chunks into %s", collection.count(), persist_dir)

    zip_path = catalog_xml.with_name(catalog_xml.name + ".zip")
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "config": config_meta,
        "catalog": {
            "path": str(catalog_xml),
            "version": catalog.version,
            "date": catalog.date,
            "xml_sha256": _sha256(catalog_xml),
            "zip_sha256": _sha256(zip_path) if zip_path.exists() else None,
            "n_entries": len(catalog.entries),
            "n_weaknesses": catalog.n_weaknesses,
            "n_categories": catalog.n_categories,
            "n_deprecated_filtered": catalog.n_deprecated_filtered,
            "fields": list(CORPUS_FIELDS),
        },
        "chunking": {
            "splitter": "llama_index.core.node_parser.SentenceSplitter",
            "llama_index_core_version": importlib_metadata.version("llama-index-core"),
            "tokenizer": tokenizer_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_prefix_format": "<CWE-id>: <name>\\n (token cost reserved from chunk_size)",
            "n_chunks": len(chunks),
        },
        "embedding": {
            "model": embedder.name,
            "revision": embedder.revision,
            "dimension": dimension,
            "normalized": True,
            "prefixes": prefixes,
            "max_tokens": embedder.max_tokens,
            "chunks_over_embedder_limit": {
                "n": n_over,
                "share": round(n_over / len(chunks), 4),
            },
        },
        "vector_store": {
            "backend": "chromadb",
            "chromadb_version": importlib_metadata.version("chromadb"),
            "collection": COLLECTION_NAME,
            "distance": DISTANCE,
        },
    }
    write_manifest(persist_dir, manifest)
    return manifest


def main() -> None:
    """Parse CLI arguments, build the CWE index and print its manifest."""
    import argparse

    from vulndetect.common import config_sha256, load_config, set_seed
    from vulndetect.rag.embedding import SentenceTransformerEmbedder

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build the persistent CWE retrieval index.")
    ap.add_argument("--config", default="configs/rag.yaml", help="path to the RAG YAML")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing index")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    r = cfg["retrieval"]
    if r.get("distance", DISTANCE) != DISTANCE:
        raise ValueError(f"retrieval.distance must be {DISTANCE!r}")

    embedder = SentenceTransformerEmbedder(
        r["embedding_model"], revision=r.get("embedding_revision")
    )
    manifest = build_index(
        r["catalog_xml"],
        r["persist_dir"],
        embedder,
        chunk_size=int(r["chunk_size"]),
        chunk_overlap=int(r["chunk_overlap"]),
        seed=seed,
        overwrite=args.overwrite,
        config_meta={"path": str(args.config), "sha256": config_sha256(args.config)},
        prefixes=EMBEDDING_PREFIXES if bool(r.get("embedding_prefixes", False)) else None,
    )

    dataset_dir = r.get("query_truncation_dataset_dir")
    if dataset_dir and Path(dataset_dir).exists():
        from vulndetect.rag.measure import DEFAULT_QUERY_VERSION, measure_query_truncation

        manifest["query_truncation_estimate"] = measure_query_truncation(
            dataset_dir, embedder, r.get("query_version", DEFAULT_QUERY_VERSION)
        )
        write_manifest(r["persist_dir"], manifest)
    elif dataset_dir:
        logger.warning("dataset dir %s missing — query-truncation estimate skipped", dataset_dir)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
