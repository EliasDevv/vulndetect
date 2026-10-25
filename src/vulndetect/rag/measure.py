"""Retrieval validity measurements: query truncation, CVE overlap and gold-CWE recall."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from vulndetect.rag.catalog import load_catalog
from vulndetect.rag.embedding import Embedder

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")

DEFAULT_QUERY_VERSION = "code-v1"


def measure_query_truncation(
    dataset_dir: str | Path, embedder: Embedder, query_version: str = DEFAULT_QUERY_VERSION
) -> dict:
    """Measure the share of dataset instances whose retrieval query exceeds the embedder window."""
    import numpy as np

    from vulndetect.data import load_split
    from vulndetect.rag.run import QUERY_VERSIONS, retrieval_query

    if query_version not in QUERY_VERSIONS:
        raise ValueError(f"unknown query_version {query_version!r}; allowed: {QUERY_VERSIONS}")

    limit = embedder.max_tokens
    if limit is None:
        raise ValueError(f"embedder {embedder.name} reports no max_tokens — nothing to measure")

    per_split: dict[str, dict] = {}
    all_counts: list[int] = []
    for split in SPLITS:
        records = load_split(split, dataset_dir)
        counts = [embedder.count_tokens(retrieval_query(r, query_version)) for r in records]
        all_counts.extend(counts)
        n_over = sum(1 for c in counts if c > limit)
        per_split[split] = {
            "n": len(counts),
            "n_truncated": n_over,
            "share_truncated": round(n_over / len(counts), 4) if counts else None,
        }
        logger.info("%s: %d/%d queries over %d tokens", split, n_over, len(counts), limit)

    arr = np.asarray(all_counts)
    n_over_total = int((arr > limit).sum())
    return {
        "query_version": query_version,
        "tokenizer": f"{embedder.name} (embedder wordpiece, incl. special tokens)",
        "max_tokens": limit,
        "n": int(arr.size),
        "n_truncated": n_over_total,
        "share_truncated": round(n_over_total / arr.size, 4),
        "token_count_percentiles": {
            "p50": int(np.percentile(arr, 50)),
            "p90": int(np.percentile(arr, 90)),
            "p99": int(np.percentile(arr, 99)),
            "max": int(arr.max()),
        },
        "per_split": per_split,
    }


def measure_cwe_recall(
    dataset_dir: str | Path,
    retriever,
    *,
    split: str = "val",
    query_version: str = DEFAULT_QUERY_VERSION,
    top_n: int = 20,
    ks: tuple[int, ...] = (1, 5, 20),
) -> dict:
    """Measure recall@k of the instance's gold CWE, by vector distance and after re-ranking."""
    from vulndetect.data import load_split
    from vulndetect.rag.run import QUERY_VERSIONS, retrieval_query

    if query_version not in QUERY_VERSIONS:
        raise ValueError(f"unknown query_version {query_version!r}; allowed: {QUERY_VERSIONS}")
    if any(k > top_n for k in ks):
        raise ValueError(f"every k in {ks} must be <= top_n ({top_n})")

    records = load_split(split, dataset_dir)
    hits: dict[str, dict[int, int]] = {o: dict.fromkeys(ks, 0) for o in ("vector", "reranked")}
    trunc_hits: dict[bool, dict[int, int]] = {
        True: dict.fromkeys(ks, 0),
        False: dict.fromkeys(ks, 0),
    }
    trunc_n: dict[bool, int] = {True: 0, False: 0}

    for record in records:
        query = retrieval_query(record, query_version)
        chunks = retriever.retrieve(query, top_n=top_n, top_k=top_n)
        by_vector = sorted(chunks, key=lambda c: (c.vector_distance, c.cwe_id, c.chunk_index))
        gold = record["cwe"]
        truncated = retriever.query_is_truncated(query)
        trunc_n[truncated] += 1
        for k in ks:
            hits["vector"][k] += gold in {c.cwe_id for c in by_vector[:k]}
            hit_reranked = gold in {c.cwe_id for c in chunks[:k]}
            hits["reranked"][k] += hit_reranked
            trunc_hits[truncated][k] += hit_reranked

    n = len(records)
    manifest = getattr(retriever, "manifest", None) or {}
    catalog_meta = manifest.get("catalog", {})
    n_cwe = catalog_meta.get("n_entries") or catalog_meta.get("n_weaknesses")
    reranker = getattr(retriever, "reranker", None)

    def _rates(counts: dict[int, int]) -> dict[str, float | None]:
        return {str(k): round(c / n, 4) if n else None for k, c in counts.items()}

    chance = {str(k): round(min(k / n_cwe, 1.0), 4) for k in ks} if n_cwe else None
    report = {
        "split": split,
        "query_version": query_version,
        "n": n,
        "top_n": top_n,
        "reranker": getattr(reranker, "name", None),
        "n_cwe_in_index": n_cwe,
        "recall_at_k": {
            "by_vector_distance": _rates(hits["vector"]),
            "after_reranking": _rates(hits["reranked"]),
        },
        "chance_recall_at_k": chance,
        "by_query_truncation": {
            state: {
                "n": trunc_n[flag],
                "recall_at_k_after_reranking": {
                    str(k): (
                        round(trunc_hits[flag][k] / trunc_n[flag], 4) if trunc_n[flag] else None
                    )
                    for k in ks
                },
            }
            for state, flag in (("truncated", True), ("not_truncated", False))
        },
    }
    logger.info(
        "%s: gold CWE recall@%d after re-ranking %.3f (vector %.3f) over %d instances",
        split,
        max(ks),
        report["recall_at_k"]["after_reranking"][str(max(ks))],
        report["recall_at_k"]["by_vector_distance"][str(max(ks))],
        n,
    )
    return report


def measure_cve_overlap(dataset_dir: str | Path, catalog_xml: str | Path) -> dict:
    """Intersect Big-Vul CVE ids with the catalog's Observed_Examples."""
    from vulndetect.data import load_split

    catalog = load_catalog(catalog_xml)
    catalog_cves: dict[str, list[str]] = {}
    for entry in catalog.entries:
        for example in entry.observed_examples:
            if example.reference.upper().startswith("CVE-"):
                catalog_cves.setdefault(example.reference.upper(), []).append(entry.cwe_id)

    per_split: dict[str, dict] = {}
    overlap_all: set[str] = set()
    for split in SPLITS:
        records = load_split(split, dataset_dir)
        with_cve = [r for r in records if r["cve_id"] and r["cve_id"] != "N/A"]
        split_cves = {r["cve_id"].upper() for r in with_cve}
        overlap = split_cves & catalog_cves.keys()
        overlap_all |= overlap
        n_affected = sum(1 for r in with_cve if r["cve_id"].upper() in overlap)
        per_split[split] = {
            "n_records": len(records),
            "n_unique_cves": len(split_cves),
            "n_overlapping_cves": len(overlap),
            "n_affected_records": n_affected,
            "share_affected_records": round(n_affected / len(records), 4) if records else None,
        }

    return {
        "catalog_version": catalog.version,
        "n_catalog_cves": len(catalog_cves),
        "n_overlapping_cves_total": len(overlap_all),
        "overlapping_cves": {cve: sorted(set(catalog_cves[cve])) for cve in sorted(overlap_all)},
        "per_split": per_split,
    }


def main() -> None:
    """Parse CLI arguments, run the selected retrieval measurements and print them as JSON."""
    import argparse

    from vulndetect.common import load_config

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Retrieval-layer validity measurements.")
    ap.add_argument("--config", default="configs/rag.yaml", help="path to the RAG YAML")
    ap.add_argument(
        "--query-truncation",
        action="store_true",
        help="also re-run the query-truncation measurement (downloads the embedder)",
    )
    ap.add_argument(
        "--cwe-recall",
        action="store_true",
        help="also measure gold-CWE recall@k of the retrieval (needs the built index)",
    )
    ap.add_argument(
        "--split",
        default="val",
        help="split for --cwe-recall (default val; diagnostics stay off test)",
    )
    args = ap.parse_args()

    r = load_config(args.config)["retrieval"]
    dataset_dir = r.get("query_truncation_dataset_dir", "data/processed/bigvul")
    report = {"cve_overlap": measure_cve_overlap(dataset_dir, r["catalog_xml"])}
    if args.cwe_recall:
        from vulndetect.rag.run import load_retriever

        report["cwe_recall"] = measure_cwe_recall(
            dataset_dir,
            load_retriever(r),
            split=args.split,
            query_version=r.get("query_version", DEFAULT_QUERY_VERSION),
            top_n=int(r["top_n"]),
            ks=(1, int(r["top_k"]), int(r["top_n"])),
        )
    if args.query_truncation:
        from vulndetect.rag.embedding import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(
            r["embedding_model"], revision=r.get("embedding_revision")
        )
        report["query_truncation"] = measure_query_truncation(
            dataset_dir, embedder, r.get("query_version", DEFAULT_QUERY_VERSION)
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
