"""RAG runner (paper §3): retrieve CWE context per instance, prompt the model and evaluate."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

from vulndetect.common import (
    DEFAULT_PARSER_VERSION,
    config_sha256,
    get_parser,
    load_config,
    make_prediction,
    set_seed,
    write_run,
)
from vulndetect.common.model import CHAT_DATE_STRING, TextGenerator
from vulndetect.data import DEFAULT_DATASET_DIR, load_split
from vulndetect.data.schema import NA, Record
from vulndetect.eval import append_run, evaluate, format_report, logging_enabled, summary_metrics
from vulndetect.eval.metrics import DEFAULT_POLICY
from vulndetect.rag.prompt import (
    AUTHORS_RAG_PROMPT_VERSION,
    RAG_PROMPT_VERSIONS,
    authors_truncate,
    render_authors_rag_prompt,
)
from vulndetect.rag.retriever import CweRetriever, distinct_cwe_count

logger = logging.getLogger(__name__)

APPROACH = "rag"

QUERY_VERSIONS = ("authors-v1", "code-v1")

CONTEXT_MODES = ("retrieved", "none", "random")

AUTHORS_RETRIEVAL_QUERY = """{cwe_id} {cve_id} {lang} {commit_message}
{func_before}"""


def retrieval_query(record: Record, version: str = "code-v1") -> str:
    """Build the retrieval query (authors' metadata plus truncated code, or bare ``code``)."""
    if version == "authors-v1":
        return AUTHORS_RETRIEVAL_QUERY.format(
            cwe_id=record.get("cwe") or NA,
            cve_id=record.get("cve_id") or NA,
            lang=record["lang"] or NA,
            commit_message=authors_truncate(record["commit_message"] or "", 1000),
            func_before=authors_truncate(record["func_before"] or "", 1500),
        )
    return record["code"]


def load_retriever(retrieval_cfg: dict) -> CweRetriever:
    """Open the persistent index, check it matches the config and build the re-ranker."""
    from vulndetect.rag.index import load_manifest

    persist_dir = retrieval_cfg["persist_dir"]
    manifest_embedding = load_manifest(persist_dir).get("embedding", {})
    want = (retrieval_cfg.get("embedding_model"), retrieval_cfg.get("embedding_revision"))
    have = (manifest_embedding.get("model"), manifest_embedding.get("revision"))
    if want != have:
        raise ValueError(
            f"index at {persist_dir} was built with embedder {have}, config says {want} — "
            "rebuild the index via `make cwe-index` or fix configs/rag.yaml"
        )
    want_prefixes = bool(retrieval_cfg.get("embedding_prefixes", False))
    have_prefixes = bool(manifest_embedding.get("prefixes"))
    if want_prefixes != have_prefixes:
        raise ValueError(
            f"index at {persist_dir} was built {'with' if have_prefixes else 'without'} "
            f"bge prefixes, config says {'with' if want_prefixes else 'without'} — "
            "rebuild the index via `make cwe-index` or fix configs/rag.yaml"
        )

    reranker = None
    if bool(retrieval_cfg.get("use_reranker", True)):
        from vulndetect.rag.embedding import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            retrieval_cfg["reranker_model"], revision=retrieval_cfg.get("reranker_revision")
        )
    return CweRetriever.from_persist_dir(persist_dir, reranker=reranker)


def run(
    config_path: str | Path,
    limit: int | None = None,
    model: TextGenerator | None = None,
    retriever: CweRetriever | None = None,
) -> dict:
    """Execute one RAG run and return the metrics report."""
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)

    ds_cfg = cfg.get("dataset", {})
    records = load_split(ds_cfg.get("split", "test"), ds_cfg.get("dir", DEFAULT_DATASET_DIR))
    if limit is not None:
        records = records[:limit]
    logger.info("loaded %d records", len(records))

    r_cfg = cfg["retrieval"]
    top_n = int(r_cfg["top_n"])
    top_k = int(r_cfg["top_k"])
    use_reranker = bool(r_cfg.get("use_reranker", True))
    query_version = r_cfg.get("query_version", "code-v1")
    if query_version not in QUERY_VERSIONS:
        raise ValueError(f"unknown query_version {query_version!r}; allowed: {QUERY_VERSIONS}")
    context_mode = r_cfg.get("context_mode", "retrieved")
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"unknown context_mode {context_mode!r}; allowed: {CONTEXT_MODES}")
    rag_prompt_version = cfg.get("prompt", {}).get("version", AUTHORS_RAG_PROMPT_VERSION)
    if rag_prompt_version not in RAG_PROMPT_VERSIONS:
        raise ValueError(
            f"unknown prompt version {rag_prompt_version!r}; allowed: {RAG_PROMPT_VERSIONS}"
        )
    parser_version = cfg.get("parser_version", DEFAULT_PARSER_VERSION)
    parse = get_parser(parser_version)
    policy = cfg.get("eval", {}).get("unparseable", DEFAULT_POLICY)

    if retriever is None:
        retriever = load_retriever(r_cfg)

    chunk_lists = []
    truncated_flags = []
    control_rng = random.Random(seed)
    for record in records:
        query = retrieval_query(record, query_version)
        if context_mode == "retrieved":
            chunks = retriever.retrieve(query, top_n=top_n, top_k=top_k)
        elif context_mode == "random":
            chunks = retriever.random_chunks(top_k, control_rng)
        else:
            chunks = []
        chunk_lists.append(chunks)
        truncated_flags.append(retriever.query_is_truncated(query))
    logger.info("context_mode=%s: built context for %d records", context_mode, len(records))

    prompts = [
        render_authors_rag_prompt(record, chunks)
        for record, chunks in zip(records, chunk_lists, strict=True)
    ]

    if model is None:
        from vulndetect.common.model import load_model

        model = load_model(cfg["model"])
    outputs = model.generate(prompts)
    preds = [
        make_prediction(rec, out, parse(out)) for rec, out in zip(records, outputs, strict=True)
    ]

    out_dir = (
        Path(cfg["output"].get("predictions_dir", f"results/predictions/{APPROACH}"))
        / f"seed{seed}"
    )
    report = evaluate(preds, policy=policy)
    report["parser_version"] = parser_version

    n_truncated = sum(truncated_flags)
    distinct_counts = [distinct_cwe_count(chunks) for chunks in chunk_lists]
    rank1_distances = [
        chunks[0].vector_distance
        for chunks in chunk_lists
        if chunks and not math.isnan(chunks[0].vector_distance)
    ]
    index_manifest = retriever.manifest or {}
    catalog_meta = index_manifest.get("catalog", {})
    temperature = cfg["model"].get("temperature", 0.0)
    manifest = {
        "approach": APPROACH,
        "model": cfg["model"]["name"],
        "model_revision": cfg["model"].get("revision"),
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": config_sha256(config_path),
        "prompt_version": rag_prompt_version,
        "rag_prompt_version": rag_prompt_version,
        "parser_version": parser_version,
        "unparseable_policy": policy,
        "dataset": {**ds_cfg, "n": len(records), "limit": limit},
        "decoding": {
            "temperature": temperature,
            "top_p": cfg["model"].get("top_p"),
            "do_sample": temperature > 0,
            "max_new_tokens": cfg["model"].get("max_new_tokens"),
            "batch_size": cfg["model"].get("batch_size"),
            "chat_date_string": cfg["model"].get("chat_date_string", CHAT_DATE_STRING),
        },
        "logged": limit is None and logging_enabled(cfg),
        "retrieval": {
            "persist_dir": str(r_cfg["persist_dir"]),
            "query_version": query_version,
            "context_mode": context_mode,
            "top_n": top_n,
            "top_k": top_k,
            "use_reranker": use_reranker,
            "reranker": (
                {
                    "model": r_cfg.get("reranker_model"),
                    "revision": r_cfg.get("reranker_revision"),
                }
                if use_reranker
                else None
            ),
            "index": {
                "catalog_version": catalog_meta.get("version"),
                "catalog_xml_sha256": catalog_meta.get("xml_sha256"),
                "embedding_model": index_manifest.get("embedding", {}).get("model"),
                "embedding_revision": index_manifest.get("embedding", {}).get("revision"),
                "chunk_size": index_manifest.get("chunking", {}).get("chunk_size"),
                "chunk_overlap": index_manifest.get("chunking", {}).get("chunk_overlap"),
                "n_chunks": index_manifest.get("chunking", {}).get("n_chunks"),
            },
            "stats": {
                "n_queries": len(records),
                "n_truncated": n_truncated,
                "share_truncated": round(n_truncated / len(records), 4) if records else None,
                "distinct_cwe_count": {
                    "mean": round(sum(distinct_counts) / len(distinct_counts), 4),
                    "min": min(distinct_counts),
                },
                "rank1_vector_distance_mean": (
                    round(sum(rank1_distances) / len(rank1_distances), 4)
                    if rank1_distances
                    else None
                ),
            },
        },
    }
    write_run(out_dir, preds, manifest)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    retrieval_lines = [
        json.dumps(
            {
                "id": record["id"],
                "query_truncated": truncated,
                "chunks": [
                    {
                        "cwe_id": c.cwe_id,
                        "chunk_index": c.chunk_index,
                        "vector_distance": (
                            None if math.isnan(c.vector_distance) else round(c.vector_distance, 6)
                        ),
                        "rerank_score": (
                            round(c.rerank_score, 6) if c.rerank_score is not None else None
                        ),
                    }
                    for c in chunks
                ],
            },
            ensure_ascii=False,
        )
        for record, chunks, truncated in zip(records, chunk_lists, truncated_flags, strict=True)
    ]
    (out_dir / "retrieval.jsonl").write_text("\n".join(retrieval_lines) + "\n", encoding="utf-8")

    if manifest["logged"]:
        append_run(
            approach=APPROACH,
            model=cfg["model"]["name"],
            seed=seed,
            params={
                "rag_prompt_version": rag_prompt_version,
                "parser_version": parser_version,
                "unparseable_policy": policy,
                "query_version": query_version,
                "context_mode": context_mode,
                "top_n": top_n,
                "top_k": top_k,
                "use_reranker": use_reranker,
                "catalog_version": catalog_meta.get("version"),
                **manifest["decoding"],
            },
            metrics=summary_metrics(report),
            output=str(out_dir),
        )

    print(format_report(report))
    logger.info("wrote run artifacts to %s", out_dir)
    return report


def main() -> None:
    """Parse CLI arguments and run the RAG experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="RAG: retrieval-augmented prompting (paper §3).")
    ap.add_argument("--config", default="configs/rag.yaml", help="path to the RAG YAML")
    ap.add_argument("--limit", type=int, help="only the first N records (smoke run, not logged)")
    args = ap.parse_args()
    run(args.config, limit=args.limit)


if __name__ == "__main__":
    main()
