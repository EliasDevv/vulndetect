"""End-to-end tests of the RAG runner with a stub retriever and fake models."""

from __future__ import annotations

import csv
import json

import pytest
import yaml

from vulndetect.common.predictions import read_predictions
from vulndetect.data.build import build_dataset, to_records, write_jsonl
from vulndetect.rag.prompt import AUTHORS_RAG_PROMPT_VERSION
from vulndetect.rag.retriever import RetrievedChunk
from vulndetect.rag.run import load_retriever, run

N_TEST = 8
TOP_N = 20
TOP_K = 3


def make_chunk(cwe_id: str, index: int, distance: float, score: float | None) -> RetrievedChunk:
    # Braces in the text check that prompt assembly does not use str.format.
    return RetrievedChunk(
        text=f"{cwe_id}: Some Weakness\nExample code: if (x) {{ memcpy(dst, src, n); }}",
        cwe_id=cwe_id,
        cwe_name="Some Weakness",
        chunk_index=index,
        vector_distance=distance,
        rerank_score=score,
    )


FIXED_CHUNKS = [
    make_chunk("CWE-119", 0, 0.10, 9.5),
    make_chunk("CWE-787", 2, 0.15, 7.25),
    make_chunk("CWE-119", 4, 0.20, 3.0),
]


class StubRetriever:
    """Returns fixed chunks, records call args and flags one known query as truncated."""

    def __init__(self, truncated_code: str | None = None):
        self.calls: list[tuple[int, int]] = []
        self.truncated_code = truncated_code
        self.manifest = {
            "catalog": {"version": "9.9", "xml_sha256": "deadbeef"},
            "embedding": {"model": "stub-embedder", "revision": "0"},
            "chunking": {"chunk_size": 512, "chunk_overlap": 32, "n_chunks": 42},
        }

    def retrieve(self, code: str, *, top_n: int, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((top_n, top_k))
        return FIXED_CHUNKS[:top_k]

    def random_chunks(self, k: int, rng) -> list[RetrievedChunk]:
        pool = [make_chunk(f"CWE-9{i}", i, float("nan"), None) for i in range(10)]
        return rng.sample(pool, k)

    def query_is_truncated(self, code: str) -> bool:
        return code == self.truncated_code


class OracleModel:
    """Fake model that returns the gold labels in prompt order."""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.calls = 0

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        self.calls += 1
        assert len(prompts) == len(self.labels)
        return [f"{label}. Reasoning: because the test says so." for label in self.labels]


@pytest.fixture
def workspace(tmp_path, raw_df, base_cfg, monkeypatch):
    """A tmp cwd with a built dataset test split + full RAG config."""
    monkeypatch.chdir(tmp_path)
    records = to_records(build_dataset(raw_df, base_cfg))
    test_records: list[dict] = []
    for cwe in sorted({r["cwe"] for r in records}):
        cwe_test = [r for r in records if r["split"] == "test" and r["cwe"] == cwe]
        vuln = [r for r in cwe_test if r["vul"] == 1]
        safe = [r for r in cwe_test if r["vul"] == 0]
        if vuln and safe and len(test_records) < N_TEST:
            test_records += [vuln[0], safe[0]]
    assert len(test_records) >= 4, "fixture needs at least two CWEs in the test split"
    write_jsonl(test_records, tmp_path / "dataset" / "test.jsonl")

    cfg = {
        "experiment": "rag",
        "seed": 42,
        "retrieval": {
            "persist_dir": "unused-because-retriever-is-injected",
            "embedding_model": "stub-embedder",
            "embedding_revision": "0",
            "reranker_model": "stub-reranker",
            "reranker_revision": "1",
            "top_n": TOP_N,
            "top_k": TOP_K,
            "use_reranker": True,
        },
        "model": {
            "provider": "huggingface",
            "name": "meta-llama/Llama-3.2-3B-Instruct",
            "temperature": 0.0,
            "max_new_tokens": 512,
            "batch_size": 8,
        },
        "dataset": {"name": "bigvul", "split": "test", "dir": "dataset"},
        "output": {"predictions_dir": "results/predictions/rag"},
    }
    cfg_path = tmp_path / "rag.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path, test_records


def test_authors_rag_template_matches_fixture(fixtures_dir):
    from vulndetect.rag.prompt import AUTHORS_RAG_TEMPLATE

    assert AUTHORS_RAG_TEMPLATE == (fixtures_dir / "authors_rag_v1.txt").read_text(encoding="utf-8")


def test_authors_retrieval_query_matches_fixture(fixtures_dir):
    from vulndetect.rag.run import AUTHORS_RETRIEVAL_QUERY

    assert AUTHORS_RETRIEVAL_QUERY == (
        fixtures_dir / "authors_rag_retrieval_query_v1.txt"
    ).read_text(encoding="utf-8")


def test_render_authors_rag_prompt(workspace):
    from vulndetect.rag.prompt import render_authors_rag_prompt

    _, test_records = workspace
    record = test_records[0]
    prompt = render_authors_rag_prompt(record, FIXED_CHUNKS)
    ctx_pos = prompt.index("Retrieved domain knowledge from the MITRE CWE database:")
    assert (
        prompt.index("4) Assess the Overall Security Impact")
        < ctx_pos
        < prompt.index("Here is the data:")
    )
    assert "[1] (CWE-119)" in prompt
    assert "{ memcpy(dst, src, n); }" in prompt
    assert record["func_before"] in prompt


def test_render_authors_context_budget_and_fallback():
    from vulndetect.rag.prompt import AUTHORS_NO_CONTEXT, render_authors_context

    assert render_authors_context([]) == AUTHORS_NO_CONTEXT
    first_block_len = len(f"[1] (CWE-119) {FIXED_CHUNKS[0].text}")
    ctx = render_authors_context(FIXED_CHUNKS, max_chars=first_block_len)
    assert "[1] (CWE-119)" in ctx
    assert "[2]" not in ctx


def test_authors_truncate():
    from vulndetect.rag.prompt import authors_truncate

    assert authors_truncate("short", 10) == "short"
    assert authors_truncate("x" * 12, 10) == "x" * 10 + "\n... [truncated]"
    assert authors_truncate(None, 10) == ""


def test_authors_retrieval_query(workspace):
    from vulndetect.rag.run import retrieval_query

    _, test_records = workspace
    record = test_records[0]
    query = retrieval_query(record, "authors-v1")
    assert query.startswith(f"{record['cwe']} ")
    assert record["commit_message"][:50] in query
    assert record["func_before"][:100] in query
    assert retrieval_query(record, "code-v1") == record["code"]


def test_authors_rag_run_wiring(workspace, tmp_path):
    cfg_path, test_records = workspace
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["prompt"] = {"version": "authors-rag-v1"}
    cfg["retrieval"]["query_version"] = "authors-v1"
    cfg["parser_version"] = "authors-v1"
    cfg["eval"] = {"unparseable": "drop"}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    model = OracleModel([r["label"] for r in test_records])
    report = run(cfg_path, model=model, retriever=StubRetriever())
    assert report["macro"]["f1"] == pytest.approx(1.0)
    assert report["unparseable_policy"] == "drop"

    manifest = json.loads(
        (tmp_path / "results" / "predictions" / "rag" / "seed42" / "manifest.json").read_text()
    )
    assert manifest["prompt_version"] == "authors-rag-v1"
    assert manifest["parser_version"] == "authors-v1"
    assert manifest["unparseable_policy"] == "drop"
    assert manifest["retrieval"]["query_version"] == "authors-v1"


def test_rag_run_end_to_end(workspace, tmp_path):
    cfg_path, test_records = workspace
    truncated_code = test_records[0]["code"]
    retriever = StubRetriever(truncated_code=truncated_code)
    model = OracleModel([r["label"] for r in test_records])
    report = run(cfg_path, model=model, retriever=retriever)

    assert model.calls == 1
    assert report["parse_failure_rate"] == 0.0
    assert report["macro"]["f1"] == pytest.approx(1.0)

    out_dir = tmp_path / "results" / "predictions" / "rag" / "seed42"
    preds = read_predictions(out_dir / "predictions.jsonl")
    assert [p["id"] for p in preds] == [r["id"] for r in test_records]

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["approach"] == "rag"
    assert manifest["prompt_version"] == AUTHORS_RAG_PROMPT_VERSION
    assert manifest["decoding"]["batch_size"] == 8
    assert manifest["rag_prompt_version"] == AUTHORS_RAG_PROMPT_VERSION
    assert manifest["retrieval"]["query_version"] == "code-v1"
    retrieval = manifest["retrieval"]
    assert retrieval["top_n"] == TOP_N
    assert retrieval["top_k"] == TOP_K
    assert retrieval["use_reranker"] is True
    assert retrieval["reranker"] == {"model": "stub-reranker", "revision": "1"}
    assert retrieval["index"]["catalog_version"] == "9.9"
    assert retrieval["index"]["n_chunks"] == 42
    stats = retrieval["stats"]
    assert stats["n_queries"] == len(test_records)
    assert stats["n_truncated"] == 1
    assert stats["share_truncated"] == pytest.approx(1 / len(test_records), abs=1e-4)
    assert stats["distinct_cwe_count"] == {"mean": 2.0, "min": 2}
    assert stats["rank1_vector_distance_mean"] == pytest.approx(0.10, abs=1e-6)

    lines = (out_dir / "retrieval.jsonl").read_text().splitlines()
    assert len(lines) == len(test_records)
    first = json.loads(lines[0])
    assert first["id"] == test_records[0]["id"]
    assert first["query_truncated"] is True
    assert [c["cwe_id"] for c in first["chunks"]] == ["CWE-119", "CWE-787", "CWE-119"]
    assert first["chunks"][0]["rerank_score"] == pytest.approx(9.5)

    rows = list(csv.DictReader((tmp_path / "results" / "experiments.csv").open()))
    assert len(rows) == 1
    assert rows[0]["approach"] == "rag"
    params = json.loads(rows[0]["params"])
    assert params["rag_prompt_version"] == AUTHORS_RAG_PROMPT_VERSION
    assert params["top_k"] == TOP_K
    assert params["use_reranker"] is True
    assert params["catalog_version"] == "9.9"


def test_limit_smoke_run_not_logged(workspace, tmp_path):
    cfg_path, test_records = workspace
    limit = 3
    model = OracleModel([r["label"] for r in test_records[:limit]])
    report = run(cfg_path, limit=limit, model=model, retriever=StubRetriever())
    assert report["n"] == limit
    assert (tmp_path / "results" / "predictions" / "rag" / "seed42").exists()
    assert not (tmp_path / "results" / "experiments.csv").exists()


def test_val_style_run_not_logged(workspace, tmp_path):
    """Checks that a full run with log_experiments disabled writes no CSV row."""
    cfg_path, test_records = workspace
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["output"] = {"predictions_dir": "results/predictions/rag_val", "log_experiments": False}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    model = OracleModel([r["label"] for r in test_records])
    report = run(cfg_path, model=model, retriever=StubRetriever())
    assert report["n"] == len(test_records)

    out_dir = tmp_path / "results" / "predictions" / "rag_val" / "seed42"
    assert (out_dir / "predictions.jsonl").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["logged"] is False
    assert not (tmp_path / "results" / "experiments.csv").exists()


def test_top_n_top_k_come_from_config_only(workspace):
    cfg_path, test_records = workspace
    retriever = StubRetriever()
    run(cfg_path, model=OracleModel([r["label"] for r in test_records]), retriever=retriever)
    assert retriever.calls == [(TOP_N, TOP_K)] * len(test_records)


def test_run_is_deterministic(workspace, tmp_path):
    cfg_path, test_records = workspace
    out = tmp_path / "results" / "predictions" / "rag" / "seed42"

    def one_run() -> tuple[bytes, bytes]:
        run(
            cfg_path,
            model=OracleModel([r["label"] for r in test_records]),
            retriever=StubRetriever(),
        )
        return (out / "predictions.jsonl").read_bytes(), (out / "retrieval.jsonl").read_bytes()

    assert one_run() == one_run()


def _patch_index_manifest(monkeypatch, embedding: dict) -> None:
    monkeypatch.setattr(
        "vulndetect.rag.index.load_manifest", lambda persist_dir: {"embedding": embedding}
    )


def test_load_retriever_use_reranker_flag(monkeypatch):
    _patch_index_manifest(monkeypatch, {"model": "m", "revision": "r"})
    captured = {}

    def fake_from_persist_dir(persist_dir, embedder=None, reranker=None):
        captured["reranker"] = reranker
        return "sentinel-retriever"

    class FakeReranker:
        def __init__(self, model_name, revision=None):
            self.model_name = model_name
            self.revision = revision

    monkeypatch.setattr(
        "vulndetect.rag.retriever.CweRetriever.from_persist_dir", fake_from_persist_dir
    )
    monkeypatch.setattr("vulndetect.rag.embedding.CrossEncoderReranker", FakeReranker)

    base = {"persist_dir": "x", "embedding_model": "m", "embedding_revision": "r"}
    assert load_retriever({**base, "use_reranker": False}) == "sentinel-retriever"
    assert captured["reranker"] is None

    load_retriever({**base, "use_reranker": True, "reranker_model": "rr", "reranker_revision": "v"})
    assert isinstance(captured["reranker"], FakeReranker)
    assert (captured["reranker"].model_name, captured["reranker"].revision) == ("rr", "v")


def test_load_retriever_rejects_embedder_mismatch(monkeypatch):
    _patch_index_manifest(monkeypatch, {"model": "other-model", "revision": "other-rev"})
    with pytest.raises(ValueError, match="rebuild the index"):
        load_retriever({"persist_dir": "x", "embedding_model": "m", "embedding_revision": "r"})


class RecordingModel:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        self.prompts = list(prompts)
        return ["Not Vulnerable. Reasoning: control run."] * len(prompts)


def _run_with_mode(workspace, tmp_path, mode: str):
    import yaml

    from vulndetect.rag.run import run

    cfg_path, _ = workspace
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["retrieval"]["context_mode"] = mode
    cfg["experiment"] = f"rag_{mode}"
    cfg["output"]["predictions_dir"] = str(tmp_path / f"results/predictions/rag_{mode}")
    cfg["output"]["log_experiments"] = False
    p = tmp_path / f"rag_{mode}.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    retriever, model = StubRetriever(), RecordingModel()
    run(p, model=model, retriever=retriever)
    out_dir = tmp_path / f"results/predictions/rag_{mode}" / "seed42"
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    retrieval = [json.loads(line) for line in (out_dir / "retrieval.jsonl").open(encoding="utf-8")]
    return retriever, model, manifest, retrieval


def test_context_mode_none_feeds_the_authors_fallback_line(workspace, tmp_path):
    retriever, model, manifest, retrieval = _run_with_mode(workspace, tmp_path, "none")
    assert retriever.calls == []
    assert all("No relevant knowledge retrieved." in p for p in model.prompts)
    assert not any("[1] (" in p for p in model.prompts)
    assert manifest["retrieval"]["context_mode"] == "none"
    assert manifest["retrieval"]["stats"]["rank1_vector_distance_mean"] is None
    assert all(r["chunks"] == [] for r in retrieval)


def test_context_mode_random_is_seeded_and_unrelated_to_the_query(workspace, tmp_path):
    retriever, model, manifest, retrieval = _run_with_mode(workspace, tmp_path, "random")
    assert retriever.calls == []
    assert all("(CWE-9" in p for p in model.prompts)
    assert not any("(CWE-119)" in p for p in model.prompts)
    assert manifest["retrieval"]["context_mode"] == "random"
    assert all(len(r["chunks"]) == 3 for r in retrieval)
    assert all(c["vector_distance"] is None for r in retrieval for c in r["chunks"])
    (tmp_path / "again").mkdir()
    _, model2, _, _ = _run_with_mode(workspace, tmp_path / "again", "random")
    assert model2.prompts == model.prompts


def test_context_mode_unknown_fails(workspace, tmp_path):
    with pytest.raises(ValueError, match="unknown context_mode"):
        _run_with_mode(workspace, tmp_path, "shuffled")
