"""Offline tests for CWE catalog parsing, chunking, indexing and retrieval with stub models."""

from __future__ import annotations

import hashlib
import math
import re

import pytest

from vulndetect.rag import (
    CweChunker,
    CweEntry,
    CweRetriever,
    RetrievedChunk,
    build_index,
    chunk_prefix,
    distinct_cwe_count,
    load_catalog,
    open_collection,
    render_body,
)
from vulndetect.rag.index import chroma_client, create_collection

# Short sentences so that whole sentences fit into the chunk overlap.
LONG_BODY = " ".join(f"Resource leak number {i} persists." for i in range(80))

CATALOG_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<Weakness_Catalog xmlns="http://cwe.mitre.org/cwe-7"
                  xmlns:xhtml="http://www.w3.org/1999/xhtml"
                  Name="CWE" Version="9.9" Date="2099-01-01">
 <Weaknesses>
  <Weakness ID="1001" Name="Test Buffer Overflow" Abstraction="Base" Structure="Simple"
            Status="Stable">
   <Description>The product writes data past the end of a buffer.</Description>
   <Extended_Description>
    <xhtml:p>Buffer overflows corrupt memory.</xhtml:p>
    <xhtml:p>They may allow code execution.</xhtml:p>
   </Extended_Description>
   <Common_Consequences>
    <Consequence>
     <Scope>Integrity</Scope>
     <Impact>Execute Unauthorized Code or Commands</Impact>
     <Note>Memory corruption.</Note>
    </Consequence>
   </Common_Consequences>
   <Potential_Mitigations>
    <Mitigation>
     <Phase>Implementation</Phase>
     <Strategy>Input Validation</Strategy>
     <Description><xhtml:p>Check buffer bounds.</xhtml:p></Description>
    </Mitigation>
   </Potential_Mitigations>
   <Demonstrative_Examples>
    <Demonstrative_Example>
     <Intro_Text>Copying without bounds check.</Intro_Text>
     <Example_Code Nature="Bad" Language="C">strcpy(dst, src);</Example_Code>
     <Body_Text>The destination buffer may overflow.</Body_Text>
    </Demonstrative_Example>
   </Demonstrative_Examples>
   <Observed_Examples>
    <Observed_Example>
     <Reference>CVE-2020-0001</Reference>
     <Description>Buffer overflow in test product.</Description>
     <Link>https://example.org/1</Link>
    </Observed_Example>
    <Observed_Example>
     <Reference>CVE-2021-0002</Reference>
     <Description>Second overflow.</Description>
     <Link>https://example.org/2</Link>
    </Observed_Example>
   </Observed_Examples>
  </Weakness>
  <Weakness ID="1002" Name="Test SQL Injection" Abstraction="Base" Structure="Simple"
            Status="Draft">
   <Description>The product builds SQL queries from unsanitized input.</Description>
  </Weakness>
  <Weakness ID="1003" Name="Old Deprecated Weakness" Abstraction="Base" Structure="Simple"
            Status="Deprecated">
   <Description>Deprecated entry that must not be indexed.</Description>
  </Weakness>
  <Weakness ID="1004" Name="Test Resource Leak" Abstraction="Base" Structure="Simple"
            Status="Incomplete">
   <Description>{LONG_BODY}</Description>
  </Weakness>
 </Weaknesses>
</Weakness_Catalog>
"""

CATEGORY_CATALOG_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Weakness_Catalog xmlns="http://cwe.mitre.org/cwe-7"
                  xmlns:xhtml="http://www.w3.org/1999/xhtml"
                  Name="CWE" Version="9.9" Date="2099-01-01">
 <Weaknesses>
  <Weakness ID="2001" Name="Test Out-of-bounds Write" Abstraction="Base" Structure="Simple"
            Status="Stable">
   <Description>The product writes past the end of a buffer.</Description>
  </Weakness>
  <Weakness ID="2002" Name="Test Gone Weakness" Abstraction="Base" Structure="Simple"
            Status="Deprecated">
   <Description>Deprecated, but still referenced by a category below.</Description>
  </Weakness>
 </Weaknesses>
 <Categories>
  <Category ID="3001" Name="Test Resource Category" Status="Draft">
   <Summary>Weaknesses in this category are related to resource management.</Summary>
   <Relationships>
    <Has_Member CWE_ID="2001" View_ID="1000"/>
    <Has_Member CWE_ID="2002" View_ID="1000"/>
    <Has_Member CWE_ID="2001" View_ID="699"/>
   </Relationships>
  </Category>
  <Category ID="3002" Name="Test Obsolete Category" Status="Obsolete">
   <Summary>An obsolete category that still carries a real summary.</Summary>
  </Category>
  <Category ID="3003" Name="Test Deprecated Category" Status="Deprecated">
   <Summary>Must not be indexed.</Summary>
  </Category>
  <Category ID="3004" Name="Test Empty Category" Status="Draft"/>
 </Categories>
</Weakness_Catalog>
"""

CHUNK_SIZE = 60
CHUNK_OVERLAP = 8
TOP_N = 10
TOP_K = 5


def ws_tokenizer(text: str) -> list[str]:
    return text.split()


class StubEmbedder:
    """Deterministic keyword-count embeddings with a constant bias dimension."""

    name = "stub-embedder"
    revision = "0"
    max_tokens = 40

    VOCAB = ("buffer", "overflow", "sql", "queries", "unsanitized", "resource", "leak", "handles")

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            words = text.lower().split()
            v = [float(words.count(w)) for w in self.VOCAB] + [1.0]
            norm = math.sqrt(sum(x * x for x in v))
            out.append([x / norm for x in v])
        return out

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class StubReranker:
    """Score = number of query words present in the chunk text."""

    name = "stub-reranker"
    revision = "0"

    def scores(self, query: str, texts: list[str]) -> list[float]:
        q = set(query.lower().split())
        return [float(len(q & set(t.lower().split()))) for t in texts]


@pytest.fixture(scope="module")
def catalog_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("cwe_xml") / "cwec_test.xml"
    path.write_text(CATALOG_XML, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def catalog(catalog_path):
    return load_catalog(catalog_path)


@pytest.fixture(scope="module")
def category_catalog(tmp_path_factory):
    path = tmp_path_factory.mktemp("cwe_cat_xml") / "cwec_categories.xml"
    path.write_text(CATEGORY_CATALOG_XML, encoding="utf-8")
    return load_catalog(path)


def _build(catalog_path, persist_dir, **kwargs):
    return build_index(
        catalog_path,
        persist_dir,
        StubEmbedder(),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=ws_tokenizer,
        tokenizer_name="whitespace-stub",
        seed=42,
        **kwargs,
    )


@pytest.fixture(scope="module")
def built(catalog_path, tmp_path_factory):
    persist_dir = tmp_path_factory.mktemp("cwe_index")
    manifest = _build(catalog_path, persist_dir)
    return persist_dir, manifest


@pytest.fixture
def retriever(built):
    persist_dir, _ = built
    return CweRetriever.from_persist_dir(persist_dir, embedder=StubEmbedder())


def test_load_catalog_filters_deprecated_and_reads_version(catalog):
    assert catalog.version == "9.9"
    assert catalog.date == "2099-01-01"
    assert [e.cwe_id for e in catalog.entries] == ["CWE-1001", "CWE-1002", "CWE-1004"]
    assert catalog.n_deprecated_filtered == 1
    assert (catalog.n_weaknesses, catalog.n_categories) == (3, 0)


def test_full_entry_fields(catalog):
    e = catalog.entries[0]
    assert e.name == "Test Buffer Overflow"
    assert e.abstraction == "Base"
    assert e.description == "The product writes data past the end of a buffer."
    assert "corrupt memory" in e.extended_description
    assert "code execution" in e.extended_description
    assert e.consequences == (
        "Scope: Integrity. Impact: Execute Unauthorized Code or Commands. Memory corruption.",
    )
    assert e.mitigations == (
        "Phase: Implementation. Strategy: Input Validation. Check buffer bounds.",
    )
    assert e.demonstrative_examples == (
        "Copying without bounds check. [Bad code, C] strcpy(dst, src); "
        "The destination buffer may overflow.",
    )
    assert [o.reference for o in e.observed_examples] == ["CVE-2020-0001", "CVE-2021-0002"]
    assert e.observed_examples[0].description == "Buffer overflow in test product."


def test_minimal_entry_has_empty_optionals(catalog):
    e = catalog.entries[1]
    assert e.extended_description == ""
    assert e.consequences == ()
    assert e.mitigations == ()
    assert e.demonstrative_examples == ()
    assert e.observed_examples == ()


def test_categories_are_indexed_after_the_weaknesses(category_catalog):
    """Checks that categories are parsed and indexed after the weaknesses."""
    assert [e.cwe_id for e in category_catalog.entries] == [
        "CWE-2001",
        "CWE-3001",
        "CWE-3002",
        "CWE-3004",
    ]
    assert (category_catalog.n_weaknesses, category_catalog.n_categories) == (1, 3)
    assert category_catalog.n_deprecated_filtered == 2


def test_category_fields_and_member_resolution(category_catalog):
    by_id = {e.cwe_id: e for e in category_catalog.entries}
    category = by_id["CWE-3001"]
    assert category.abstraction == "Category"
    assert category.status == "Draft"
    assert category.description == "Weaknesses in this category are related to resource management."
    assert category.members == ("CWE-2001: Test Out-of-bounds Write", "CWE-2002")
    assert category.extended_description == ""
    assert category.consequences == ()
    assert category.observed_examples == ()


def test_obsolete_category_is_kept_but_deprecated_is_not(category_catalog):
    """Checks that obsolete categories are kept and deprecated ones are filtered."""
    by_id = {e.cwe_id: e for e in category_catalog.entries}
    assert by_id["CWE-3002"].status == "Obsolete"
    assert by_id["CWE-3002"].members == ()
    assert "CWE-3003" not in by_id


def test_category_body_renders_members_and_stays_indexable(category_catalog):
    by_id = {e.cwe_id: e for e in category_catalog.entries}
    body = render_body(by_id["CWE-3001"])
    assert body.startswith("Abstraction: Category.")
    assert "Members:\n- CWE-2001: Test Out-of-bounds Write\n- CWE-2002" in body
    assert chunk_prefix(by_id["CWE-3001"]) == "CWE-3001: Test Resource Category\n"
    assert render_body(by_id["CWE-3004"]) == ""


def test_weakness_rendering_is_unchanged_by_the_members_section(catalog):
    """Checks that weakness bodies contain no members section."""
    for entry in catalog.entries:
        assert entry.members == ()
        assert "Members:" not in render_body(entry)


def test_render_body_sections_and_prefix(catalog):
    e = catalog.entries[0]
    assert chunk_prefix(e) == "CWE-1001: Test Buffer Overflow\n"
    body = render_body(e)
    assert body.startswith("Abstraction: Base.")
    for section in (
        "Description:",
        "Extended Description:",
        "Common Consequences:",
        "Potential Mitigations:",
        "Demonstrative Examples:",
        "Observed Examples (CVEs):",
    ):
        assert section in body
    assert "CWE-1001" not in body


def test_render_body_empty_entry_yields_nothing():
    empty = CweEntry(
        cwe_id="CWE-9999",
        name="Empty",
        abstraction="Base",
        status="Draft",
        description="",
        extended_description="",
        consequences=(),
        mitigations=(),
        demonstrative_examples=(),
        observed_examples=(),
    )
    assert render_body(empty) == ""
    chunker = CweChunker(CHUNK_SIZE, CHUNK_OVERLAP, ws_tokenizer)
    assert chunker.chunk_entry(empty) == []


def test_chunks_carry_prefix_and_respect_budget(catalog):
    chunker = CweChunker(CHUNK_SIZE, CHUNK_OVERLAP, ws_tokenizer)
    for entry in catalog.entries:
        for chunk in chunker.chunk_entry(entry):
            assert chunk.text.startswith(chunk_prefix(entry))
            assert chunk.n_tokens == len(ws_tokenizer(chunk.text))
            assert chunk.n_tokens <= CHUNK_SIZE


def test_short_document_yields_single_chunk(catalog):
    chunker = CweChunker(CHUNK_SIZE, CHUNK_OVERLAP, ws_tokenizer)
    chunks = chunker.chunk_entry(catalog.entries[1])
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_long_document_is_split_with_overlap(catalog):
    chunker = CweChunker(CHUNK_SIZE, CHUNK_OVERLAP, ws_tokenizer)
    chunks = chunker.chunk_entry(catalog.entries[2])
    assert len(chunks) >= 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    overlapping_pairs = 0
    for a, b in zip(chunks, chunks[1:], strict=False):
        nums_a = [int(n) for n in re.findall(r"number (\d+)", a.text)]
        nums_b = [int(n) for n in re.findall(r"number (\d+)", b.text)]
        if nums_a and nums_b and min(nums_b) <= max(nums_a):
            overlapping_pairs += 1
    assert overlapping_pairs > 0


def test_chunker_rejects_budget_smaller_than_prefix_plus_overlap(catalog):
    chunker = CweChunker(chunk_size=10, chunk_overlap=8, tokenizer=ws_tokenizer)
    with pytest.raises(ValueError, match="chunk_size"):
        chunker.chunk_entry(catalog.entries[0])


def test_manifest_records_snapshot_and_stats(built, catalog_path):
    persist_dir, manifest = built
    expected_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    assert manifest["catalog"]["version"] == "9.9"
    assert manifest["catalog"]["xml_sha256"] == expected_sha
    assert manifest["catalog"]["n_weaknesses"] == 3
    assert manifest["catalog"]["n_entries"] == 3
    assert manifest["catalog"]["n_categories"] == 0
    assert manifest["catalog"]["n_deprecated_filtered"] == 1
    assert manifest["chunking"]["chunk_size"] == CHUNK_SIZE
    assert manifest["chunking"]["chunk_overlap"] == CHUNK_OVERLAP
    assert manifest["chunking"]["tokenizer"] == "whitespace-stub"
    assert manifest["chunking"]["n_chunks"] == open_collection(persist_dir).count()
    assert manifest["embedding"]["model"] == "stub-embedder"
    assert manifest["embedding"]["dimension"] == len(StubEmbedder.VOCAB) + 1
    assert manifest["seed"] == 42

    chunker = CweChunker(CHUNK_SIZE, CHUNK_OVERLAP, ws_tokenizer)
    catalog = load_catalog(catalog_path)
    chunks = chunker.chunk_catalog(catalog.entries)
    n_over = sum(1 for c in chunks if len(c.text.split()) > StubEmbedder.max_tokens)
    assert n_over > 0
    stats = manifest["embedding"]["chunks_over_embedder_limit"]
    assert stats["n"] == n_over
    assert stats["share"] == pytest.approx(n_over / len(chunks), abs=1e-4)


def test_collection_space_is_cosine(built, tmp_path):
    persist_dir, manifest = built
    assert manifest["vector_store"]["distance"] == "cosine"
    assert open_collection(persist_dir).metadata["hnsw:space"] == "cosine"
    # Cosine gives distances [0, 1] here; Chroma's l2 default would give [1, 5].
    probe = create_collection(chroma_client(tmp_path / "probe"))
    probe.add(ids=["x", "y"], embeddings=[[1.0, 0.0], [0.0, 1.0]], documents=["x", "y"])
    distances = probe.query(query_embeddings=[[2.0, 0.0]], n_results=2, include=["distances"])[
        "distances"
    ][0]
    assert distances == pytest.approx([0.0, 1.0], abs=1e-6)


def test_build_refuses_silent_overwrite(catalog_path, tmp_path):
    _build(catalog_path, tmp_path)
    with pytest.raises(FileExistsError):
        _build(catalog_path, tmp_path)
    manifest = _build(catalog_path, tmp_path, overwrite=True)
    assert manifest["chunking"]["n_chunks"] == open_collection(tmp_path).count()


def test_retrieve_relevant_first_distances_ascending(retriever):
    results = retriever.retrieve(
        "buffer overflow memory corruption strcpy", top_n=TOP_N, top_k=TOP_K
    )
    assert len(results) == TOP_K
    assert results[0].cwe_id == "CWE-1001"
    assert all(r.rerank_score is None for r in results)
    distances = [r.vector_distance for r in results]
    assert distances == sorted(distances)


def test_retrieve_is_deterministic(built, retriever, catalog_path, tmp_path):
    query = "resource leak file handles"
    first = retriever.retrieve(query, top_n=TOP_N, top_k=TOP_K)
    second = retriever.retrieve(query, top_n=TOP_N, top_k=TOP_K)
    assert first == second

    persist_dir, _ = built
    again = CweRetriever.from_persist_dir(persist_dir, embedder=StubEmbedder())
    assert again.retrieve(query, top_n=TOP_N, top_k=TOP_K) == first

    _build(catalog_path, tmp_path)
    rebuilt = CweRetriever.from_persist_dir(tmp_path, embedder=StubEmbedder())
    assert [
        (r.cwe_id, r.chunk_index, r.text) for r in rebuilt.retrieve(query, top_n=TOP_N, top_k=TOP_K)
    ] == [(r.cwe_id, r.chunk_index, r.text) for r in first]


def test_reranker_reorders_and_scores_descending(built):
    persist_dir, _ = built
    retriever = CweRetriever.from_persist_dir(
        persist_dir, embedder=StubEmbedder(), reranker=StubReranker()
    )
    results = retriever.retrieve("unsanitized sql queries buffer overflow", top_n=TOP_N, top_k=3)
    assert len(results) == 3
    assert results[0].cwe_id == "CWE-1002"
    scores = [r.rerank_score for r in results]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_invalid_queries_and_parameters_raise(retriever):
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("", top_n=TOP_N, top_k=TOP_K)
    with pytest.raises(ValueError, match="empty"):
        retriever.retrieve("   \n\t", top_n=TOP_N, top_k=TOP_K)
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("buffer", top_n=3, top_k=5)
    with pytest.raises(ValueError, match="top_n/top_k"):
        retriever.retrieve("buffer", top_n=0, top_k=0)
    with pytest.raises(TypeError):
        retriever.retrieve("buffer")


def test_very_long_query_is_handled(retriever):
    long_query = "buffer overflow " * 20_000
    results = retriever.retrieve(long_query, top_n=TOP_N, top_k=TOP_K)
    assert len(results) == TOP_K
    assert retriever.query_is_truncated(long_query) is True
    assert retriever.query_is_truncated("buffer overflow") is False


def test_distinct_cwe_count():
    def chunk(cwe_id, index):
        return RetrievedChunk(
            text="t",
            cwe_id=cwe_id,
            cwe_name="n",
            chunk_index=index,
            vector_distance=0.1,
            rerank_score=None,
        )

    assert distinct_cwe_count([]) == 0
    assert distinct_cwe_count([chunk("CWE-1", 0), chunk("CWE-1", 1), chunk("CWE-2", 0)]) == 2


def test_cwe_recall_separates_hit_from_miss_and_both_rankings(tmp_path, built):
    """Gold-CWE recall must reflect the actual retrieval, not a constant."""
    import json

    from vulndetect.rag.measure import measure_cwe_recall

    persist_dir, _ = built

    def record(cwe: str, text: str) -> dict:
        return {
            "id": f"{cwe}:{text[:6]}",
            "cwe": cwe,
            "cve_id": "CVE-2020-0001",
            "lang": "C",
            "commit_message": text,
            "func_before": text,
            "code": text,
            "func_after": text,
            "label": "Vulnerable",
            "project": "p",
            "split": "val",
            "vul": 1,
        }

    rows = [
        record("CWE-1001", "buffer overflow buffer overflow"),
        record("CWE-1002", "buffer overflow buffer overflow"),
    ]
    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "val.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    report = measure_cwe_recall(
        ds,
        CweRetriever.from_persist_dir(persist_dir, embedder=StubEmbedder()),
        split="val",
        query_version="authors-v1",
        top_n=TOP_N,
        ks=(1, TOP_K, TOP_N),
    )

    assert report["n"] == 2
    assert report["query_version"] == "authors-v1"
    assert report["reranker"] is None
    assert report["recall_at_k"]["by_vector_distance"]["1"] == 0.5
    assert report["recall_at_k"]["after_reranking"]["1"] == 0.5
    assert report["recall_at_k"]["after_reranking"][str(TOP_N)] == 1.0
    by_k = report["recall_at_k"]["after_reranking"]
    assert by_k["1"] <= by_k[str(TOP_K)] <= by_k[str(TOP_N)]
    tabs = report["by_query_truncation"]
    assert tabs["truncated"]["n"] + tabs["not_truncated"]["n"] == 2

    with pytest.raises(ValueError, match="must be <= top_n"):
        measure_cwe_recall(
            ds,
            CweRetriever.from_persist_dir(persist_dir, embedder=StubEmbedder()),
            query_version="authors-v1",
            top_n=TOP_N,
            ks=(TOP_N + 1,),
        )


def test_query_truncation_measures_the_configured_query_form(tmp_path, raw_df, base_cfg):
    """Checks that query truncation is measured for the configured query version."""
    from vulndetect.data.build import build_dataset, to_records, write_jsonl
    from vulndetect.rag.measure import measure_query_truncation

    records = to_records(build_dataset(raw_df, base_cfg))
    for split in ("train", "val", "test"):
        rows = [r for r in records if r["split"] == split]
        assert rows, f"fixture produced no {split} rows"
        write_jsonl(rows, tmp_path / "ds" / f"{split}.jsonl")

    code_v1 = measure_query_truncation(tmp_path / "ds", StubEmbedder(), "code-v1")
    authors = measure_query_truncation(tmp_path / "ds", StubEmbedder(), "authors-v1")

    assert code_v1["query_version"] == "code-v1"
    assert authors["query_version"] == "authors-v1"
    assert authors["token_count_percentiles"]["p50"] > code_v1["token_count_percentiles"]["p50"]

    with pytest.raises(ValueError, match="unknown query_version"):
        measure_query_truncation(tmp_path / "ds", StubEmbedder(), "nope")
