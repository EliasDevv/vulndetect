"""RAG prompt assembly using the authors' released classification prompt with retrieved context."""

from __future__ import annotations

from vulndetect.data.schema import NA, Record
from vulndetect.rag.retriever import RetrievedChunk

AUTHORS_RAG_PROMPT_VERSION = "authors-rag-v1"
RAG_PROMPT_VERSIONS = (AUTHORS_RAG_PROMPT_VERSION,)

AUTHORS_RAG_TEMPLATE = """You are an expert in cybersecurity and software vulnerability detection. Given a code snippet with modifications, analyze whether the change fixes or mitigates a vulnerability. Use the commit message, function before and after changes, and CVE information (if available) to determine vulnerability.

1) Analyze the Commit Message
   Does the commit message mention security or vulnerability related terms like buffer overflow, fix, CVE, or patch?
2) Compare Function Before and After Changes
   What specific modifications were made to the code? Do the changes address known security issues, or do they introduce new risks?
3) Check for CVE/CWE References
   If a CVE is mentioned, what vulnerability does it relate to? If a CWE is provided, does the change align with known vulnerability patterns?
4) Assess the Overall Security Impact
   Does the modification fix an existing issue, introduce a new vulnerability, or is it neutral?

Retrieved domain knowledge from the MITRE CWE database:
{context}

Here is the data:
- Commit Message: {commit_message}
- Code Before Change: {func_before}
- Code After Change: {func_after}
- CVE ID (if available): {cve_id}
- CWE ID (if available): {cwe_id}
- Project: {project}
- Programming Language: {lang}

Question:
Does this code change fix or mitigate a vulnerability? Answer with Vulnerable or Not Vulnerable and provide reasoning."""

AUTHORS_NO_CONTEXT = "No relevant knowledge retrieved."
AUTHORS_MAX_CONTEXT_CHARS = 6000
AUTHORS_MAX_CODE_CHARS = 6000


def authors_truncate(text: str | None, limit: int) -> str:
    """Cut text at ``limit`` chars and append a truncation marker (authors' truncate())."""
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def render_authors_context(
    chunks: list[RetrievedChunk], max_chars: int = AUTHORS_MAX_CONTEXT_CHARS
) -> str:
    """Render chunks as ``[i] (CWE-id) text`` blocks, stopping at the first one over the char budget."""
    if not chunks:
        return AUTHORS_NO_CONTEXT
    blocks: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        block = f"[{i}] ({chunk.cwe_id}) {chunk.text}"
        total += len(block)
        if total > max_chars:
            break
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else AUTHORS_NO_CONTEXT


def render_authors_rag_prompt(record: Record, chunks: list[RetrievedChunk]) -> str:
    """Fill the authors' classification prompt with the record and retrieved context."""
    return AUTHORS_RAG_TEMPLATE.format(
        context=render_authors_context(chunks),
        commit_message=record["commit_message"] or "",
        func_before=authors_truncate(record["func_before"], AUTHORS_MAX_CODE_CHARS),
        func_after=authors_truncate(record["func_after"], AUTHORS_MAX_CODE_CHARS),
        cve_id=record.get("cve_id") or NA,
        cwe_id=record.get("cwe") or NA,
        project=record["project"] or NA,
        lang=record["lang"] or NA,
    )
