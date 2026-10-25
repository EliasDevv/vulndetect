"""Prompt templates and renderers for the GAN roles and the memory summarisation."""

from __future__ import annotations

from vulndetect.common.prompt import render_prompt
from vulndetect.data.schema import NA, Record
from vulndetect.gan.parsing import (
    M_COMMIT_MESSAGE,
    M_DECEPTION_NOTES,
    M_EVIDENCE,
    M_LESSON,
    M_MODIFIED_FUNCTION,
    M_NEW_STRATEGIES,
    M_WENT_WRONG,
    marker,
)

GENERATOR_PROMPT_VERSION = "gan-generator-v1"
DETECTOR_BLOCK_VERSION = "gan-detector-block-v1"
STRATEGY_UPDATE_VERSION = "gan-strategy-update-v1"
REFLECTOR_PROMPT_VERSION = "gan-reflector-v1"
SUMMARIZE_PROMPT_VERSION = "gan-summarize-v1"

PROMPT_VERSIONS = {
    "generator": GENERATOR_PROMPT_VERSION,
    "detector_block": DETECTOR_BLOCK_VERSION,
    "strategy_update": STRATEGY_UPDATE_VERSION,
    "reflector": REFLECTOR_PROMPT_VERSION,
    "summarize": SUMMARIZE_PROMPT_VERSION,
}

CWE_NAMES = {
    "CWE-119": "Improper Restriction of Operations within the Bounds of a Memory Buffer",
    "CWE-20": "Improper Input Validation",
    "CWE-399": "Resource Management Errors",
    "CWE-264": "Permissions, Privileges, and Access Controls",
    "CWE-200": "Exposure of Sensitive Information to an Unauthorized Actor",
}


def cwe_label(cwe: str) -> str:
    """Format a CWE id with its name, e.g. ``"CWE-20 (Improper Input Validation)"``."""
    name = CWE_NAMES.get(cwe)
    return f"{cwe} ({name})" if name else cwe


GENERATOR_HEADER = """\
You are an expert in offensive software security, acting as an adversarial code
generator. Below is a C/C++ function that does not currently contain the weakness
class named in the metadata. Write a modified version of that function which DOES
contain an instance of that weakness class, and make the flaw as hard to notice as
possible for an automated code reviewer.

Requirements:
1) Keep the function's name, signature, and overall purpose unchanged. The modified
   function must still read like ordinary production code of this project.
2) Introduce exactly one instance of the weakness class given below. The flaw must be
   real: there must be an input or program state under which the modified function
   behaves unsafely.
3) Do not add comments, identifiers, or markers that hint at the flaw, and do not
   mention the weakness anywhere in the code.
4) Keep the change small. A reviewer diffing your version against the original should
   see a plausible, unremarkable modification.
5) Also write a commit message of the kind a developer would write for this change. It
   must present the change as routine work (refactoring, cleanup, performance, feature)
   and must not mention security, vulnerabilities, or the weakness.
"""

GENERATOR_STRATEGY_HEADER = """
Strategies you compiled in earlier rounds. Apply the ones that fit this function:
"""

GENERATOR_FORMAT = f"""
Reply in exactly this format, with nothing before or after it:
{marker(M_COMMIT_MESSAGE)}
<one or two lines>
{marker(M_MODIFIED_FUNCTION)}
<the complete modified function, code only, no markdown fences>
{marker(M_DECEPTION_NOTES)}
<two to four sentences: what you changed, why it is unsafe, and what makes the flaw
hard to notice>
"""


def render_generator_prompt(source: Record, strategies: str) -> str:
    """Render the Generator prompt that turns a clean function into a vulnerable one (eq. 1)."""
    parts = [GENERATOR_HEADER]
    if strategies:
        parts += [GENERATOR_STRATEGY_HEADER, strategies, "\n"]
    parts += [
        "\nHere is the data:\n",
        "- Project: ",
        source["project"] or NA,
        "\n- Programming Language: ",
        source["lang"] or NA,
        "\n- Weakness class to introduce: ",
        cwe_label(source["cwe"]),
        "\n- Function:\n",
        source["func_before"],
        "\n",
        GENERATOR_FORMAT,
    ]
    return "".join(parts)


GENERATOR_UPDATE_HEADER = """\
You are an expert in offensive software security, maintaining a written playbook of
strategies for hiding vulnerabilities in code so that automated reviewers miss them.

In the last round your modified function was correctly identified as vulnerable. The
reviewer's own assessment below states how it reached that conclusion. Work out what
your playbook is still missing.
"""

GENERATOR_UPDATE_PLAYBOOK_HEADER = "\nYour current playbook:\n"
SEP_ORIGINAL_FUNCTION = "\n--- The original function ---\n"
SEP_MODIFIED_FUNCTION = "\n\n--- Your modified function ---\n"
SEP_DECEPTION_NOTES = "\n\n--- Your notes on how you hid the flaw ---\n"
SEP_REVIEWER_ASSESSMENT = "\n\n--- The reviewer's assessment ---\n"

# Placeholders: {k} max new entries, {w} word cap per entry.
STRATEGY_OUTPUT_INSTRUCTION = """

Write at most {k} new {noun} entries. Each entry must be a single imperative
sentence of at most {w} words, must generalise to other {scope} (do not mention this
function, its identifiers, or this project), and must not repeat an entry already in
the list. If nothing new can be learned, write NONE.

{new_strategies_marker}
- <entry>
"""


def _output_instruction(k: int, max_words: int, noun: str, scope: str) -> str:
    return STRATEGY_OUTPUT_INSTRUCTION.format(
        k=k,
        w=max_words,
        noun=noun,
        scope=scope,
        new_strategies_marker=marker(M_NEW_STRATEGIES),
    )


def render_generator_update_prompt(
    source: Record,
    generated_function: str,
    deception_notes: str,
    detector_output: str,
    strategies: str,
    *,
    max_new: int,
    max_words: int,
) -> str:
    """Render the prompt in which the Generator derives new strategies for S_G (eq. 3)."""
    parts = [GENERATOR_UPDATE_HEADER]
    if strategies:
        parts += [GENERATOR_UPDATE_PLAYBOOK_HEADER, strategies, "\n"]
    parts += [
        SEP_ORIGINAL_FUNCTION,
        source["func_before"],
        SEP_MODIFIED_FUNCTION,
        generated_function,
        SEP_DECEPTION_NOTES,
        deception_notes,
        SEP_REVIEWER_ASSESSMENT,
        detector_output,
        _output_instruction(max_new, max_words, "playbook", "functions"),
    ]
    return "".join(parts)


DETECTOR_HEADER = """\
You have compiled the following detection strategies from earlier cases. Consider the
ones that apply to the case below, but base your verdict on the evidence in the data —
a strategy is a hint, not a verdict.
"""

DETECTOR_BLOCK_SEPARATOR = "\n---\n\n"


def render_detector_prompt(record: Record, strategies: str) -> str:
    """Prepend the strategy block to the unchanged baseline prompt (eqs. 2, 5 and inference)."""
    if not strategies:
        return render_prompt(record)
    return DETECTOR_HEADER + strategies + DETECTOR_BLOCK_SEPARATOR + render_prompt(record)


UPDATE_KINDS = ("adversary", "reflection")

DETECTOR_UPDATE_HEADER = """\
You are an expert in cybersecurity and software vulnerability detection, maintaining a
written list of detection strategies: the lessons you apply when judging whether code
contains a vulnerability.

You judged the case below incorrectly.
"""

DETECTOR_UPDATE_ADVERSARY_NOTE = """\
The code was deliberately modified to contain a vulnerability, and the notes from its
author explain how the flaw was hidden from you.
"""

DETECTOR_UPDATE_REFLECTION_NOTE = """\
The reflection below explains where your reasoning went wrong and what the data
actually shows.
"""

DETECTOR_UPDATE_STRATEGY_HEADER = "\nYour current strategies:\n"
SEP_CASE = "\n--- The case as it was presented to you ---\n"
SEP_OWN_ASSESSMENT = "\n\n--- Your assessment ---\n"
SEP_EVIDENCE = {
    "adversary": "\n\n--- How the flaw was hidden ---\n",
    "reflection": "\n\n--- Reflection on your mistake ---\n",
}


def render_detector_update_prompt(
    record: Record,
    detector_output: str,
    evidence: str,
    kind: str,
    strategies: str,
    *,
    max_new: int,
    max_words: int,
) -> str:
    """Render the prompt in which the Detector derives new strategies for S_D (eqs. 4 and 7)."""
    if kind not in UPDATE_KINDS:
        raise ValueError(f"unknown update kind {kind!r}; allowed: {UPDATE_KINDS}")
    note = (
        DETECTOR_UPDATE_ADVERSARY_NOTE if kind == "adversary" else DETECTOR_UPDATE_REFLECTION_NOTE
    )
    parts = [DETECTOR_UPDATE_HEADER, note]
    if strategies:
        parts += [DETECTOR_UPDATE_STRATEGY_HEADER, strategies, "\n"]
    parts += [
        SEP_CASE,
        render_prompt(record),
        SEP_OWN_ASSESSMENT,
        detector_output,
        SEP_EVIDENCE[kind],
        evidence,
        _output_instruction(max_new, max_words, "strategy", "cases"),
    ]
    return "".join(parts)


REFLECTOR_HEADER = """\
You are a senior security reviewer acting as a reflector. A junior analyst was given
the task below and answered "{pred}", but the correct answer is "{gold}". Explain the
mistake so that the analyst can learn from it. Do not be vague: name the specific step
in the analyst's reasoning that failed.
"""

SEP_ANALYST_TASK = "\n--- The task given to the analyst (including the data) ---\n"
SEP_ANALYST_ASSESSMENT = "\n\n--- The analyst's assessment ---\n"

REFLECTOR_FORMAT = """

Answer in exactly this format, with nothing before or after it:
{went_wrong_marker}
<which claim or assumption in the assessment is unsupported or wrong>
{evidence_marker}
<what in the provided data supports "{gold}" — refer to concrete code, the commit
message, or the CWE information>
{lesson_marker}
<one imperative sentence of at most {w} words that generalises to other cases>
"""

NO_VERDICT = "no clear verdict"


def render_reflector_prompt(
    record: Record,
    detector_output: str,
    pred_label: str | None,
    *,
    max_words: int,
) -> str:
    """Render the Reflector prompt that explains one incorrect training prediction (eq. 6)."""
    gold = record["label"]
    parts = [
        REFLECTOR_HEADER.format(pred=pred_label or NO_VERDICT, gold=gold),
        SEP_ANALYST_TASK,
        render_prompt(record),
        SEP_ANALYST_ASSESSMENT,
        detector_output,
        REFLECTOR_FORMAT.format(
            gold=gold,
            w=max_words,
            went_wrong_marker=marker(M_WENT_WRONG),
            evidence_marker=marker(M_EVIDENCE),
            lesson_marker=marker(M_LESSON),
        ),
    ]
    return "".join(parts)


SUMMARIZE_PROMPT = """\
Condense the following list of {n} {role} strategies into at most {m} entries
without losing any distinct lesson. Merge entries that say the same thing, drop
entries that are special cases of another entry, and keep the wording imperative and
at most {w} words per entry.

Current strategies:
"""


def render_summarize_prompt(
    entries: list[str], role: str, *, max_entries: int, max_words: int
) -> str:
    """Render the prompt that condenses an over-full memory (``summarize`` policy)."""
    numbered = "\n".join(f"{i}. {e}" for i, e in enumerate(entries, start=1))
    return (
        SUMMARIZE_PROMPT.format(n=len(entries), role=role, m=max_entries, w=max_words)
        + numbered
        + "\n\n"
        + marker(M_NEW_STRATEGIES)
        + "\n- <entry>\n"
    )
