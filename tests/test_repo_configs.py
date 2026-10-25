"""Guards the shipped run configs against drift.

Checks the smoke, val and test stages per approach and the frozen official pipelines.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

CONFIGS = Path(__file__).parents[1] / "configs"
APPROACHES = ("baseline", "rag", "dual_agent")


def load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("approach", APPROACHES)
def test_official_config_logs_and_uses_test_split(approach):
    cfg = load(approach)
    assert cfg["experiment"] == approach
    assert cfg["dataset"]["split"] == "test"
    assert cfg["output"]["predictions_dir"] == f"results/predictions/{approach}"
    assert cfg["output"].get("log_experiments", True) is True


@pytest.mark.parametrize("approach", APPROACHES)
def test_val_config_matches_official(approach):
    base, variant = load(approach), load(f"{approach}_val")
    assert variant["experiment"] == f"{approach}_val"
    assert variant["seed"] == base["seed"]
    assert variant["model"] == base["model"]
    assert variant["dataset"] == {**base["dataset"], "split": "val"}
    assert variant["output"] == {
        **base["output"],
        "predictions_dir": base["output"]["predictions_dir"] + "_val",
        "log_experiments": False,
    }
    assert variant.get("retrieval") == base.get("retrieval")
    assert variant.get("dual_agent") == base.get("dual_agent")
    assert variant.get("prompt") == base.get("prompt")
    assert variant.get("parser_version") == base.get("parser_version")
    assert variant.get("eval") == base.get("eval")
    assert set(variant) == set(base)


@pytest.mark.parametrize("approach", APPROACHES)
def test_smoke_config_matches_official(approach):
    base, variant = load(approach), load(f"{approach}_smoke")
    assert variant["experiment"] == f"{approach}_smoke"
    assert variant["seed"] == base["seed"]
    assert "batch_size" in variant["model"]
    assert {k: v for k, v in variant["model"].items() if k != "batch_size"} == {
        k: v for k, v in base["model"].items() if k != "batch_size"
    }
    assert variant["dataset"] == {**base["dataset"], "split": "val"}
    assert variant["output"] == {
        **base["output"],
        "predictions_dir": base["output"]["predictions_dir"] + "_smoke",
        "log_experiments": False,
    }
    assert variant.get("retrieval") == base.get("retrieval")
    assert variant.get("dual_agent") == base.get("dual_agent")
    assert variant.get("prompt") == base.get("prompt")
    assert variant.get("parser_version") == base.get("parser_version")
    assert variant.get("eval") == base.get("eval")
    assert set(variant) == set(base)


def test_sft_val_config_matches_official():
    """Checks that sft_val mirrors sft.yaml apart from split, output and experiment."""
    base, variant = load("sft"), load("sft_val")
    assert variant["experiment"] == "sft_val"
    assert variant["seed"] == base["seed"]
    assert variant["model"] == base["model"]
    assert variant["training"] == base["training"]
    assert variant["dataset"] == {**base["dataset"], "split": "val"}
    assert variant["output"] == {
        **base["output"],
        "predictions_dir": base["output"]["predictions_dir"] + "_val",
        "log_experiments": False,
    }
    assert variant.get("parser_version") == base.get("parser_version")
    assert variant.get("eval") == base.get("eval")
    assert set(variant) == set(base)


SFT_CONFIGS = ("sft", "sft_val", "sft_smoke", "sft_seed43", "sft_seed44", "sft_extended")


@pytest.mark.parametrize("name", SFT_CONFIGS)
def test_sft_training_precision_is_fp16(name):
    """Checks that every SFT config trains with float16 compute dtype."""
    assert load(name)["training"]["quantization"]["compute_dtype"] == "float16"


def test_official_configs_pin_author_alignment():
    """Checks the prompt, parser and decoding settings of the official configs."""
    base = load("baseline")
    assert base["prompt"]["version"] == "fig1-v3"
    assert base["parser_version"] == "d09-v2"
    assert base["eval"]["unparseable"] == "drop"
    assert base["model"]["temperature"] == 0.6
    assert base["model"]["top_p"] == 0.9
    assert base["model"]["max_new_tokens"] == 512

    rag = load("rag")
    assert rag["prompt"]["version"] == "authors-rag-v1"
    assert rag["retrieval"]["query_version"] == "authors-v1"
    assert rag["retrieval"]["embedding_prefixes"] is True
    assert rag["model"]["temperature"] == 0.6
    assert rag["model"]["top_p"] == 0.9
    assert rag["model"]["max_new_tokens"] == 512

    da = load("dual_agent")
    assert da["dual_agent"]["prompt_set"] == "authors-v1"
    assert da["model"]["temperature"] == 0.6
    assert da["model"]["max_new_tokens"] == 512

    sft = load("sft")
    assert sft["training"]["format"] == "authors-alpaca"
    assert sft["training"]["seed"] == 3407
    assert sft["training"]["max_seq_length"] == 2048
    assert sft["training"]["completion_only_loss"] is False
    assert sft["model"]["temperature"] == 0.6
    assert sft["model"]["top_p"] == 0.9
    assert sft["model"]["max_new_tokens"] == 256

    ablation = load("sft_chat_ablation")
    assert ablation["training"]["format"] == "chat"
    assert ablation["training"]["completion_only_loss"] is True
    assert ablation["parser_version"] == "d09-v2"


@pytest.mark.parametrize("approach", APPROACHES)
def test_ladder_stages_never_share_predictions_dir(approach):
    dirs = [load(f"{approach}{s}")["output"]["predictions_dir"] for s in ("", "_smoke", "_val")]
    assert len(set(dirs)) == 3


def test_no_diagnostic_config_runs_on_test_or_logs():
    """Checks that every smoke and val config uses the val split and does not log."""
    paths = (
        sorted(CONFIGS.glob("*_smoke.yaml"))
        + sorted(CONFIGS.glob("*_val.yaml"))
        + sorted(CONFIGS.glob("*_val_*.yaml"))
    )
    assert len(paths) >= 7
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["dataset"]["split"] == "val", path.name
        assert cfg["output"]["log_experiments"] is False, path.name


PINNED_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"


def _configs_with_model() -> list[Path]:
    return [p for p in sorted(CONFIGS.glob("*.yaml")) if "model" in yaml.safe_load(p.read_text())]


def test_every_model_block_pins_the_checkpoint_revision():
    """Checks that every model block pins the checkpoint revision."""
    paths = _configs_with_model()
    assert len(paths) >= 20
    for path in paths:
        model = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
        assert model["name"] == "meta-llama/Llama-3.2-3B-Instruct", path.name
        assert model["revision"] == PINNED_REVISION, path.name


def test_sampling_configs_state_top_p_explicitly():
    """Checks that every sampling config sets top_p explicitly."""
    for path in _configs_with_model():
        model = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
        if float(model.get("temperature", 0.0)) > 0:
            assert model.get("top_p") == 0.9, path.name


def test_data_config_keeps_bigvul_cpp_spelling():
    """Checks that the language filter includes the Big-Vul spelling cpp."""
    cfg = yaml.safe_load((CONFIGS / "data.yaml").read_text(encoding="utf-8"))
    assert {lang.lower() for lang in cfg["lang_filter"]} == {"c", "c++", "cpp"}


def test_baseline_date_ablation_differs_only_in_the_chat_date():
    """Checks that the date ablation differs from baseline_val only in the chat date."""
    base, variant = load("baseline_val"), load("baseline_val_date")
    assert variant["experiment"] == "baseline_val_date"
    assert variant["model"].pop("chat_date_string") == "04 Sep 2026"
    assert "chat_date_string" not in base["model"]
    assert variant["model"] == base["model"]
    assert variant["output"]["predictions_dir"] == "results/predictions/baseline_val_date"
    assert variant["output"]["log_experiments"] is False
    for key in ("prompt", "parser_version", "eval", "dataset", "seed"):
        assert variant[key] == base[key], key


FROZEN_BASELINE = {
    "config_sha256": "6aaeb9062ec28f55accee7b8c54b95d21f2c457121e747b1752ef6edc8d3172c",
    "run_config_sha256": "d26fa2fae19d5a3e70a76e0baaaa4dc832ce5eb55ea4c2483d83867c686ddcb7",
    "prompt_sha256": "4edf12f5feddfef49618befaef244d8389a55e30b9d66858ec763e253a2bcb4c",
    "parser_sha256": "1eccd74f00cceedd51f6a585ee299d750b7006863813cadbd7006426e153f2ab",
    "split_reference_sha256": "f1b0a598a90e00f3be2c90351bc65ff3576edfa08564ae0814afe001d3ff3e80",
    "chat_date_string": "26 Jul 2024",
}
_UNFREEZE = "baseline is frozen: revert the change, or re-pin the frozen values deliberately"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_baseline_is_frozen():
    """Checks the baseline config, prompt, parser, chat date and split against pinned hashes."""
    from vulndetect.common import parsing, prompt
    from vulndetect.common.model import CHAT_DATE_STRING

    config = (CONFIGS / "baseline.yaml").read_bytes()
    assert _sha256(config) == FROZEN_BASELINE["config_sha256"], _UNFREEZE
    assert _sha256(prompt.PROMPT_TEMPLATE.encode("utf-8")) == FROZEN_BASELINE["prompt_sha256"], (
        _UNFREEZE
    )
    rules = json.dumps(
        [
            parsing._ANSWER_RE.pattern,
            parsing._REFUSAL_RE.pattern,
            parsing._VULN_CUE_RE.pattern,
            parsing._SAFE_CUE_RE.pattern,
            parsing._CONCLUSION_RE.pattern,
            parsing._TAIL_CHARS,
            parsing._SENTENCE_BREAKS,
        ]
    )
    assert _sha256(rules.encode("utf-8")) == FROZEN_BASELINE["parser_sha256"], _UNFREEZE
    assert parsing.PARSERS["d09-v2"] is parsing.parse_label, _UNFREEZE
    assert CHAT_DATE_STRING == FROZEN_BASELINE["chat_date_string"], _UNFREEZE

    data = load("data")
    reference = CONFIGS.parent / data["split"]["reference"]
    assert _sha256(reference.read_bytes()) == FROZEN_BASELINE["split_reference_sha256"], _UNFREEZE


def test_baseline_seed_variants_differ_only_in_the_seed():
    """Checks that baseline seed configs differ from baseline.yaml only in the seed."""
    base = load("baseline")
    for path in sorted(CONFIGS.glob("baseline_seed*.yaml")):
        variant = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert variant["seed"] != base["seed"], path.name
        strip = lambda cfg: {k: v for k, v in cfg.items() if k != "seed"}  # noqa: E731
        assert strip(variant) == strip(base), path.name


FROZEN_RAG = {
    "config_sha256": "c52e7df29ae8a2a5ac672a20b38d96bb943e85bbba75bad10e952cefb11db16c",
    "run_config_sha256": "3439bbb1b2b191a91ad21d02db4dd85bd964475a4b02747293f011cd0311121a",
    "template_sha256": "3eb4e63d859f993c2db3c316df74de3204c2ca5295726fda172f43ae8bfcb730",
    "retrieval_query_sha256": "2f5fe82e23193131b48d00570c8715f17aaf99f817d70b2037d0c389ddf579a5",
    "max_context_chars": 6000,
    "max_code_chars": 6000,
    "no_context_line": "No relevant knowledge retrieved.",
    "embedding_prefixes": {"passage": "passage: ", "query": "query: "},
    "distance": "cosine",
    "index": {
        "catalog_version": "4.20",
        "catalog_xml_sha256": "1f5a78bd62e00f86436b4fe32d5034a57e8f0da88e4063b2072b664ae510912e",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "chunk_size": 512,
        "chunk_overlap": 32,
        "n_chunks": 2234,
    },
}
_UNFREEZE_RAG = "RAG is frozen: revert the change, or re-pin the frozen values deliberately"
RUNS = CONFIGS.parent / "results" / "predictions"


def test_rag_is_frozen():
    """Checks the RAG config, templates and retrieval constants against pinned values."""
    from vulndetect.rag import index
    from vulndetect.rag import prompt as rag_prompt
    from vulndetect.rag import run as rag_run

    config = (CONFIGS / "rag.yaml").read_bytes()
    assert _sha256(config) == FROZEN_RAG["config_sha256"], _UNFREEZE_RAG
    template = rag_prompt.AUTHORS_RAG_TEMPLATE.encode("utf-8")
    assert _sha256(template) == FROZEN_RAG["template_sha256"], _UNFREEZE_RAG
    query = rag_run.AUTHORS_RETRIEVAL_QUERY.encode("utf-8")
    assert _sha256(query) == FROZEN_RAG["retrieval_query_sha256"], _UNFREEZE_RAG
    assert rag_prompt.AUTHORS_MAX_CONTEXT_CHARS == FROZEN_RAG["max_context_chars"], _UNFREEZE_RAG
    assert rag_prompt.AUTHORS_MAX_CODE_CHARS == FROZEN_RAG["max_code_chars"], _UNFREEZE_RAG
    assert rag_prompt.AUTHORS_NO_CONTEXT == FROZEN_RAG["no_context_line"], _UNFREEZE_RAG
    assert index.EMBEDDING_PREFIXES == FROZEN_RAG["embedding_prefixes"], _UNFREEZE_RAG
    assert index.DISTANCE == FROZEN_RAG["distance"], _UNFREEZE_RAG


@pytest.mark.parametrize(
    ("approach", "frozen"),
    [("baseline", FROZEN_BASELINE), ("rag", FROZEN_RAG)],
)
def test_official_run_used_the_frozen_config(approach, frozen):
    """Checks that the committed seed-42 test run used the frozen config."""
    manifest = json.loads((RUNS / approach / "seed42" / "manifest.json").read_text())
    assert manifest["config_sha256"] == frozen["run_config_sha256"], approach
    assert manifest["dataset"]["split"] == "test" and manifest["dataset"]["limit"] is None
    if approach == "rag":
        assert manifest["retrieval"]["index"] == FROZEN_RAG["index"], _UNFREEZE_RAG
        assert manifest["retrieval"]["context_mode"] == "retrieved"


def test_rag_seed_variants_differ_only_in_the_seed():
    """Checks that RAG seed configs differ from rag.yaml only in the seed."""
    base = load("rag")
    for path in sorted(CONFIGS.glob("rag_seed*.yaml")):
        variant = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert variant["seed"] != base["seed"], path.name
        strip = lambda cfg: {k: v for k, v in cfg.items() if k != "seed"}  # noqa: E731
        assert strip(variant) == strip(base), path.name


FROZEN_DUAL_AGENT = {
    "config_sha256": "3ba13939d92b60916ca4140ab41c2c9a642eda3300e7c53193a728e65b77c680",
    "run_config_sha256": "9e9cb3acda7c61515e0e316d346e39452f74a4944b4fe0e63a6c27df07557ac8",
    "validator_sha256": "9b28b627f07d38baa852f70d8f9738f3a10cd716eb102a5b73ce24569e940bfc",
}
_UNFREEZE_DUAL = "Dual-Agent is frozen: revert the change, or re-pin the frozen values deliberately"


def test_dual_agent_is_frozen():
    """Checks the Dual-Agent config, prompts and final label rule against pinned values."""
    from vulndetect.data.schema import LABEL_SAFE, LABEL_VULN
    from vulndetect.dual_agent import prompt as dual_prompt
    from vulndetect.dual_agent.run import resolve_final_label

    config = (CONFIGS / "dual_agent.yaml").read_bytes()
    assert _sha256(config) == FROZEN_DUAL_AGENT["config_sha256"], _UNFREEZE_DUAL
    detector = dual_prompt.AUTHORS_DETECTOR_TEMPLATE.encode("utf-8")
    assert _sha256(detector) == FROZEN_BASELINE["prompt_sha256"], _UNFREEZE_DUAL
    validator = dual_prompt.AUTHORS_VALIDATOR_TEMPLATE.encode("utf-8")
    assert _sha256(validator) == FROZEN_DUAL_AGENT["validator_sha256"], _UNFREEZE_DUAL
    assert resolve_final_label(LABEL_VULN, LABEL_SAFE, "validator") == LABEL_SAFE, _UNFREEZE_DUAL
    assert resolve_final_label(LABEL_VULN, None, "validator") is None, _UNFREEZE_DUAL


def test_dual_agent_official_run_used_the_frozen_state():
    """Checks that the committed Dual-Agent seed-42 test run used the frozen state."""
    manifest = json.loads((RUNS / "dual_agent" / "seed42" / "manifest.json").read_text())
    assert manifest["config_sha256"] == FROZEN_DUAL_AGENT["run_config_sha256"]
    assert manifest["dataset"]["split"] == "test" and manifest["dataset"]["limit"] is None
    assert manifest["prompt_version"] == "authors-detector-v1"
    assert manifest["validator_prompt_version"] == "authors-validator-v1"
    run = manifest["dual_agent"]
    assert run["final_label_rule"] == "validator" and run["rounds"] == 1
    assert run["detector_outputs_from"] is None


def test_dual_agent_seed_variants_differ_only_in_the_seed():
    """Checks that Dual-Agent seed configs differ from dual_agent.yaml only in the seed."""
    base = load("dual_agent")
    for path in sorted(CONFIGS.glob("dual_agent_seed*.yaml")):
        variant = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert variant["seed"] != base["seed"], path.name
        strip = lambda cfg: {k: v for k, v in cfg.items() if k != "seed"}  # noqa: E731
        assert strip(variant) == strip(base), path.name


@pytest.mark.parametrize("approach", APPROACHES)
def test_every_replication_approach_has_test_seeds_43_and_44(approach):
    """Checks that each approach has seed 43 and 44 configs."""
    seeds = sorted(load(f"{approach}_seed{s}")["seed"] for s in (43, 44))
    assert seeds == [43, 44]
