"""Shared infrastructure for all approaches: prompts, parsing, predictions and config."""

from __future__ import annotations

from .config import config_sha256, load_config, set_seed
from .parsing import DEFAULT_PARSER_VERSION, PARSERS, get_parser, parse_label
from .predictions import (
    PREDICTION_FIELDS,
    Prediction,
    make_prediction,
    read_predictions,
    write_predictions,
    write_run,
)
from .prompt import (
    PROMPT_TEMPLATE,
    PROMPT_VARIANTS,
    PROMPT_VERSION,
    PromptVariant,
    get_prompt_variant,
    render_prompt,
)

__all__ = [
    "DEFAULT_PARSER_VERSION",
    "PARSERS",
    "PREDICTION_FIELDS",
    "PROMPT_TEMPLATE",
    "PROMPT_VARIANTS",
    "PROMPT_VERSION",
    "Prediction",
    "PromptVariant",
    "config_sha256",
    "get_parser",
    "get_prompt_variant",
    "load_config",
    "make_prediction",
    "parse_label",
    "read_predictions",
    "render_prompt",
    "set_seed",
    "write_predictions",
    "write_run",
]
