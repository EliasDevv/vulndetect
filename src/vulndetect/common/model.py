"""Model loading and text generation with HF transformers, the single inference path."""

from __future__ import annotations

import logging
from typing import Protocol

from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BATCH_SIZE = 8

# Fixed chat-template date so prompts do not depend on the current day.
CHAT_DATE_STRING = "26 Jul 2024"


class TextGenerator(Protocol):
    """Interface approaches require of a model: prompts in, completions out."""

    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate one completion per prompt."""
        ...


def force_kv_expansion() -> bool:
    """Force KV-head expansion before SDPA to avoid an OOM; returns whether it was installed."""
    try:
        import transformers.integrations.sdpa_attention as sdpa
    except ImportError:  # pragma: no cover - other transformers layouts
        return False
    sdpa.use_gqa_in_sdpa = lambda *args, **kwargs: False
    return True


class HFModel:
    """Llama-3.2-3B-Instruct via HF transformers (chat template, dtype from config)."""

    def __init__(self, model_cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.kv_expansion_forced = force_kv_expansion()
        self.name = model_cfg["name"]
        self.revision = model_cfg.get("revision")
        self.date_string = model_cfg.get("chat_date_string", CHAT_DATE_STRING)
        self.temperature = float(model_cfg.get("temperature", 0.0))
        top_p = model_cfg.get("top_p")
        self.top_p = None if top_p is None else float(top_p)
        self.max_new_tokens = int(model_cfg.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
        self.batch_size = int(model_cfg.get("batch_size", DEFAULT_BATCH_SIZE))

        dtype = getattr(torch, model_cfg.get("dtype", "float16"))
        logger.info(
            "loading %s @ %s (%s)", self.name, self.revision, model_cfg.get("dtype", "float16")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, revision=self.revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, revision=self.revision, dtype=dtype, device_map="auto"
        )
        self.adapter = model_cfg.get("adapter")
        if self.adapter:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError(
                    "model.adapter requires peft — install with `uv sync --extra train`"
                ) from e
            logger.info("merging adapter %s into %s (%s)", self.adapter, self.name, dtype)
            self.model = PeftModel.from_pretrained(self.model, self.adapter).merge_and_unload()
        self.model.eval()

    @property
    def context_window(self) -> int:
        """Context length in tokens (``max_position_embeddings``; 131072 for Llama 3.2)."""
        return int(self.model.config.max_position_embeddings)

    def count_tokens(self, prompt: str, *, system: str | None = None, chat: bool = True) -> int:
        """Token count of ``prompt`` exactly as :meth:`generate` would feed it."""
        if chat:
            system_turn = [{"role": "system", "content": system}] if system else []
            text = self.tokenizer.apply_chat_template(
                system_turn + [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                date_string=self.date_string,
            )
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return len(self.tokenizer(prompt, add_special_tokens=True)["input_ids"])

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        system: str | None = None,
        chat: bool = True,
    ) -> list[str]:
        """Batched generation returning only the new text; decoding can be overridden per call."""
        import torch

        temp = self.temperature if temperature is None else float(temperature)
        n_new = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        p_top = self.top_p if top_p is None else float(top_p)
        do_sample = temp > 0

        outputs: list[str] = []
        for start in tqdm(range(0, len(prompts), self.batch_size), desc="generate"):
            batch = prompts[start : start + self.batch_size]
            if chat:
                system_turn = [{"role": "system", "content": system}] if system else []
                texts = [
                    self.tokenizer.apply_chat_template(
                        system_turn + [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                        date_string=self.date_string,
                    )
                    for p in batch
                ]
                add_special = False
            else:
                texts = batch
                add_special = True
            enc = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=add_special
            ).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=n_new,
                    do_sample=do_sample,
                    temperature=temp if do_sample else None,
                    top_p=p_top if (do_sample and p_top is not None) else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1] :]
            outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs


def load_model(model_cfg: dict) -> HFModel:
    """Build the canonical inference model from the config's ``model:`` block."""
    provider = model_cfg.get("provider", "huggingface")
    if provider != "huggingface":
        raise ValueError(f"unsupported provider {provider!r}; only huggingface is supported")
    return HFModel(model_cfg)
