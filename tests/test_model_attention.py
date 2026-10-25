"""Tests that grouped-query KV expansion is forced in transformers' SDPA integration."""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from vulndetect.common.model import force_kv_expansion  # noqa: E402


def test_force_kv_expansion_disables_gqa_passthrough():
    import transformers.integrations.sdpa_attention as sdpa

    assert force_kv_expansion() is True
    assert sdpa.use_gqa_in_sdpa(None, object(), object()) is False
    assert sdpa.use_gqa_in_sdpa(object(), object(), object()) is False
