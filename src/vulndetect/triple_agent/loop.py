"""The Triple-Agent loop: the GAN loop with the Dual-Agent as detecting party."""

from __future__ import annotations

from pathlib import Path

from vulndetect.common import load_config
from vulndetect.gan import loop as gan_loop
from vulndetect.triple_agent.party import party_factory_from_config

LOOP_APPROACH = "triple_agent_loop"


def run_loop(config_path: str | Path, *, generator_model=None, detector_model=None) -> dict:
    """Run Phase A and Phase B with the Detector -> Validator chain as M_D; return the manifest."""
    factory = party_factory_from_config(load_config(config_path))
    return gan_loop.run_loop(
        config_path,
        generator_model=generator_model,
        detector_model=detector_model,
        party_factory=factory,
        approach=LOOP_APPROACH,
    )
