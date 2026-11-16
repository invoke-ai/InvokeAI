"""The MOSS-Image denoise working-memory estimator: what it reads off the transformer and adds to the fitted term.

The fitted coefficient is held to its measured points by tests/backend/working_memory/test_shipped_coefficients.py;
these cover the analytic terms and the features the estimator records.
"""

from typing import Any

import pytest
import torch

from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.moss_image import working_memory as module
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer
from invokeai.backend.moss_image.working_memory import (
    MOSS_IMAGE_DENOISE_SPEC,
    estimate_moss_image_denoise_working_memory,
)
from invokeai.backend.patches.lora_conversions.moss_image_lora_conversion_utils import (
    lora_model_from_moss_image_state_dict,
)
from invokeai.backend.working_memory.coefficients import shipped_coefficients
from invokeai.backend.working_memory.platform import platform_for


def _transformer() -> MOSSImageTransformer:
    return MOSSImageTransformer(dim=64, ffn_dim=96, num_heads=2, num_layers=1, text_dim=24, freq_dim=16).eval()


def test_the_score_matrix_is_priced_for_the_whole_attended_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: dict[str, Any] = {}

    def score_matrix_bytes(**kwargs: Any) -> int:
        asked.update(kwargs)
        return 12345

    monkeypatch.setattr(module, "sdpa_score_matrix_bytes", score_matrix_bytes)
    device = torch.device("cpu")

    estimate = estimate_moss_image_denoise_working_memory(
        _transformer(),
        height=96,
        width=64,
        text_tokens=7,
        model_format=ModelFormat.Checkpoint,
        device=device,
        dtype=torch.bfloat16,
    )

    # 6x4 image tokens and the prompt's 7, attended together by 2 heads of 32.
    assert {key: asked[key] for key in ("num_heads", "head_dim", "seq_len", "dtype")} == {
        "num_heads": 2,
        "head_dim": 32,
        "seq_len": 31,
        "dtype": torch.bfloat16,
    }
    k_token = shipped_coefficients(MOSS_IMAGE_DENOISE_SPEC, platform_for(device)).values["k_token"]
    assert estimate.features == {
        "height": 96,
        "width": 64,
        "text_tokens": 7,
        "score_matrix_bytes": 12345,
        "dequant_bytes": 0,
        "lora_sidecar_bytes": 0,
    }
    assert estimate.reservation_bytes == int(31 * k_token + 12345)


def test_a_quantized_build_adds_its_dequant_transient() -> None:
    values = {"k_token": 1000.0}
    features = {"height": 32, "width": 32, "text_tokens": 4, "score_matrix_bytes": 0, "dequant_bytes": 5000}

    terms = MOSS_IMAGE_DENOISE_SPEC.terms(features, values)

    assert {term.name: term.bytes for term in terms if term.included} == {
        "activations": 8 * 1000.0,
        "attention_scores": 0,
        "dequant_transient": 5000,
        "lora_sidecar": 0,
    }


def test_points_measured_without_loras_have_no_lora_term() -> None:
    """The shipped points carry no `lora_sidecar_bytes`: a point without it prices no LoRA."""
    features = {"height": 32, "width": 32, "text_tokens": 4, "score_matrix_bytes": 0, "dequant_bytes": 0}

    terms = MOSS_IMAGE_DENOISE_SPEC.terms(features, {"k_token": 1000.0})

    assert {term.name: term.bytes for term in terms if term.included}["lora_sidecar"] == 0


def _lora(module: str, rank: int = 2, in_features: int = 64, out_features: int = 64):
    state_dict = {
        f"diffusion_model.{module}.lora_down.weight": torch.zeros(rank, in_features),
        f"diffusion_model.{module}.lora_up.weight": torch.zeros(out_features, rank),
    }
    return lora_model_from_moss_image_state_dict(state_dict, name="test")


def test_loras_are_priced_by_the_rows_the_forward_sends_through_them() -> None:
    """On a GGUF build every layer is a sidecar. A block's Linear reads a row per token; the modulations and the timestep
    MLP read the timestep embedding, a row per sample."""
    transformer = _transformer()
    kwargs: dict[str, Any] = {
        "height": 96,
        "width": 64,
        "text_tokens": 7,
        "model_format": ModelFormat.GGUFQuantized,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }

    def priced(*loras) -> int:
        return estimate_moss_image_denoise_working_memory(transformer, loras=list(loras), **kwargs).features[
            "lora_sidecar_bytes"
        ]

    tokens = 6 * 4 + 7
    assert priced() == 0
    # The LoRA's own 2 x 64 + 64 x 2 elements, and a row of 64 outputs and 2 ranks per token.
    assert priced(_lora("layers.0.proj.q")) == (256 + tokens * (64 + 2)) * 4
    # shared_block_modulation: 6 x 64 outputs from one row.
    assert priced(_lora("shared_block_modulation", out_features=6 * 64)) == (2 * 64 + 6 * 64 * 2 + (6 * 64 + 2)) * 4


def test_a_forward_with_a_lora_is_off_the_plain_line_the_fit_reads() -> None:
    """The activations line is fitted to forwards with no analytic term, as a score matrix or a dequantization keeps a
    point off it."""
    features = {"height": 32, "width": 32, "text_tokens": 4, "score_matrix_bytes": 0, "dequant_bytes": 0}

    plain = MOSS_IMAGE_DENOISE_SPEC.line(features)
    with_lora = MOSS_IMAGE_DENOISE_SPEC.line({**features, "lora_sidecar_bytes": 4096})

    assert plain is not None and plain.fits
    assert with_lora is not None and not with_lora.fits
    assert with_lora.label.endswith("lora_sidecar_bytes=4096")
