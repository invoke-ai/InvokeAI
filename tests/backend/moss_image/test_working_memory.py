"""The MOSS-Image denoise working-memory estimator: what it reads off the transformer and adds to the fitted term.

The fitted coefficient is held to its measured points by tests/backend/working_memory/test_shipped_coefficients.py;
these cover the analytic terms and the features the estimator records.
"""

from typing import Any

import pytest
import torch

from invokeai.backend.moss_image import working_memory as module
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer
from invokeai.backend.moss_image.working_memory import (
    MOSS_IMAGE_DENOISE_SPEC,
    estimate_moss_image_denoise_working_memory,
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
        _transformer(), height=96, width=64, text_tokens=7, device=device, dtype=torch.bfloat16
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
    }
