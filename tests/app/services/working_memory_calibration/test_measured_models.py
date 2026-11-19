"""Cache storage suffixes must not hide a measured model from reset or attribution."""

import pytest

from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import CalibrationPoint
from invokeai.app.services.working_memory_calibration.working_memory_calibration_fits import measured_models


@pytest.mark.parametrize(
    "cache_key,owner",
    [
        ("model", "model"),
        ("model:vae", "model"),
        ("namespace:model:transformer", "namespace:model"),
        ("namespace:model:unknown", "namespace:model:unknown"),
        ("namespace:float32", "namespace:float32"),
        ("model:text_encoder_2:fp8-float32", "model"),
        ("model:text_encoder_3:bfloat16", "model"),
        ("namespace:model:transformer:storage=fp8-float16", "namespace:model"),
        ("namespace:model:storage=full-float32", "namespace:model"),
        ("model:transformer:storage=fp8-bfloat16-layerwise", "model"),
        ("studio:as_text_llm", "studio:as_text_llm"),
        ("studio:as_text_llm:vae", "studio:as_text_llm"),
        ("namespace:model:text_encoder:fp8-bfloat16", "namespace:model:text_encoder:fp8-bfloat16"),
        ("studio:text_encoder:float32", "studio:text_encoder:float32"),
        ("model:text_encoder:full-float32", "model:text_encoder:full-float32"),
        ("model:text_encoder:fp8-bfloat16-layerwise", "model:text_encoder:fp8-bfloat16-layerwise"),
        ("namespace:model:text_encoder:as_qwen3_5_encoder:storage=full-float32", "namespace:model"),
        ("namespace:model:as_unrecognized_type", "namespace:model:as_unrecognized_type"),
        ("model:transformer:storage=unknown", "model:transformer:storage=unknown"),
        ("model:unknown:fp8-float32", "model:unknown:fp8-float32"),
    ],
)
def test_measurement_attribution_understands_only_known_cache_suffixes(cache_key, owner):
    point = CalibrationPoint.model_construct(model_key=cache_key, run_id="run")
    assert measured_models(point, {}) == frozenset({owner})
    assert measured_models(point, {"run": "calibrated"}) == frozenset({owner, "calibrated"})
