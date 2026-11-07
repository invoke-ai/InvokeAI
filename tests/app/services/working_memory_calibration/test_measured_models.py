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
        ("model:transformer:storage=unknown", "model:transformer:storage=unknown"),
        ("model:unknown:fp8-float32", "model:unknown:fp8-float32"),
    ],
)
def test_measurement_attribution_understands_only_known_cache_suffixes(cache_key, owner):
    point = CalibrationPoint.model_construct(model_key=cache_key, run_id="run")
    assert measured_models(point, {}) == frozenset({owner})
    assert measured_models(point, {"run": "calibrated"}) == frozenset({owner, "calibrated"})
