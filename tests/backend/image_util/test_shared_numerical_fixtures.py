import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from invokeai.backend.image_util.color import prepare_lightness
from invokeai.backend.image_util.guided_filter import guided_filter
from invokeai.backend.image_util.multiscale import edge_aware_band

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = (
    REPO_ROOT
    / "invokeai/frontend/webv2/src/features/generation/core/__fixtures__/shared_image_manipulation_numerics.json"
)


def test_shared_numerical_fixtures_match_python_cores() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    guided = fixtures["guided_filter"]
    guided_result = guided_filter(
        np.asarray(guided["guide"], dtype=np.float32),
        np.asarray(guided["signal"], dtype=np.float32),
        radius=guided["radius"],
        epsilon=guided["epsilon"],
    )
    np.testing.assert_allclose(guided_result, guided["expected"], atol=guided["atol"], rtol=0)

    multiscale = fixtures["multiscale_band"]
    band_result = edge_aware_band(
        np.asarray(multiscale["lightness"], dtype=np.float32),
        inner_radius=multiscale["inner_radius"],
        outer_radius=multiscale["outer_radius"],
        epsilon=multiscale["epsilon"],
    )
    np.testing.assert_allclose(band_result, multiscale["expected"], atol=multiscale["atol"], rtol=0)

    adapter = fixtures["color_adapter"]
    source = Image.fromarray(np.asarray(adapter["rgb"], dtype=np.uint8), mode="RGB")
    color_tensor, lightness, alpha = prepare_lightness(source)
    assert alpha is None
    np.testing.assert_allclose(
        color_tensor.numpy().transpose(1, 2, 0), adapter["expected_oklab"], atol=adapter["atol"], rtol=0
    )
    np.testing.assert_allclose(lightness, np.asarray(adapter["expected_oklab"])[:, :, 0], atol=adapter["atol"], rtol=0)
    assert color_tensor.dtype == torch.float32
