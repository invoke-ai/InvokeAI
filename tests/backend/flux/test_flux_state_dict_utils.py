from pathlib import Path

import pytest

from invokeai.backend.flux.flux_state_dict_utils import get_flux_in_channels_from_state_dict
from invokeai.backend.model_manager.config import ModelOnDisk

test_cases = [
    # Unquantized
    ("FLUX Dev.safetensors", 64),
    ("FLUX Schnell.safetensors", 64),
    ("FLUX Fill.safetensors", 384),
    # BNB-NF4 quantized
    ("FLUX Dev (Quantized).safetensors", 1),  # BNB-NF4
    ("FLUX Schnell (Quantized).safetensors", 1),  # BNB-NF4
    # GGUF quantized FLUX Fill
    ("flux1-fill-dev-Q8_0.gguf", 384),
    # Fine-tune w/ "model.diffusion_model.img_in.weight" instead of "img_in.weight"
    ("midjourneyReplica_flux1Dev.safetensors", 64),
    # Not a FLUX model, testing fallback case
    ("Noodles Style.safetensors", None),
]


@pytest.mark.parametrize("model_file_name,expected_in_channels", test_cases)
def test_get_flux_in_channels_from_state_dict(model_file_name: str, expected_in_channels: int, override_model_loading):
    model_path = Path(f"tests/test_model_probe/stripped_models/{model_file_name}")

    mod = ModelOnDisk(model_path)

    state_dict = mod.load_state_dict()

    in_channels = get_flux_in_channels_from_state_dict(state_dict)

    assert in_channels == expected_in_channels
