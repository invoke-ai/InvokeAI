import pytest
import torch

from invokeai.backend.sd3.sd3_mmditx import Sd3MMDiTX
from invokeai.backend.sd3.sd3_state_dict_utils import infer_sd3_mmditx_params, is_sd3_checkpoint
from tests.backend.sd3.sd3_5_mmditx_state_dict import sd3_sd_shapes


@pytest.mark.parametrize(
    ["sd_shapes", "expected"],
    [
        (sd3_sd_shapes, True),
        ({}, False),
        ({"foo": [1]}, False),
    ],
)
def test_is_sd3_checkpoint(sd_shapes: dict[str, list[int]], expected: bool):
    # Build mock state dict from the provided shape dict.
    sd = {k: None for k in sd_shapes}
    assert is_sd3_checkpoint(sd) == expected


def test_infer_sd3_mmditx_params():
    # Build mock state dict on the meta device.
    with torch.device("meta"):
        sd = {k: torch.zeros(shape) for k, shape in sd3_sd_shapes.items()}

    # Filter the MMDiTX parameters from the state dict.
    sd = {k: v for k, v in sd.items() if k.startswith("model.diffusion_model.")}

    params = infer_sd3_mmditx_params(sd)

    # Construct model from params.
    with torch.device("meta"):
        model = Sd3MMDiTX(params=params)

    model_sd = model.state_dict()

    # Assert that the model state dict is compatible with the original state dict.
    sd_without_prefix = {k.split("model.diffusion_model.")[-1]: v for k, v in model_sd.items()}
    assert set(model_sd.keys()) == set(sd_without_prefix.keys())
    for k in model_sd:
        assert model_sd[k].shape == sd_without_prefix[k].shape
