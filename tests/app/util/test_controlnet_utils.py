import numpy as np
import pytest
from PIL import Image

from invokeai.app.util.controlnet_utils import prepare_control_image
from invokeai.backend.image_util.util import nms


@pytest.mark.parametrize("num_channels", [1, 2, 3])
def test_prepare_control_image_num_channels(num_channels):
    """Test that the `num_channels` parameter is applied correctly in prepare_control_image(...)."""
    np_image = np.zeros((256, 256, 3), dtype=np.uint8)
    pil_image = Image.fromarray(np_image)

    torch_image = prepare_control_image(
        image=pil_image,
        width=256,
        height=256,
        num_channels=num_channels,
        device="cpu",
        do_classifier_free_guidance=False,
    )

    assert torch_image.shape == (1, num_channels, 256, 256)


@pytest.mark.parametrize("num_channels", [0, 4])
def test_prepare_control_image_num_channels_too_large(num_channels):
    """Test that an exception is raised in prepare_control_image(...) if the `num_channels` parameter is out of the
    supported range.
    """
    np_image = np.zeros((256, 256, 3), dtype=np.uint8)
    pil_image = Image.fromarray(np_image)

    with pytest.raises(ValueError):
        _ = prepare_control_image(
            image=pil_image,
            width=256,
            height=256,
            num_channels=num_channels,
            device="cpu",
            do_classifier_free_guidance=False,
        )


@pytest.mark.parametrize("threshold,sigma", [(None, 1.0), (1, None)])
def test_nms_invalid_options(threshold: None | int, sigma: None | float):
    """Test that an exception is raised in nms(...) if only one of the `threshold` or `sigma` parameters are provided."""
    with pytest.raises(ValueError):
        nms(np.zeros((256, 256, 3), dtype=np.uint8), threshold, sigma)
