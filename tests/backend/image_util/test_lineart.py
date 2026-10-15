import math

import pytest
import torch
from PIL import Image

from invokeai.backend.image_util.lineart import LineartEdgeDetector


class _StrideRoundedLineartModel(torch.nn.Module):
    """Models the lineart generator's two stride-2 down/up-sampling stages."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        output_height = math.ceil(height / 4) * 4
        output_width = math.ceil(width / 4) * 4
        return torch.zeros((1, 1, output_height, output_width), dtype=image.dtype, device=image.device)


@pytest.mark.parametrize("size", [(64, 64), (65, 67), (66, 68), (67, 69)])
def test_lineart_edge_detector_preserves_input_dimensions(size: tuple[int, int]) -> None:
    source = Image.new("RGB", size, "white")
    detector = LineartEdgeDetector(_StrideRoundedLineartModel())  # type: ignore[arg-type]

    result = detector.run(source)

    assert result.size == source.size
