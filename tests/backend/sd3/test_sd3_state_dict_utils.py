import pytest

from invokeai.backend.sd3.sd3_state_dict_utils import is_sd3_checkpoint
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
