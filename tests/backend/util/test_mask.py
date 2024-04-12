import pytest
import torch

from invokeai.backend.util.mask import to_standard_float_mask


def test_to_standard_float_mask_wrong_ndim():
    with pytest.raises(ValueError):
        to_standard_float_mask(mask=torch.zeros((1, 1, 5, 10)), out_dtype=torch.float32)


def test_to_standard_float_mask_wrong_shape():
    with pytest.raises(ValueError):
        to_standard_float_mask(mask=torch.zeros((2, 5, 10)), out_dtype=torch.float32)


def check_mask_result(mask: torch.Tensor, expected_mask: torch.Tensor):
    """Helper function to check the result of `to_standard_float_mask()`."""
    assert mask.shape == expected_mask.shape
    assert mask.dtype == expected_mask.dtype
    assert torch.allclose(mask, expected_mask)


def test_to_standard_float_mask_ndim_2():
    """Test the case where the input mask has shape (h, w)."""
    mask = torch.zeros((3, 2), dtype=torch.float32)
    mask[0, 0] = 1.0
    mask[1, 1] = 1.0

    expected_mask = torch.zeros((1, 3, 2), dtype=torch.float32)
    expected_mask[0, 0, 0] = 1.0
    expected_mask[0, 1, 1] = 1.0

    new_mask = to_standard_float_mask(mask=mask, out_dtype=torch.float32)

    check_mask_result(mask=new_mask, expected_mask=expected_mask)


def test_to_standard_float_mask_ndim_3():
    """Test the case where the input mask has shape (1, h, w)."""
    mask = torch.zeros((1, 3, 2), dtype=torch.float32)
    mask[0, 0, 0] = 1.0
    mask[0, 1, 1] = 1.0

    expected_mask = torch.zeros((1, 3, 2), dtype=torch.float32)
    expected_mask[0, 0, 0] = 1.0
    expected_mask[0, 1, 1] = 1.0

    new_mask = to_standard_float_mask(mask=mask, out_dtype=torch.float32)

    check_mask_result(mask=new_mask, expected_mask=expected_mask)


@pytest.mark.parametrize(
    "out_dtype",
    [torch.float32, torch.float16],
)
def test_to_standard_float_mask_bool_to_float(out_dtype: torch.dtype):
    """Test the case where the input mask has dtype bool."""
    mask = torch.zeros((3, 2), dtype=torch.bool)
    mask[0, 0] = True
    mask[1, 1] = True

    expected_mask = torch.zeros((1, 3, 2), dtype=out_dtype)
    expected_mask[0, 0, 0] = 1.0
    expected_mask[0, 1, 1] = 1.0

    new_mask = to_standard_float_mask(mask=mask, out_dtype=out_dtype)

    check_mask_result(mask=new_mask, expected_mask=expected_mask)


@pytest.mark.parametrize(
    "out_dtype",
    [torch.float32, torch.float16],
)
def test_to_standard_float_mask_float_to_float(out_dtype: torch.dtype):
    """Test the case where the input mask has type float (but not all values are 0.0 or 1.0)."""
    mask = torch.zeros((3, 2), dtype=torch.float32)
    mask[0, 0] = 0.1  # Should be converted to 0.0
    mask[0, 1] = 0.9  # Should be converted to 1.0

    expected_mask = torch.zeros((1, 3, 2), dtype=out_dtype)
    expected_mask[0, 0, 1] = 1.0

    new_mask = to_standard_float_mask(mask=mask, out_dtype=out_dtype)

    check_mask_result(mask=new_mask, expected_mask=expected_mask)
