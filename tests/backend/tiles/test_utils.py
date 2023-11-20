import numpy as np
import pytest

from invokeai.backend.tiles.utils import TBLR, paste


def test_paste_no_mask_success():
    """Test successful paste with mask=None."""
    dst_image = np.zeros((5, 5, 3), dtype=np.uint8)

    # Create src_image with a pattern that can be used to validate that it was pasted correctly.
    src_image = np.zeros((3, 3, 3), dtype=np.uint8)
    src_image[0, :, 0] = 1  # Row of 1s in channel 0.
    src_image[:, 0, 1] = 2  # Column of 2s in channel 1.

    # Paste in bottom-center of dst_image.
    box = TBLR(top=2, bottom=5, left=1, right=4)

    # Construct expected output image.
    expected_output = np.zeros((5, 5, 3), dtype=np.uint8)
    expected_output[2, 1:4, 0] = 1
    expected_output[2:5, 1, 1] = 2

    paste(dst_image=dst_image, src_image=src_image, box=box)

    np.testing.assert_array_equal(dst_image, expected_output, strict=True)


def test_paste_with_mask_success():
    """Test successful paste with a mask."""
    dst_image = np.zeros((5, 5, 3), dtype=np.uint8)

    # Create src_image with a pattern that can be used to validate that it was pasted correctly.
    src_image = np.zeros((3, 3, 3), dtype=np.uint8)
    src_image[0, :, 0] = 64  # Row of 64s in channel 0.
    src_image[:, 0, 1] = 128  # Column of 128s in channel 1.

    # Paste in bottom-center of dst_image.
    box = TBLR(top=2, bottom=5, left=1, right=4)

    # Create a mask that blends the top-left corner of 'src_image' at 50%, and ignores the rest of src_image.
    mask = np.zeros((3, 3))
    mask[0, 0] = 0.5

    # Construct expected output image.
    expected_output = np.zeros((5, 5, 3), dtype=np.uint8)
    expected_output[2, 1, 0] = 32
    expected_output[2, 1, 1] = 64

    paste(dst_image=dst_image, src_image=src_image, box=box, mask=mask)

    np.testing.assert_array_equal(dst_image, expected_output, strict=True)


@pytest.mark.parametrize("use_mask", [True, False])
def test_paste_box_overflows_dst_image(use_mask: bool):
    """Test that an exception is raised if 'box' overflows the 'dst_image'."""
    dst_image = np.zeros((5, 5, 3), dtype=np.uint8)
    src_image = np.zeros((3, 3, 3), dtype=np.uint8)
    mask = None
    if use_mask:
        mask = np.zeros((3, 3))

    # Construct box that overflows bottom of dst_image.
    top = 3
    left = 0
    box = TBLR(top=top, bottom=top + src_image.shape[0], left=left, right=left + src_image.shape[1])

    with pytest.raises(ValueError):
        paste(dst_image=dst_image, src_image=src_image, box=box, mask=mask)


@pytest.mark.parametrize("use_mask", [True, False])
def test_paste_src_image_does_not_match_box(use_mask: bool):
    """Test that an exception is raised if the 'src_image' shape does not match the 'box' dimensions."""
    dst_image = np.zeros((5, 5, 3), dtype=np.uint8)
    src_image = np.zeros((3, 3, 3), dtype=np.uint8)
    mask = None
    if use_mask:
        mask = np.zeros((3, 3))

    # Construct box that is smaller than src_image.
    box = TBLR(top=0, bottom=src_image.shape[0] - 1, left=0, right=src_image.shape[1])

    with pytest.raises(ValueError):
        paste(dst_image=dst_image, src_image=src_image, box=box, mask=mask)


def test_paste_mask_does_not_match_src_image():
    """Test that an exception is raised if the 'mask' shape is different than the 'src_image' shape."""
    dst_image = np.zeros((5, 5, 3), dtype=np.uint8)
    src_image = np.zeros((3, 3, 3), dtype=np.uint8)

    # Construct mask that is smaller than the src_image.
    mask = np.zeros((src_image.shape[0] - 1, src_image.shape[1]))

    # Construct box that matches src_image shape.
    box = TBLR(top=0, bottom=src_image.shape[0], left=0, right=src_image.shape[1])

    with pytest.raises(ValueError):
        paste(dst_image=dst_image, src_image=src_image, box=box, mask=mask)
