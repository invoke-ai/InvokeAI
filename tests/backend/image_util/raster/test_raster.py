import copy as copy_module
import dataclasses
import pickle

import numpy as np
import pytest

from invokeai.backend.image_util.raster import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterConversionError,
    RasterDescriptor,
    RasterDescriptorError,
    RasterImage,
    ValueRange,
)


def make_descriptor(
    channels=RasterChannels.RGB,
    color=ColorEncoding.SRGB,
    value_range=ValueRange.UNIT,
    alpha=AlphaMode.NONE,
    channel_count=3,
    icc_profile=None,
) -> RasterDescriptor:
    return RasterDescriptor(channels, color, value_range, alpha, channel_count, icc_profile)


RGB = make_descriptor()
RGBA = make_descriptor(RasterChannels.RGBA, alpha=AlphaMode.STRAIGHT, channel_count=4)


def rgb_array(height=4, width=6, dtype=np.uint8) -> np.ndarray:
    return (np.arange(height * width * 3) % 251).astype(dtype).reshape(height, width, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channels": RasterChannels.RGB, "channel_count": 4},
        {"channels": RasterChannels.L, "channel_count": 3},
        {"channels": RasterChannels.RGB, "alpha": AlphaMode.STRAIGHT},
        {"channels": RasterChannels.RGBA, "channel_count": 4},
        {"channels": RasterChannels.LA, "channel_count": 2},
        {"channels": RasterChannels.DATA, "color": ColorEncoding.NONE, "alpha": AlphaMode.STRAIGHT, "channel_count": 2},
        {"channels": RasterChannels.DATA, "color": ColorEncoding.SRGB, "channel_count": 2},
        {"channels": RasterChannels.DATA, "color": ColorEncoding.NONE, "channel_count": 0},
        {"channels": RasterChannels.RGB, "color": ColorEncoding.NONE},
        {"channels": RasterChannels.DATA, "color": ColorEncoding.NONE, "channel_count": 2, "icc_profile": b"icc"},
    ],
)
def test_descriptor_rejects_inconsistent_combinations(kwargs):
    with pytest.raises(RasterDescriptorError):
        make_descriptor(**kwargs)


def test_descriptor_accepts_multichannel_data_and_is_immutable():
    data = make_descriptor(RasterChannels.DATA, ColorEncoding.NONE, ValueRange.UNBOUNDED, channel_count=5)
    assert data.channel_count == 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        data.channel_count = 2  # type: ignore[misc]


def test_geometry_comes_from_the_array():
    raster = RasterImage.from_array(rgb_array(4, 6, np.float16), make_descriptor(value_range=ValueRange.UNIT))
    assert (raster.height, raster.width, raster.channel_count, raster.dtype) == (4, 6, 3, np.float16)
    assert raster.nbytes == 4 * 6 * 3 * 2


@pytest.mark.parametrize(
    ("array", "descriptor", "error"),
    [
        (np.zeros((4, 6), np.uint8), RGB, RasterDescriptorError),
        (np.zeros((2, 4, 6, 3), np.uint8), RGB, RasterDescriptorError),
        (np.zeros((0, 6, 3), np.uint8), RGB, RasterDescriptorError),
        (np.zeros((4, 0, 3), np.uint8), RGB, RasterDescriptorError),
        (np.zeros((4, 6, 4), np.uint8), RGB, RasterDescriptorError),
        (np.zeros((4, 6, 3), np.int32), RGB, RasterConversionError),
        (np.zeros((4, 6, 3), np.float64), RGB, RasterConversionError),
        (np.zeros((4, 6, 3), ">u2"), RGB, RasterConversionError),
        (np.zeros((4, 6, 3), np.uint8), make_descriptor(value_range=ValueRange.UNBOUNDED), RasterDescriptorError),
        ([[[0, 0, 0]]], RGB, RasterDescriptorError),
    ],
)
def test_construction_rejects_invalid_arrays(array, descriptor, error):
    with pytest.raises(error):
        RasterImage.from_array(array, descriptor)
    with pytest.raises(error):
        RasterImage.borrow(array, descriptor)


def test_unbounded_float_keeps_nonfinite_and_out_of_range_values():
    descriptor = make_descriptor(value_range=ValueRange.UNBOUNDED)
    source = np.array([[[np.nan, np.inf, -np.inf], [1e30, -3.5, 0.0]]], np.float32)
    raster = RasterImage.from_array(source, descriptor)
    np.testing.assert_array_equal(raster.view(), source)


def test_from_array_is_independent_of_the_source():
    source = rgb_array()
    expected = source.copy()
    raster = RasterImage.from_array(source, RGB)
    source[:] = 0
    np.testing.assert_array_equal(raster.view(), expected)
    assert raster.is_owned


@pytest.mark.parametrize(
    "make_source",
    [
        lambda a: a[:, ::-1, :],  # negative stride
        lambda a: a[::2, ::2, :],  # non-contiguous
        lambda a: np.asfortranarray(a),
        lambda a: np.lib.stride_tricks.as_strided(a, a.shape, a.strides),
    ],
)
def test_from_array_copies_awkward_layouts_into_a_compact_array(make_source):
    base = rgb_array(8, 8)
    source = make_source(base)
    raster = RasterImage.from_array(source, RGB)
    np.testing.assert_array_equal(raster.view(), source)
    assert raster.view().flags.c_contiguous
    assert raster.retained_nbytes == raster.nbytes


def test_from_array_accepts_readonly_source():
    source = rgb_array()
    source.flags.writeable = False
    raster = RasterImage.from_array(source, RGB)
    np.testing.assert_array_equal(raster.view(), source)


def test_view_is_readonly_and_does_not_copy():
    raster = RasterImage.from_array(rgb_array(), RGB)
    view = raster.view()
    with pytest.raises(ValueError):
        view[0, 0, 0] = 1
    with pytest.raises(ValueError):
        view.flags.writeable = True
    assert np.shares_memory(view, raster.view())


def test_to_array_is_a_writable_independent_copy():
    raster = RasterImage.from_array(rgb_array(), RGB)
    expected = raster.view().copy()
    array = raster.to_array()
    assert array.flags.writeable and array.flags.c_contiguous
    assert not np.shares_memory(array, raster.view())
    array[:] = 7
    np.testing.assert_array_equal(raster.view(), expected)


def test_borrow_is_zero_copy_readonly_and_not_owned():
    source = rgb_array()
    raster = RasterImage.borrow(source, RGB)
    assert np.shares_memory(raster.view(), source)
    assert not raster.is_owned
    with pytest.raises(ValueError):
        raster.view()[0, 0, 0] = 1
    assert source.flags.writeable  # the caller's array keeps its own flags


def test_borrowed_crop_retains_the_parent_buffer_until_compacted():
    big = np.zeros((1000, 1000, 3), np.uint8)
    crop = RasterImage.borrow(big[10:20, 10:20], RGB)
    assert crop.nbytes == 10 * 10 * 3
    assert crop.retained_nbytes == big.nbytes

    big[10, 10] = (1, 2, 3)  # a borrow tracks its source
    assert tuple(crop.view()[0, 0]) == (1, 2, 3)

    compact = crop.owned()
    assert compact is not crop and compact.is_owned
    assert compact.retained_nbytes == compact.nbytes == 300
    big[10, 10] = 9
    assert tuple(compact.view()[0, 0]) == (1, 2, 3)


def test_owned_returns_self_for_compact_raster():
    raster = RasterImage.from_array(rgb_array(), RGB)
    assert raster.owned() is raster


def test_borrowing_a_noncontiguous_array_is_still_zero_copy():
    source = rgb_array(8, 8)[:, ::-1, :]
    raster = RasterImage.borrow(source, RGB)
    assert np.shares_memory(raster.view(), source)
    assert raster.owned().view().flags.c_contiguous


def test_frombuffer_crop_retains_the_whole_buffer():
    buffer = bytearray(500 * 500 * 3)
    crop = RasterImage.borrow(np.frombuffer(buffer, np.uint8).reshape(500, 500, 3)[5:15, 5:15], RGB)
    assert crop.nbytes == 300
    assert crop.retained_nbytes == len(buffer)


def test_torch_backed_slice_retains_the_whole_tensor_storage():
    torch = pytest.importorskip("torch")
    tensor = torch.zeros(100, 100, 3, dtype=torch.uint8)
    # The numpy array's base is a tensor that is itself a slice: only its storage knows the full size.
    crop = RasterImage.borrow(tensor[10:20].numpy()[:, 10:20], RGB)
    assert crop.nbytes == 10 * 10 * 3
    assert crop.retained_nbytes == tensor.untyped_storage().nbytes() == 100 * 100 * 3


def test_strided_parent_is_counted_conservatively():
    parent = np.zeros((100, 100, 3), np.uint8)
    strided = np.lib.stride_tricks.as_strided(parent, (100, 100, 3), parent.strides)
    crop = RasterImage.borrow(strided[:2, :2], RGB)
    assert crop.retained_nbytes >= parent.nbytes


def test_owned_copies_a_borrowed_contiguous_array():
    source = rgb_array()
    borrowed = RasterImage.borrow(source, RGB)
    compact = borrowed.owned()
    assert compact is not borrowed and compact.is_owned
    assert not np.shares_memory(compact.view(), source)
    np.testing.assert_array_equal(compact.view(), source)


def test_owned_rasters_are_c_contiguous_even_from_transposed_sources():
    source = np.ascontiguousarray(rgb_array(4, 6)).transpose(1, 0, 2)  # (6, 4, 3), not C-contiguous
    borrowed = RasterImage.borrow(source, RGB)
    assert not borrowed.is_owned
    compact = borrowed.owned()
    assert compact.is_owned and compact.view().flags.c_contiguous
    np.testing.assert_array_equal(compact.view(), source)


def test_rasters_cannot_be_constructed_directly():
    with pytest.raises(TypeError):
        RasterImage(rgb_array(), RGB, True)  # type: ignore[call-arg]


@pytest.mark.parametrize("copier", ["pickle", "deepcopy"])
def test_pickle_and_deepcopy_stay_readonly_and_equal(copier):
    original = RasterImage.borrow(rgb_array(), make_descriptor(icc_profile=b"icc"))
    copy = pickle.loads(pickle.dumps(original)) if copier == "pickle" else copy_module.deepcopy(original)
    assert copy.descriptor == original.descriptor and copy.is_owned
    np.testing.assert_array_equal(copy.view(), original.view())
    assert not np.shares_memory(copy.view(), original.view())
    with pytest.raises(ValueError):
        copy.view().flags.writeable = True


def test_descriptor_coerces_valid_strings_and_rejects_the_rest():
    coerced = RasterDescriptor("data", "none", "unbounded", "none", 5)
    assert coerced == make_descriptor(RasterChannels.DATA, ColorEncoding.NONE, ValueRange.UNBOUNDED, channel_count=5)
    for bad in [
        {"channels": "rgb"},
        {"channels": 3},
        {"color": "gamma"},
        {"value_range": None},
        {"alpha": "maybe"},
        {"channel_count": 3.0},
        {"icc_profile": "text"},
        {"icc_profile": bytearray(b"x")},
    ]:
        with pytest.raises(RasterDescriptorError):
            make_descriptor(**bad)
    with pytest.raises(RasterDescriptorError):
        RasterDescriptor("RGB", "none", "unit", "none", 3)
    with pytest.raises(RasterDescriptorError):  # True == 1 would otherwise satisfy L's channel count
        RasterDescriptor(RasterChannels.L, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, True)
