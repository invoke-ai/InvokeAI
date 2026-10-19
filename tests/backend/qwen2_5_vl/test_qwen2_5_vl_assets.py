"""Tests for the vendored Qwen2.5-VL architecture and preprocessor configs.

Single-file Qwen2.5-VL encoder checkpoints carry weights only. These two files are what the
loader and the conditioning node used to fetch from the hub at run time.
"""

from invokeai.backend.qwen2_5_vl.qwen2_5_vl_assets import (
    load_bundled_qwen2_5_vl_config_dict,
    load_bundled_qwen2_5_vl_preprocessor_config_dict,
)


def test_bundled_config_carries_the_released_architecture() -> None:
    """The constants that decide the module tree the single-file weights are folded into."""
    config = load_bundled_qwen2_5_vl_config_dict()

    assert config["hidden_size"] == 3584
    assert config["num_hidden_layers"] == 28
    assert config["rope_theta"] == 1000000.0
    assert config["vision_config"]["out_hidden_size"] == 3584


def test_bundled_preprocessor_config_carries_the_released_pixel_budget() -> None:
    """The previous last-resort rung built a bare ``Qwen2VLImageProcessor()``, whose class defaults
    omit these, so an offline run resized reference images differently with nothing in the log.
    """
    preprocessor = load_bundled_qwen2_5_vl_preprocessor_config_dict()

    assert preprocessor["min_pixels"] == 3136
    assert preprocessor["max_pixels"] == 12845056
    assert preprocessor["patch_size"] == 14
    assert preprocessor["merge_size"] == 2


def test_building_an_image_processor_from_the_bundle_keeps_the_release_pixel_budget() -> None:
    """What the conditioning node now builds, in the units that actually bite.

    transformers' own class default caps a reference image at 1,003,520 pixels, so the node's old
    last-resort rung — a bare ``Qwen2VLImageProcessor()`` when the hub was unreachable — downscaled
    every reference image by ~3.6x per side before the visual tower saw it, with nothing in the log
    to say so. Not asserted against a freshly constructed processor on purpose: ``from_dict``
    rewrites that class-level default in place, so such a comparison would pass or fail on test
    ordering rather than on this bundle.
    """
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    processor = Qwen2VLImageProcessor.from_dict(load_bundled_qwen2_5_vl_preprocessor_config_dict())

    assert processor.size.longest_edge == 12845056
    assert processor.size.shortest_edge == 3136
    assert processor.patch_size == 14
    assert processor.merge_size == 2


def test_config_dicts_are_not_shared_between_callers() -> None:
    """The loader sets ``torch_dtype`` on what it builds from this dict."""
    load_bundled_qwen2_5_vl_config_dict()["hidden_size"] = 1

    assert load_bundled_qwen2_5_vl_config_dict()["hidden_size"] == 3584
