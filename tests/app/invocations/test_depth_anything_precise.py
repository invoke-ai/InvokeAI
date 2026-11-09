"""`depth_anything_depth_estimation` with precise image storage, on CPU with a tiny real Depth Anything model run
through transformers' own `DepthEstimationPipeline` (only the weights are random)."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor
from transformers.pipelines import DepthEstimationPipeline

from invokeai.app.invocations.depth_anything import DepthAnythingDepthEstimationInvocation
from invokeai.backend.image_util.depth_anything.depth_anything_pipeline import DepthAnythingPipeline
from invokeai.backend.image_util.raster.descriptor import RasterChannels


@pytest.fixture
def depth_model() -> DepthAnythingPipeline:
    torch.manual_seed(0)
    backbone = Dinov2Config(
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
        out_features=["stage1", "stage2", "stage3", "stage4"],
        reshape_hidden_states=False,
    )
    config = DepthAnythingConfig(
        backbone_config=backbone,
        reassemble_hidden_size=16,
        patch_size=14,
        neck_hidden_sizes=[4, 8, 12, 16],
        fusion_hidden_size=8,
        head_hidden_size=8,
        backbone=None,
    )
    processor = DPTImageProcessor(size={"height": 28, "width": 28}, keep_aspect_ratio=False, ensure_multiple_of=14)
    model = DepthAnythingForDepthEstimation(config).eval()
    return DepthAnythingPipeline(
        DepthEstimationPipeline(model=model, image_processor=processor, task="depth-estimation")
    )


def _run(precise_world, depth_model: DepthAnythingPipeline) -> tuple[str, Image.Image]:
    rng = np.random.default_rng(1)
    source = Image.fromarray(rng.integers(0, 256, (37, 53, 3), dtype=np.uint8))
    name = precise_world.store_png(np.asarray(source), "RGB")

    @contextmanager
    def loaded(_url, _loader):
        yield depth_model

    precise_world.context.models = SimpleNamespace(load_remote_model=loaded)
    precise_world.context.logger = MagicMock()
    node = DepthAnythingDepthEstimationInvocation.model_construct(
        id="depth", image=SimpleNamespace(image_name=name), model_size="small_v2"
    )
    return node.invoke(precise_world.context).image.image_name, source


def _png_of(precise_world, image: Image.Image) -> bytes:
    was = precise_world.configuration.precise_image_storage
    precise_world.configuration.precise_image_storage = False
    try:
        return precise_world.stored_bytes(precise_world.images.save(image).image_name)
    finally:
        precise_world.configuration.precise_image_storage = was


def test_precise_depth_keeps_the_normalized_float_depth_and_the_gallery_png_stays_legacy(
    precise_world, depth_model
) -> None:
    name, source = _run(precise_world, depth_model)

    reference = depth_model._pipeline(source)  # an untouched transformers call
    predicted = reference["predicted_depth"].float().numpy()
    expected = ((predicted - predicted.min()) / (predicted.max() - predicted.min())).astype(np.float32)
    raster = precise_world.images.get_raster(name)
    assert raster.dtype == np.float32
    assert raster.descriptor == precise_world.descriptor(RasterChannels.L)
    np.testing.assert_array_equal(raster.view()[:, :, 0], expected)
    assert len(np.unique(raster.view())) > 256  # not quantized
    assert precise_world.stored_bytes(name) == _png_of(precise_world, reference["depth"])


def test_depth_with_precise_storage_disabled_is_the_legacy_save_without_a_raster(
    precise_world, depth_model, monkeypatch
) -> None:
    precise_world.configuration.precise_image_storage = False
    calls = []
    monkeypatch.setattr(precise_world.images, "save_raster", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(
        DepthAnythingPipeline,
        "generate_depth_with_unit",
        lambda *a, **k: pytest.fail("the float path must not run when precise storage is off"),
    )

    name, source = _run(precise_world, depth_model)

    assert calls == []
    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.stored_bytes(name) == _png_of(precise_world, depth_model._pipeline(source)["depth"])


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:divide by zero encountered:RuntimeWarning")
def test_non_finite_depth_falls_back_to_the_legacy_png_and_warns(precise_world, depth_model) -> None:
    # A constant depth map normalizes as 0/0.
    with torch.no_grad():
        for parameter in depth_model._pipeline.model.parameters():
            parameter.zero_()

    name, source = _run(precise_world, depth_model)

    precise_world.context.logger.warning.assert_called_once()
    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.stored_bytes(name) == _png_of(precise_world, depth_model._pipeline(source)["depth"])


def test_a_depth_output_of_another_size_falls_back_to_the_legacy_png_and_warns(
    precise_world, depth_model, monkeypatch
) -> None:
    real = DepthAnythingPipeline.generate_depth_with_unit
    monkeypatch.setattr(
        DepthAnythingPipeline,
        "generate_depth_with_unit",
        lambda self, image: (lambda depth, unit: (depth, unit[:-1]))(*real(self, image)),
    )

    name, source = _run(precise_world, depth_model)

    precise_world.context.logger.warning.assert_called_once()
    assert precise_world.images.get_raster(name).dtype == np.uint8
    assert precise_world.stored_bytes(name) == _png_of(precise_world, depth_model._pipeline(source)["depth"])


def test_a_half_precision_depth_is_normalized_in_float32() -> None:
    """The seam is the pipeline's output: a half precision `predicted_depth` (a fp16 model) with many distinct values."""
    predicted = torch.linspace(0.0, 40.0, 37 * 53).reshape(37, 53).half()
    image = Image.new("L", (53, 37))

    class HalfPipeline:
        def __call__(self, _image):
            return {"predicted_depth": predicted, "depth": image}

    _, unit = DepthAnythingPipeline(HalfPipeline()).generate_depth_with_unit(image)

    half = predicted.numpy()
    in_half = (half - half.min()) / (half.max() - half.min())
    widened = predicted.float().numpy()
    np.testing.assert_array_equal(
        unit, ((widened - widened.min()) / (widened.max() - widened.min())).astype(np.float32)
    )
    assert unit.dtype == np.float32
    assert len(np.unique(unit)) > len(np.unique(in_half))
