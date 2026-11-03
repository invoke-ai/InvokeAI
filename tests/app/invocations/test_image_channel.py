"""ImageChannelInvocation reads and writes rasters; its output must stay what PIL's `getchannel` produced."""

import itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from invokeai.app.invocations.image import ImageChannelInvocation
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.shared.database.startup import init_database
from invokeai.app.services.shared.invocation_context import ImagesInterface
from invokeai.backend.image_util.raster.convert import raster_to_pil
from invokeai.backend.image_util.raster.descriptor import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    ValueRange,
)
from invokeai.backend.image_util.raster.raster import RasterImage
from invokeai.backend.util.logging import InvokeAILogger


@pytest.fixture
def stack(tmp_path: Path):
    """Real image service, records and PNG storage; the node only talks to `context.images`."""
    config = InvokeAIAppConfig(use_memory_db=True, image_subfolder_strategy="flat")
    config._root = tmp_path
    storage = DiskImageFileStorage(tmp_path / "images")
    invoker = MagicMock()
    invoker.services.configuration.pil_compress_level = 6
    invoker.services.configuration.image_subfolder_strategy = "flat"
    invoker.services.configuration.multiuser = False
    storage.start(invoker)
    db = init_database(config=config, logger=InvokeAILogger.get_logger(), image_files=storage)
    counter = itertools.count()
    invoker.services.names.create_image_name.side_effect = lambda: f"image-{next(counter)}.png"
    invoker.services.image_records = ImageRecordStorage(db)
    invoker.services.image_files = storage
    invoker.services.board_image_records.get_board_for_image.return_value = None
    invoker.services.urls.get_image_url.return_value = "url"
    invoker.services.logger = MagicMock()
    service = ImageService()
    service.start(invoker)
    invoker.services.images = service
    data = SimpleNamespace(
        queue_item=SimpleNamespace(
            user_id=None,
            project_id=None,
            session_id=None,
            session=SimpleNamespace(graph=None),
            workflow=None,
            root_item_id=None,
            _workflow_json_loaded=False,
            _workflow_json_snapshot=None,
        ),
        invocation=SimpleNamespace(is_intermediate=False, id="node"),
    )
    images = ImagesInterface(invoker.services, data, MagicMock())
    return SimpleNamespace(service=service, storage=storage, images=images, context=SimpleNamespace(images=images))


def _noise(mode: str, seed: int) -> Image.Image:
    channels = len(mode)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(7, 5, channels), dtype=np.uint8)
    return Image.fromarray(array[:, :, 0] if channels == 1 else array, mode)


def _run(stack, mode: str, channel: str) -> tuple[str, str] | type[ValueError]:
    """Stored output file names of the node and of the PIL implementation it replaced, or the error type."""
    with _noise(mode, seed=3) as source:
        name = stack.service.create(source, ResourceOrigin.EXTERNAL, ImageCategory.GENERAL).image_name
    node = ImageChannelInvocation(id="channel", image={"image_name": name}, channel=channel)

    with stack.service.get_pil_image(name).copy() as pil:
        try:
            expected = pil.getchannel(channel)
        except ValueError:
            with pytest.raises(ValueError):
                node.invoke(stack.context)
            return ValueError
    old_name = stack.images.save(expected).image_name
    new_name = node.invoke(stack.context).image.image_name
    return old_name, new_name


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA"])
@pytest.mark.parametrize("channel", ["A", "R", "G", "B"])
def test_channel_output_is_byte_identical_to_pil_getchannel(stack, mode: str, channel: str) -> None:
    outcome = _run(stack, mode, channel)

    if outcome is ValueError:
        assert (mode, channel) in {
            ("L", "A"),
            ("L", "R"),
            ("L", "G"),
            ("L", "B"),
            ("LA", "R"),
            ("LA", "G"),
            ("LA", "B"),
            ("RGB", "A"),
        }
        return
    old_name, new_name = outcome
    assert stack.storage.get_path(new_name).read_bytes() == stack.storage.get_path(old_name).read_bytes()
    with Image.open(stack.storage.get_path(new_name)) as result:
        assert result.mode == "L"


def test_channel_output_does_not_inherit_the_sources_icc_profile() -> None:
    """The gallery writer drops ICC today, so the stored file cannot show it; the raster handed over must be clean."""
    descriptor = RasterDescriptor(
        RasterChannels.RGB, ColorEncoding.SRGB, ValueRange.UNIT, AlphaMode.NONE, 3, icc_profile=b"rgb-profile"
    )
    source = RasterImage.from_array(np.arange(12, dtype=np.uint8).reshape(2, 2, 3), descriptor)
    saved: list[RasterImage] = []
    images = SimpleNamespace(
        get_raster=lambda name: source,
        save_raster=lambda raster: saved.append(raster) or SimpleNamespace(image_name="out.png", width=2, height=2),
    )

    ImageChannelInvocation(id="channel", image={"image_name": "in.png"}, channel="G").invoke(
        SimpleNamespace(images=images)
    )

    assert saved[0].descriptor.icc_profile is None
    assert saved[0].view()[:, :, 0].tolist() == [[1, 4], [7, 10]]
    assert "icc_profile" not in raster_to_pil(saved[0]).info
