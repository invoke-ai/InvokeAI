"""Shared fixtures for the VAE invocation tests."""

import itertools
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.shared.database.startup import init_database
from invokeai.app.services.shared.invocation_context import ImagesInterface
from invokeai.backend.image_util.raster.descriptor import (
    AlphaMode,
    ColorEncoding,
    RasterChannels,
    RasterDescriptor,
    ValueRange,
)
from invokeai.backend.image_util.raster.raster import RasterImage
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate, WorkingMemoryTerm


def _build_flux_shaped_vae() -> AutoencoderKL:
    """A real `AutoencoderKL` with the FLUX.1 VAE's *shape* and almost none of its weight.

    8059 parameters against 84 million, built in ~6 ms, and it decodes an 8x8 latent in ~7 ms —
    but it is the genuine class, so the properties these tests are about are the real ones rather
    than a mock's:

    * four blocks, hence the same 8x spatial compression, which is what `diffusers_latent_tile`
      computes its geometry from;
    * `tile_overlap_factor`, `tile_sample_min_size`, `tile_latent_min_size` and `use_tiling` as
      instance attributes the tiling scope actually reads and restores;
    * `latent_channels`, `scaling_factor` and `shift_factor`, which is what `is_flux_family_vae`
      reads -- the class alone no longer says which latent space a VAE encodes into;
    * a working `tiled_decode`/`tiled_encode`, so a cell can assert the assembled shape rather
      than only that a flag was set.

    A `MagicMock(spec=AutoencoderKL)` cannot stand in for it: `tile_overlap_factor` is set in
    `__init__`, so it is not on the class and a spec'd mock does not have it at all — the tiling
    scope would raise `AttributeError` on a double that looked correct.
    """
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(4, 4, 4, 4),
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=2,
        sample_size=64,
        scaling_factor=0.3611,
        shift_factor=0.1159,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )


@pytest.fixture
def flux_shaped_vae() -> Callable[[], AutoencoderKL]:
    """Factory, not an instance: a test that needs two independent VAEs should not share one."""
    return _build_flux_shaped_vae


def _fixed_working_memory(reservation_bytes: int) -> WorkingMemoryEstimate:
    return WorkingMemoryEstimate(
        estimator="test.fixed",
        version=1,
        features={},
        terms=(WorkingMemoryTerm("fixed", reservation_bytes, included=True),),
        source="test",
    )


@pytest.fixture
def fixed_working_memory() -> Callable[[int], WorkingMemoryEstimate]:
    """An estimate that reserves exactly the given bytes, for a node whose estimator a test patches out."""
    return _fixed_working_memory


def _srgb_descriptor(channels: RasterChannels, **overrides) -> RasterDescriptor:
    has_alpha = channels in (RasterChannels.LA, RasterChannels.RGBA)
    fields = {
        "channels": channels,
        "color": ColorEncoding.SRGB,
        "value_range": ValueRange.UNIT,
        "alpha": AlphaMode.STRAIGHT if has_alpha else AlphaMode.NONE,
        "channel_count": len(channels.value),
    }
    fields.update(overrides)
    return RasterDescriptor(**fields)


@pytest.fixture
def srgb_descriptor() -> Callable[..., RasterDescriptor]:
    """Builds the descriptor of a precise sRGB unit-range raster with the given channels (fields overridable)."""
    return _srgb_descriptor


@pytest.fixture
def precise_world(tmp_path: Path, srgb_descriptor):
    """Real image service, records and PNG storage behind a context whose model, tensor and config parts are small."""
    config = InvokeAIAppConfig(use_memory_db=True, image_subfolder_strategy="flat")
    config._root = tmp_path
    storage = DiskImageFileStorage(tmp_path / "images")
    invoker = MagicMock()
    invoker.services.configuration.pil_compress_level = 6
    invoker.services.configuration.image_subfolder_strategy = "flat"
    invoker.services.configuration.multiuser = False
    invoker.services.configuration.precise_image_storage = True
    storage.start(invoker)
    db = init_database(config=config, logger=InvokeAILogger.get_logger(), image_files=storage)
    counter = itertools.count()
    invoker.services.names.create_image_name.side_effect = lambda: f"image-{next(counter)}.png"
    invoker.services.database = db
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

    saved_tensors: dict[str, torch.Tensor] = {}

    def save_tensor(tensor: torch.Tensor) -> str:
        name = f"tensor-{len(saved_tensors)}"
        saved_tensors[name] = tensor
        return name

    vae_holder: dict[str, MagicMock] = {}
    context = SimpleNamespace(
        images=images,
        tensors=SimpleNamespace(load=lambda name: saved_tensors[name].clone(), save=save_tensor),
        models=SimpleNamespace(load=lambda _field: vae_holder["info"]),
        util=MagicMock(),
        config=SimpleNamespace(get=lambda: SimpleNamespace(force_tiled_decode=False)),
    )

    def store_png(array: np.ndarray, mode: str) -> str:
        image = Image.fromarray(array[:, :, 0] if array.shape[2] == 1 else array, mode)
        return service.create(image, ResourceOrigin.EXTERNAL, ImageCategory.GENERAL).image_name

    return SimpleNamespace(
        descriptor=srgb_descriptor,
        stored_bytes=lambda image_name: storage.get_path(image_name).read_bytes(),
        store_png=store_png,
        store_precise=lambda array, channels: images.save_raster(
            RasterImage.from_array(array, srgb_descriptor(channels))
        ).image_name,
        service=service,
        database=db,
        storage=storage,
        images=images,
        context=context,
        configuration=invoker.services.configuration,
        tensors=saved_tensors,
        vae_holder=vae_holder,
        use_vae=lambda info: vae_holder.__setitem__("info", info),
    )
