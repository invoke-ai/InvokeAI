"""Loading an ERNIE-Image transformer from a single safetensors file.

The geometry is not in the file: Comfy-Org ships weights alone, so the loader carries the released
`transformer/config.json` as a constant. Diffusers' own defaults describe a *different*, smaller
model (3072 wide, 24 layers), so a constant that drifts from the release does not fall back to
something reasonable -- it fails on the first generation after a ~16GB download, or builds the
wrong network. Both the values and the kwarg contract are pinned here.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import accelerate
import pytest
import torch
from safetensors.torch import save_file

import invokeai.backend.model_manager.load.model_loaders.ernie_image as loader_module
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_ErnieImage_Config
from invokeai.backend.model_manager.load.model_loaders.ernie_image import (
    ERNIE_IMAGE_TRANSFORMER_CONFIG,
    ErnieImageCheckpointModel,
)
from invokeai.backend.model_manager.taxonomy import SubModelType

# `baidu/ERNIE-Image`'s released `transformer/config.json`, verbatim. Kept as a literal so this
# fails when the loader's constant drifts, rather than agreeing with whatever the loader says.
RELEASED_TRANSFORMER_CONFIG = {
    "eps": 1e-06,
    "ffn_hidden_size": 12288,
    "hidden_size": 4096,
    "in_channels": 128,
    "num_attention_heads": 32,
    "num_layers": 36,
    "out_channels": 128,
    "patch_size": 1,
    "qk_layernorm": True,
    "rope_axes_dim": (32, 48, 48),
    "rope_theta": 256,
    "text_in_dim": 3072,
}

# The same shape at toy width: 2 heads of 32 make hidden 64, and `rope_axes_dim` has to sum to the
# head dimension the way the released (32, 48, 48) sums to 4096/32.
TINY_TRANSFORMER_CONFIG = {
    "eps": 1e-06,
    "ffn_hidden_size": 128,
    "hidden_size": 64,
    "in_channels": 8,
    "num_attention_heads": 2,
    "num_layers": 1,
    "out_channels": 8,
    "patch_size": 1,
    "qk_layernorm": True,
    "rope_axes_dim": (8, 12, 12),
    "rope_theta": 256,
    "text_in_dim": 16,
}


def test_the_carried_geometry_is_the_released_one() -> None:
    """The file has no config of its own, so this constant *is* the architecture."""
    carried = dict(ERNIE_IMAGE_TRANSFORMER_CONFIG)
    carried["rope_axes_dim"] = tuple(carried["rope_axes_dim"])

    assert carried == RELEASED_TRANSFORMER_CONFIG


def test_the_released_geometry_still_builds_a_model() -> None:
    """Pins the kwarg contract against a diffusers bump. Under `init_empty_weights` the 8B model
    allocates nothing, so this stays cheap while covering every name the constant uses."""
    from diffusers import ErnieImageTransformer2DModel

    with accelerate.init_empty_weights():
        model = ErnieImageTransformer2DModel(**ERNIE_IMAGE_TRANSFORMER_CONFIG)

    assert model.config.num_layers == RELEASED_TRANSFORMER_CONFIG["num_layers"]
    assert model.config.hidden_size == RELEASED_TRANSFORMER_CONFIG["hidden_size"]
    assert model.config.text_in_dim == RELEASED_TRANSFORMER_CONFIG["text_in_dim"]


def _loader(monkeypatch: pytest.MonkeyPatch) -> tuple[ErnieImageCheckpointModel, MagicMock]:
    ram_cache = MagicMock()
    loader = object.__new__(ErnieImageCheckpointModel)
    loader._ram_cache = ram_cache
    loader._apply_fp8_layerwise_casting = lambda model, _config, _submodel: model
    monkeypatch.setattr(loader_module.TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(
        loader_module.TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _device: torch.float32)
    )
    return loader, ram_cache


def _tiny_checkpoint(path: Path) -> torch.nn.Module:
    from diffusers import ErnieImageTransformer2DModel

    torch.manual_seed(0)
    reference = ErnieImageTransformer2DModel(**TINY_TRANSFORMER_CONFIG)
    save_file({key: value.contiguous() for key, value in reference.state_dict().items()}, str(path))
    return reference


def test_a_single_file_loads_into_the_model_it_describes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The load path itself: reserve, cast, build empty, assign, and leave nothing on meta."""
    monkeypatch.setattr(loader_module, "ERNIE_IMAGE_TRANSFORMER_CONFIG", TINY_TRANSFORMER_CONFIG)
    checkpoint = tmp_path / "ernie-image.safetensors"
    reference = _tiny_checkpoint(checkpoint)
    loader, ram_cache = _loader(monkeypatch)
    config = Main_Checkpoint_ErnieImage_Config.model_construct(path=str(checkpoint), name="ernie-image")

    model = loader._load_model(config, SubModelType.Transformer)

    assert [name for name, param in model.named_parameters() if param.is_meta] == []
    expected = reference.state_dict()
    for name, param in model.named_parameters():
        assert torch.equal(param, expected[name]), name
    # The reservation is what keeps a 16GB file from evicting itself mid-load.
    ram_cache.make_room.assert_called_once()
    assert ram_cache.make_room.call_args.args[0] > 0


def test_the_checkpoint_refuses_submodels_it_does_not_contain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A single file holds the transformer alone. An old workflow wired to take a VAE out of the
    main model has to be told what to do instead, not handed a shape error."""
    loader, _ = _loader(monkeypatch)
    config = Main_Checkpoint_ErnieImage_Config.model_construct(
        path=str(tmp_path / "ernie-image.safetensors"), name="ernie-image"
    )

    with pytest.raises(ValueError, match="holds only the transformer"):
        loader._load_model(config, SubModelType.VAE)


def test_a_foreign_config_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    loader, _ = _loader(monkeypatch)

    with pytest.raises(ValueError, match="Main_Checkpoint_ErnieImage_Config"):
        loader._load_model(SimpleNamespace(path="whatever"), SubModelType.Transformer)  # type: ignore[arg-type]
