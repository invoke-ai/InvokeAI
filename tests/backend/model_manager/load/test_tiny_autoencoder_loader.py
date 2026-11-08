"""Loading a Tiny AutoEncoder: the decoder a file holds, from any published layout or a ComfyUI `.pth`."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.tiny_autoencoder import (
    TinyAutoencoder_Checkpoint_Config_Base,
    TinyAutoencoder_Checkpoint_FLUX_Config,
    TinyAutoencoder_Checkpoint_LTX2_Config,
    TinyAutoencoder_Checkpoint_QwenImage21_Config,
    TinyAutoencoder_Checkpoint_SD1_Config,
    TinyAutoencoder_Checkpoint_Wan_Config,
)
from invokeai.backend.model_manager.load.model_loaders.tiny_autoencoder import TinyAutoencoderLoader
from invokeai.backend.tiny_autoencoder.checkpoint import TAELTX2_3, TAEW2_2
from invokeai.backend.tiny_autoencoder.decoder import TinyAutoencoderDecoder
from invokeai.backend.tiny_autoencoder.taehv import TaehvDecoder
from tests.fixtures.tiny_autoencoders import TAEQI2_1, TAESD, TAESD3, diffusers_weights, native_weights, taehv_weights


def _load(
    path: Path,
    config_class: type[TinyAutoencoder_Checkpoint_Config_Base] = TinyAutoencoder_Checkpoint_SD1_Config,
    dtype: torch.dtype = torch.float32,
) -> TinyAutoencoderDecoder:
    loader = object.__new__(TinyAutoencoderLoader)
    loader._torch_dtype = dtype
    loader._logger = SimpleNamespace(info=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)
    config = config_class.model_construct(path=str(path), name=path.name)
    return loader._load_model(config)


def _assert_holds(decoder: TinyAutoencoderDecoder, native: dict[str, torch.Tensor]) -> None:
    loaded = decoder.state_dict()
    assert loaded.keys() == native.keys()
    for key, value in native.items():
        torch.testing.assert_close(loaded[key], value.to(loaded[key].dtype), rtol=0, atol=0)


def test_a_native_checkpoint_loads_into_an_inference_ready_decoder(tmp_path: Path) -> None:
    native = native_weights(TAESD)
    save_file(native, tmp_path / "taesd_decoder.safetensors")

    decoder = _load(tmp_path / "taesd_decoder.safetensors")

    _assert_holds(decoder, native)
    assert decoder.latent_channels == 4 and decoder.spatial_compression == 8
    assert not decoder.training
    assert not any(p.requires_grad for p in decoder.parameters())


def test_a_combined_checkpoint_loads_its_decoder_and_drops_the_encoder(tmp_path: Path) -> None:
    native = native_weights(TAESD3)
    save_file(diffusers_weights(native), tmp_path / "diffusion_pytorch_model.safetensors")

    decoder = _load(tmp_path / "diffusion_pytorch_model.safetensors", TinyAutoencoder_Checkpoint_FLUX_Config)

    _assert_holds(decoder, native)


def test_an_f16_rgba_checkpoint_published_in_half_precision_loads_at_the_loader_dtype(tmp_path: Path) -> None:
    native = native_weights(TAEQI2_1, dtype=torch.float16)
    save_file(diffusers_weights(native), tmp_path / "taeqi2_1.safetensors")

    decoder = _load(tmp_path / "taeqi2_1.safetensors", TinyAutoencoder_Checkpoint_QwenImage21_Config)

    _assert_holds(decoder, native)
    assert all(p.dtype == torch.float32 for p in decoder.parameters())
    assert decoder.spatial_compression == 16


def test_a_comfyui_pth_decoder_loads(tmp_path: Path) -> None:
    native = native_weights(TAESD)
    torch.save(native, tmp_path / "taesd_decoder.pth")

    _assert_holds(_load(tmp_path / "taesd_decoder.pth"), native)


def test_a_pth_decoder_wrapped_as_a_training_checkpoint_identifies_and_loads_the_weights_it_wraps(
    tmp_path: Path,
) -> None:
    """Training checkpoints nest their weights under `state_dict`. Identification unwraps them, so the loader must
    too: otherwise the file installs, then fails to load and every preview falls back to linear."""
    native = native_weights(TAESD)
    path = tmp_path / "taesd_decoder.pth"
    torch.save({"state_dict": native}, path)

    identified = ModelConfigFactory.from_model_on_disk(
        path, {"hash": "blake3:fake", "file_size": 1, "source": str(path), "source_type": "path"}
    ).config

    assert isinstance(identified, TinyAutoencoder_Checkpoint_SD1_Config)
    _assert_holds(_load(path, type(identified)), native)


def test_the_decoder_takes_the_loaders_dtype(tmp_path: Path) -> None:
    save_file(native_weights(TAESD), tmp_path / "taesd_decoder.safetensors")

    decoder = _load(tmp_path / "taesd_decoder.safetensors", dtype=torch.float16)

    assert all(p.dtype == torch.float16 for p in decoder.parameters())


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_a_checkpoint_that_is_not_exactly_the_decoder_is_refused(tmp_path: Path, change: str) -> None:
    sd = native_weights(TAESD)
    if change == "missing":
        del sd["18.conv.4.weight"]
    else:
        sd["18.conv.6.weight"] = torch.zeros(64, 64, 3, 3)
    save_file(sd, tmp_path / "taesd_decoder.safetensors")

    with pytest.raises(ValueError, match="not a supported Tiny AutoEncoder"):
        _load(tmp_path / "taesd_decoder.safetensors")


@pytest.mark.parametrize(
    ("layout", "config_class", "compression"),
    [(TAEW2_2, TinyAutoencoder_Checkpoint_Wan_Config, 16), (TAELTX2_3, TinyAutoencoder_Checkpoint_LTX2_Config, 32)],
    ids=["taew2_2", "taeltx"],
)
def test_a_taehv_checkpoint_loads_its_decoder_and_drops_the_encoder(
    tmp_path: Path, layout, config_class, compression: int
) -> None:
    published = taehv_weights(layout, dtype=torch.float16)
    save_file(published, tmp_path / "tae.safetensors")

    decoder = _load(tmp_path / "tae.safetensors", config_class)

    assert isinstance(decoder, TaehvDecoder)
    _assert_holds(decoder, {k: v for k, v in published.items() if k.startswith("decoder.")})
    assert decoder.spatial_compression == compression
    assert all(p.dtype == torch.float32 for p in decoder.parameters())
    assert not decoder.training


def test_a_taehv_checkpoint_with_a_tensor_the_decoder_lacks_is_refused(tmp_path: Path) -> None:
    sd = taehv_weights(TAEW2_2)
    sd["decoder.17.conv.6.weight"] = torch.zeros(64, 64, 3, 3)
    save_file(sd, tmp_path / "taew2_2.safetensors")

    with pytest.raises(ValueError, match="not a supported Tiny AutoEncoder"):
        _load(tmp_path / "taew2_2.safetensors", TinyAutoencoder_Checkpoint_Wan_Config)
