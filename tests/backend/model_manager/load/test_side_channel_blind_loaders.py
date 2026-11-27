"""Loaders that handle no quantization side channel refuse one rather than dropping it.

Ten call sites across six loader classes read a state dict and hand it to a model without ever
looking for a `weight_scale`. At the eight that load non-strictly a quantized checkpoint does not
fail -- it *loads*, with the scale dropped as an unexpected key. Most cast every tensor to the
compute dtype, turning the fp8 codes into ordinary floats off by `1/weight_scale`; the Z-Image
ControlNet assigns them with no cast at all and reports the orphan only at DEBUG.

The other two (ERNIE-Image and the Anima LLLite adapter) load strictly and already raise on the
orphaned key, so there the guard buys a message that names the cause rather than a list of
unexpected tensors -- and, for ERNIE, refuses before the cache has been asked for room.

Each refusal cell writes a real checkpoint carrying a real fp8 layer -- asserted, because an earlier
draft of this file overwrote its own codes with a dense tensor and tested nothing about fp8 at all.
The guard is key-only, so that draft still passed.

Each cell is paired with a dense one. Two of these loaders have no other test in the tree, so
without that half nothing would notice `is_scale_metadata_key` being widened until it started
refusing ordinary checkpoints -- and it has been widened before, twice.

Two of the ten are not in the table below: Ideogram 4's text encoder and VAE read a diffusers
*folder* rather than a single file, and the encoder is the one seam here that does support a
quantized layout -- Ideogram's own weight-only fp8, behind a private flag in `config.json`. Both are
covered in `test_ideogram4_diffusers_loader.py`, which owns that loader.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_Anima_Config
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_ErnieImage_Config
from invokeai.backend.model_manager.configs.vae import (
    VAE_Checkpoint_Flux2_Config,
    VAE_Checkpoint_QwenImage_Config,
    VAE_Checkpoint_SD3_Config,
    VAE_Checkpoint_Wan_Config,
)
from invokeai.backend.model_manager.load.model_loaders import anima, ernie_image, flux, vae
from invokeai.backend.model_manager.load.model_loaders.anima import AnimaControlNetLLLiteModel
from invokeai.backend.model_manager.load.model_loaders.ernie_image import ErnieImageCheckpointModel
from invokeai.backend.model_manager.load.model_loaders.flux import Flux2VAELoader
from invokeai.backend.model_manager.load.model_loaders.vae import VAELoader
from tests.fixtures.loader_seams import Seam, prepare
from tests.fixtures.quantized_payloads import quantize_scaled_fp8

#: The layout marker `_load_qwen_image_vae` probes for *before* it reads the file. It is supplied in
#: every checkpoint here so the probe is never what refuses -- the guard has to be.
QWEN_IMAGE_LAYOUT_MARKER = "decoder.conv_in.weight"

#: Deliberately not the layout marker: an earlier draft used one key for both and the marker
#: silently replaced the fp8 codes.
QUANTIZED_LAYER = "encoder.conv_out"


def _dense() -> dict[str, torch.Tensor]:
    return {QWEN_IMAGE_LAYOUT_MARKER: torch.ones(4, 4), f"{QUANTIZED_LAYER}.weight": torch.ones(4, 4)}


def _quantized() -> dict[str, torch.Tensor]:
    payload = quantize_scaled_fp8(torch.randn(4, 4))
    state_dict = _dense()
    state_dict[f"{QUANTIZED_LAYER}.weight"] = payload.codes
    state_dict[f"{QUANTIZED_LAYER}.weight_scale"] = payload.scale
    return state_dict


def _written(tmp_path: Path, state_dict: dict[str, torch.Tensor], name: str) -> Path:
    """A real file, because two of these loaders read through `_read_checkpoint` rather than a name
    a test can patch."""
    path = tmp_path / f"{name}.safetensors"
    save_file(state_dict, path)
    return path


SITES = [
    pytest.param(
        ErnieImageCheckpointModel,
        ernie_image,
        "_load_model",
        lambda path: Main_Checkpoint_ErnieImage_Config.model_construct(path=str(path), name="ernie"),
        # The loader takes its dtype from the device, so it never reads `self._torch_dtype`.
        False,
        id="ernie_image",
    ),
    pytest.param(
        VAELoader,
        vae,
        "_load_wan_vae",
        lambda path: VAE_Checkpoint_Wan_Config.model_construct(path=str(path), name="wan-vae"),
        True,
        id="wan_vae",
    ),
    pytest.param(
        VAELoader,
        vae,
        "_load_qwen_image_vae",
        lambda path: VAE_Checkpoint_QwenImage_Config.model_construct(path=str(path), name="qwen-vae"),
        True,
        id="qwen_image_vae",
    ),
    pytest.param(
        VAELoader,
        vae,
        "_load_sd3_vae",
        lambda path: VAE_Checkpoint_SD3_Config.model_construct(path=str(path), name="sd3-vae"),
        True,
        id="sd3_vae",
    ),
    pytest.param(
        # Reached by the Anima VAE registration and by the community qwen-image redistribution of
        # the same checkpoint, and it takes a bare path rather than a config.
        VAELoader,
        vae,
        "_load_wan_family_vae",
        str,
        True,
        id="wan_family_vae",
    ),
    pytest.param(
        Flux2VAELoader,
        flux,
        "_load_model",
        lambda path: VAE_Checkpoint_Flux2_Config.model_construct(path=str(path), name="flux2-vae"),
        True,
        id="flux2_vae",
    ),
    pytest.param(
        AnimaControlNetLLLiteModel,
        anima,
        "_load_model",
        lambda path: ControlNet_Checkpoint_Anima_Config.model_construct(path=str(path), name="anima-control"),
        True,
        id="anima_controlnet",
    ),
]


@pytest.mark.parametrize(("loader", "module", "entry", "argument_for", "sets_torch_dtype"), SITES)
def test_a_quantized_checkpoint_is_refused(
    monkeypatch, tmp_path, loader, module, entry, argument_for, sets_torch_dtype
) -> None:
    """The message has to say the checkpoint is quantized and that this loader cannot take it --
    otherwise the user is left with a model that generates noise and no reason why."""
    state_dict = _quantized()
    assert any(tensor.dtype is torch.float8_e4m3fn for tensor in state_dict.values()), state_dict
    checkpoint = _written(tmp_path, state_dict, "quantized")
    seam = Seam(loader=loader, module=module, entry=entry, patches_device=True, sets_torch_dtype=sets_torch_dtype)
    run = prepare(seam, monkeypatch)

    with pytest.raises(ValueError, match="quantization side channel"):
        run.load(argument_for(checkpoint))

    # Before the cache is asked for room. ERNIE reserves on the line after the guard, so moving the
    # guard below it would cost a multi-gigabyte reservation and cast for a load that cannot finish.
    assert run.reserved == []


@pytest.mark.parametrize(("loader", "module", "entry", "argument_for", "sets_torch_dtype"), SITES)
def test_a_dense_checkpoint_gets_past_the_guard(
    monkeypatch, tmp_path, loader, module, entry, argument_for, sets_torch_dtype
) -> None:
    """The other half. These checkpoints are far too small to build a model from, so the load fails
    either way -- what must not happen is that it fails *here*. Widening `is_scale_metadata_key` to
    a spelling that also matches a dense tensor name would refuse every ordinary checkpoint, and for
    two of these loaders there is no other test in the tree that would notice.
    """
    checkpoint = _written(tmp_path, _dense(), "dense")
    seam = Seam(loader=loader, module=module, entry=entry, patches_device=True, sets_torch_dtype=sets_torch_dtype)
    run = prepare(seam, monkeypatch)

    try:
        run.load(argument_for(checkpoint))
    except Exception as exc:  # noqa: BLE001 - the point is *which* failure, not whether one happens
        assert "quantization side channel" not in str(exc), f"the guard fired on a dense checkpoint: {exc}"
