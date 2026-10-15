"""The Qwen-Image VAE loader picks its path from the checkpoint's key layout, not from its base.

Two layouts are in circulation for the same 16-channel autoencoder. Files exported from the
Qwen-Image repo carry diffusers keys (`decoder.conv_in.weight`); community redistributions carry the
original Wan-family layout (`decoder.middle.0.residual.0.gamma`) and need converting.

Only the first was handled, with `strict=True`, so a redistributed file failed with 194 missing keys
-- while the byte-identical checkpoint installed under `anima` loaded fine, because that path
converts. Whether a VAE worked came down to which base it happened to be probed as.

Both branches are now keyed on a marker the layout must carry, and both registrations of the
Wan-family file land in one loader, so the dtype and strictness policy cannot diverge between them.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan

from invokeai.backend.model_manager.configs.vae import (
    VAE_Checkpoint_Anima_Config,
    VAE_Checkpoint_Wan_Config,
    VAE_Diffusers_Wan_Config,
)
from invokeai.backend.model_manager.load.model_loaders.vae import VAELoader

# One key from each rule the Wan converter implements, so the assertion below is over a real
# conversion rather than a renamed copy of its input.
_ORIGINAL_LAYOUT = {
    "conv1.weight": "quant_conv.weight",
    "conv2.weight": "post_quant_conv.weight",
    "encoder.conv1.weight": "encoder.conv_in.weight",
    "encoder.downsamples.0.residual.0.gamma": "encoder.down_blocks.0.norm1.gamma",
    "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
    "decoder.conv1.weight": "decoder.conv_in.weight",
    "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
    "decoder.upsamples.0.residual.0.gamma": "decoder.up_blocks.0.resnets.0.norm1.gamma",
    "decoder.head.0.gamma": "decoder.norm_out.gamma",
}


# The 16-channel Wan VAE's structure (8x spatial, unpatchified) at the smallest width that builds.
_TINY_WAN_KWARGS = {
    "base_dim": 2,
    "dim_mult": [1, 1, 1, 1],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
}


def _loader(dtype: torch.dtype = torch.float16) -> VAELoader:
    """float16 by default: what `precision: auto` resolves to on CUDA, and what the Wan VAE is unstable in."""
    loader = VAELoader.__new__(VAELoader)
    loader._torch_dtype = dtype  # type: ignore[attr-defined]
    loader._ram_cache = MagicMock()  # type: ignore[attr-defined]
    loader._logger = MagicMock()  # type: ignore[attr-defined]
    return loader


def _config(path: str = "vae.safetensors") -> MagicMock:
    config = MagicMock()
    config.path = path
    return config


def _original_layout_state_dict() -> dict[str, torch.Tensor]:
    return {key: torch.zeros(2, dtype=torch.float32) for key in _ORIGINAL_LAYOUT}


def _load_original_layout(loader: VAELoader) -> tuple[dict[str, torch.Tensor], MagicMock, object]:
    """Load the original layout from a (mocked) safetensors file; returns what reached the module."""
    state_dict = _original_layout_state_dict()

    with (
        patch(
            "invokeai.backend.model_manager.load.model_loaders.vae._checkpoint_keys",
            return_value=set(state_dict),
        ),
        patch("safetensors.torch.load_file", return_value=state_dict),
        patch("accelerate.init_empty_weights"),
        patch("diffusers.models.autoencoders.AutoencoderKLWan") as wan,
        patch("invokeai.backend.wan.rocm_causal_conv3d.patch_wan_causal_conv3d_for_rocm"),
    ):
        wan.return_value.load_state_dict.return_value = ([], [])
        result = loader._load_qwen_image_vae(_config())

    (loaded,), _ = wan.return_value.load_state_dict.call_args
    return loaded, wan, result


def test_the_original_layout_is_converted_and_float16_is_raised_to_bfloat16() -> None:
    """`from_single_file` would fetch the Wan config over HTTP at load time, and it would honour a
    float16 `self._torch_dtype` -- the dtype `_load_wan_vae` and `_load_wan_vae_diffusers` both refuse
    for this exact autoencoder."""
    loader = _loader(torch.float16)
    loaded, wan, result = _load_original_layout(loader)

    wan.from_single_file.assert_not_called()
    wan.assert_called_once_with(z_dim=16)
    assert wan.return_value.load_state_dict.call_args.kwargs == {"strict": False, "assign": True}
    assert set(loaded) == set(_ORIGINAL_LAYOUT.values())
    assert {tensor.dtype for tensor in loaded.values()} == {torch.bfloat16}

    loader._ram_cache.make_room.assert_called_once_with(  # type: ignore[attr-defined]
        sum(tensor.nelement() * tensor.element_size() for tensor in loaded.values())
    )
    assert result is wan.return_value


def test_an_explicit_float32_request_is_kept() -> None:
    """Only float16 is unstable on this autoencoder. Casting a float32 request down to bfloat16 rounds
    the weights for as long as the model stays cached, which is the precision the user opted out of."""
    loaded, _, _ = _load_original_layout(_loader(torch.float32))

    assert {tensor.dtype for tensor in loaded.values()} == {torch.float32}


@pytest.mark.parametrize("registration", ["anima", "qwen-image"])
def test_a_pickled_checkpoint_loads_like_a_safetensors_one(tmp_path: Path, registration: str) -> None:
    """Identification reads `.pt` as well as safetensors, so an Anima VAE saved that way installed
    cleanly and then failed at load time with a `SafetensorError`. Real file, real readers."""
    path = tmp_path / "vae.pt"
    torch.save({"state_dict": _original_layout_state_dict()}, path)
    loader = _loader(torch.float32)

    with (
        patch("accelerate.init_empty_weights"),
        patch("diffusers.models.autoencoders.AutoencoderKLWan") as wan,
        patch("invokeai.backend.wan.rocm_causal_conv3d.patch_wan_causal_conv3d_for_rocm"),
    ):
        wan.return_value.load_state_dict.return_value = ([], [])
        if registration == "anima":
            loader._load_model(VAE_Checkpoint_Anima_Config.model_construct(path=str(path)))
        else:
            loader._load_qwen_image_vae(_config(str(path)))

    (loaded,), _ = wan.return_value.load_state_dict.call_args
    assert set(loaded) == set(_ORIGINAL_LAYOUT.values())
    assert {tensor.dtype for tensor in loaded.values()} == {torch.float32}


def test_a_wan_registered_checkpoint_follows_the_same_format_and_precision_policy(tmp_path: Path) -> None:
    """Anima also accepts the 16-channel VAE installed under `wan`, and that registration has its own
    loader. It read safetensors only and forced bfloat16, so one file behaved differently depending
    on the base it happened to be probed as -- `Wan2.1_VAE.pth` installed, then failed to load."""
    torch.manual_seed(0)
    state_dict = AutoencoderKLWan(z_dim=16, **_TINY_WAN_KWARGS).state_dict()
    path = tmp_path / "Wan2.1_VAE.pth"
    torch.save(state_dict, path)

    # A real module and the loader's own strict `load_state_dict`; only the width is shrunk. A
    # checkpoint whose keys or shapes do not fit the model the loader builds fails here as it would
    # for a user.
    with (
        patch(
            "invokeai.backend.model_manager.load.model_loaders.vae._wan_vae_init_kwargs_for",
            side_effect=lambda latent_channels: {**_TINY_WAN_KWARGS, "z_dim": latent_channels},
        ),
        patch("invokeai.backend.wan.rocm_causal_conv3d.patch_wan_causal_conv3d_for_rocm"),
    ):
        model = _loader(torch.float32)._load_model(
            VAE_Checkpoint_Wan_Config.model_construct(path=str(path), latent_channels=16)
        )

    loaded = model.state_dict()
    assert loaded.keys() == state_dict.keys()
    assert all(torch.equal(loaded[key], state_dict[key]) for key in state_dict)
    assert {tensor.dtype for tensor in loaded.values() if tensor.is_floating_point()} == {torch.float32}


@pytest.mark.parametrize(("requested", "expected"), [(torch.float16, torch.bfloat16), (torch.float32, torch.float32)])
def test_a_wan_diffusers_folder_follows_the_same_precision_policy(
    requested: torch.dtype, expected: torch.dtype
) -> None:
    with (
        patch("diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan") as wan,
        patch("invokeai.backend.wan.rocm_causal_conv3d.patch_wan_causal_conv3d_for_rocm"),
    ):
        _loader(requested)._load_model(VAE_Diffusers_Wan_Config.model_construct(path="wan-vae"))

    assert wan.from_pretrained.call_args.kwargs["torch_dtype"] == expected


def test_a_conversion_that_leaves_a_tensor_unset_is_an_error_not_a_warning() -> None:
    """`from_single_file` loads with `strict=False` and reports only *unexpected* keys, so a tensor
    the conversion never produced stays on the meta device and surfaces at the first decode --
    after the model has loaded and generation has begun."""
    state_dict = _original_layout_state_dict()

    with (
        patch(
            "invokeai.backend.model_manager.load.model_loaders.vae._checkpoint_keys",
            return_value=set(state_dict),
        ),
        patch("safetensors.torch.load_file", return_value=state_dict),
        patch("accelerate.init_empty_weights"),
        patch("diffusers.models.autoencoders.AutoencoderKLWan") as wan,
        patch("invokeai.backend.wan.rocm_causal_conv3d.patch_wan_causal_conv3d_for_rocm"),
    ):
        wan.return_value.load_state_dict.return_value = (["decoder.conv_out.weight"], [])

        with pytest.raises(ValueError, match="does not convert to a complete Wan 2.1 VAE"):
            _loader()._load_qwen_image_vae(_config())


def test_the_anima_registration_takes_the_same_path() -> None:
    """The same checkpoint, registered for a different family. It used to be loaded by a separate
    branch, so `anima` and `qwen-image` disagreed about the dtype of byte-identical weights."""
    loader = _loader()
    with patch.object(VAELoader, "_load_wan_family_vae") as load_wan_family:
        result = loader._load_model(VAE_Checkpoint_Anima_Config.model_construct(path="anima-vae.safetensors"))

    load_wan_family.assert_called_once_with("anima-vae.safetensors")
    assert result is load_wan_family.return_value


def test_the_diffusers_layout_still_loads_directly() -> None:
    """`AutoencoderKLQwenImage` registers no single-file conversion, so this path stays."""
    state_dict = {"decoder.conv_in.weight": torch.zeros(1)}

    with (
        patch(
            "invokeai.backend.model_manager.load.model_loaders.vae._checkpoint_keys",
            return_value=set(state_dict),
        ),
        patch("safetensors.torch.load_file", return_value=state_dict),
        patch("accelerate.init_empty_weights"),
        patch("diffusers.models.autoencoders.autoencoder_kl_qwenimage.AutoencoderKLQwenImage") as qwen,
        patch("diffusers.models.autoencoders.AutoencoderKLWan") as wan,
    ):
        result = _loader()._load_qwen_image_vae(_config())

    wan.assert_not_called()
    (loaded,), _ = qwen.return_value.load_state_dict.call_args
    # The same autoencoder as the converted layout, so float16 is raised the same way.
    assert {tensor.dtype for tensor in loaded.values()} == {torch.bfloat16}
    assert result is qwen.return_value


def test_a_pickled_checkpoint_in_the_diffusers_layout_loads_too(tmp_path: Path) -> None:
    """The layout probe and the load are separate reads; both have to understand a `.pt` file."""
    path = tmp_path / "qwen-image-vae.pt"
    torch.save({"decoder.conv_in.weight": torch.zeros(1)}, path)

    with (
        patch("accelerate.init_empty_weights"),
        patch("diffusers.models.autoencoders.autoencoder_kl_qwenimage.AutoencoderKLQwenImage") as qwen,
    ):
        _loader(torch.float32)._load_qwen_image_vae(_config(str(path)))

    (loaded,), _ = qwen.return_value.load_state_dict.call_args
    assert set(loaded) == {"decoder.conv_in.weight"}


def test_a_checkpoint_in_neither_layout_is_rejected() -> None:
    """The discriminator used to be a negation, so this file was assumed to be the Wan layout.

    `from_single_file` checks for keys it recognises, never for keys that are missing: an
    unrecognised file falls through to `model_type = "v1"`, fetches SD 1.5's `vae/config.json`, and
    returns an `AutoencoderKLWan` whose parameters are all still on the meta device -- a warning,
    not an error, and a failure that surfaces at the first decode instead of at install time.
    """
    state_dict = {"first_stage_model.decoder.conv_in.weight": torch.zeros(1)}

    with (
        patch(
            "invokeai.backend.model_manager.load.model_loaders.vae._checkpoint_keys",
            return_value=set(state_dict),
        ),
        patch("safetensors.torch.load_file", return_value=state_dict) as load_file,
        patch("diffusers.models.autoencoders.AutoencoderKLWan") as wan,
    ):
        with pytest.raises(ValueError, match="not a Qwen-Image VAE in either known layout"):
            _loader()._load_qwen_image_vae(_config())

    wan.assert_not_called()
    wan.from_single_file.assert_not_called()
    load_file.assert_not_called()
