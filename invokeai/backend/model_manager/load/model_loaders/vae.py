"""Class for VAE model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

import accelerate
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.vae import (
    VAE_Checkpoint_Anima_Config,
    VAE_Checkpoint_Config_Base,
    VAE_Checkpoint_QwenImage_Config,
    VAE_Checkpoint_SD3_Config,
    VAE_Checkpoint_Wan_Config,
    VAE_Diffusers_FLUX_Config,
    VAE_Diffusers_Wan_Config,
)
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.quantization.fp8_scaled import reject_quantized_side_channel
from invokeai.backend.quantization.sdnq.detection import is_sdnq_folder
from invokeai.backend.quantization.sdnq.loaders import raise_on_incomplete_sdnq_load, sdnq_sd_loader
from invokeai.backend.sd3.vae import get_sd3_vae_diffusers_config
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras


def _is_sdnq_vae_folder(path: Path) -> bool:
    """Check if a VAE folder contains SDNQ-quantized weights.

    Shared detector: marker file first, then the weight/scale key pair unioned across shards, so a
    sharded or markerless export is recognized the same way identification recognizes it.
    """
    return is_sdnq_folder(path)


_QWEN_IMAGE_LAYOUT_MARKER = "decoder.conv_in.weight"
"""Present only in the diffusers export of the Qwen-Image VAE."""

_LDM_LAYOUT_MARKER = "encoder.down.0.block.0.norm1.weight"
"""Present in the original LDM layout of a 2-D `AutoencoderKL` (and in BFL's, which is the same)."""

_DIFFUSERS_LAYOUT_MARKER = "encoder.down_blocks.0.resnets.0.norm1.weight"
"""The same tensor in the diffusers layout."""

_WAN_LAYOUT_MARKER = "decoder.middle.0.residual.0.gamma"
"""Present only in the original Wan-family layout.

The same key diffusers' own `infer_diffusers_model_type` keys the Wan VAE off, so a file carrying
it is one `convert_wan_vae_to_diffusers` knows how to read.
"""


def _checkpoint_keys(path: str | Path) -> set[str]:
    """The tensor names in a single-file checkpoint, without reading its tensors.

    Layout is decided before anything is loaded, so the file is read once, by whichever branch
    actually needs the tensors. Safetensors answer from the header; a pickled checkpoint is unpickled
    onto the meta device, which skips its storages.
    """
    if Path(path).suffix != ".safetensors":
        return set(_unwrap_state_dict(torch.load(path, map_location="meta", weights_only=True)))

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return set(f.keys())


def _read_checkpoint(path: str | Path) -> dict[str, torch.Tensor]:
    """Every tensor in a single-file checkpoint, whichever format it was saved in.

    Identification reads pickled checkpoints (`.pt`, `.pth`, `.ckpt`, `.bin`) as well as safetensors,
    so a loader that reads only safetensors fails on a VAE that installed cleanly. `weights_only`
    refuses anything in the pickle but tensors and plain containers.
    """
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path)

    return _unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def _unwrap_state_dict(checkpoint: dict) -> dict:
    """Training checkpoints nest the weights under `state_dict`, and identification unwraps them too."""
    return checkpoint.get("state_dict", checkpoint)


def _wan_family_dtype(requested: torch.dtype) -> torch.dtype:
    """The precision a Wan-family VAE runs in: the configured one, except float16.

    float16 is unstable on this autoencoder and is what `precision: auto` resolves to on CUDA, so it is
    raised to bfloat16. A float32 request is kept -- casting it down would round the weights for the
    lifetime of the cached model.
    """
    return torch.bfloat16 if requested == torch.float16 else requested


# Architectural defaults for the Wan 2.2-VAE (TI2V-5B). Verbatim from the
# vae/config.json shipped with Wan-AI/Wan2.2-TI2V-5B-Diffusers — only the
# values that differ from diffusers' AutoencoderKLWan defaults are listed.
# latents_mean / latents_std are required because the model normalises latents
# against them at encode/decode time; the wrong arrays produce silent garbage.
_WAN_TI2V_5B_VAE_CONFIG: dict = {
    "base_dim": 160,
    "decoder_base_dim": 256,
    "z_dim": 48,
    "in_channels": 12,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_spatial": 16,
    "is_residual": True,
    "latents_mean": [
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.1570,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.1230,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.0520,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    ],
    "latents_std": [
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.4990,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.0600,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    ],
}


def _wan_vae_init_kwargs_for(latent_channels: int) -> dict:
    """Return the AutoencoderKLWan constructor kwargs for a given z_dim.

    z_dim=48 means TI2V-5B's Wan 2.2-VAE (different base dim, patchified IO,
    16x spatial). Anything else falls back to the A14B / Wan 2.1 defaults.
    """
    if latent_channels == 48:
        return dict(_WAN_TI2V_5B_VAE_CONFIG)
    return {"z_dim": latent_channels}


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.VAE, format=ModelFormat.Diffusers)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.VAE, format=ModelFormat.Checkpoint)
class VAELoader(GenericDiffusersLoader):
    """Class to load VAE models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, VAE_Checkpoint_Anima_Config):
            # `VAE_Checkpoint_Anima_Config` matches on the original Wan-family layout, which is what
            # `_load_wan_family_vae` reads -- the same checkpoint the community `qwen-image`
            # redistribution carries.
            return self._load_wan_family_vae(config.path)
        elif isinstance(config, VAE_Checkpoint_Wan_Config):
            return self._load_wan_vae(config)
        elif isinstance(config, VAE_Diffusers_Wan_Config):
            return self._load_wan_vae_diffusers(config)
        elif isinstance(config, VAE_Checkpoint_QwenImage_Config):
            return self._load_qwen_image_vae(config)
        elif isinstance(config, VAE_Checkpoint_SD3_Config):
            return self._load_sd3_vae(config)
        elif isinstance(config, VAE_Checkpoint_Config_Base):
            return AutoencoderKL.from_single_file(
                config.path,
                torch_dtype=self._torch_dtype,
            )

        model_path = Path(config.path)

        # Check if this is an SDNQ-quantized VAE folder
        if model_path.is_dir() and _is_sdnq_vae_folder(model_path):
            return self._load_sdnq_vae(model_path)

        if isinstance(config, VAE_Diffusers_FLUX_Config):
            # In the dtype every other FLUX.1 VAE path uses: the generic loader below would take float16,
            # which `precision: auto` picks on CUDA and MPS and which this autoencoder is broken in.
            return AutoencoderKL.from_pretrained(
                model_path, torch_dtype=self._torch_dtype_avoiding_float16(), local_files_only=True
            )

        # The FLUX, Z-Image and SD3 model loaders ask for a standalone VAE as `SubModelType.VAE`, the
        # way they ask a main model for its VAE. A standalone VAE folder *is* that submodel; the
        # generic loader would read the request as one for a submodel the folder does not have.
        if submodel_type is SubModelType.VAE:
            submodel_type = None
        return super()._load_model(config, submodel_type)

    def _load_sd3_vae(self, config: VAE_Checkpoint_SD3_Config) -> AnyModel:
        """Load a single-file SD3 VAE into an `AutoencoderKL` built with SD3's constants.

        `AutoencoderKL.from_single_file` cannot: with no config beside the weights, diffusers infers
        the model from the keys, finds no pipeline around a bare VAE and builds SD 1.5's 4-channel one.

        Both layouts are accepted, where `FluxVAELoader` refuses the diffusers one. For FLUX.1 the base
        may be nothing more than the default every unnamed 16-channel file gets; for SD3 it never is,
        because identification files a VAE under SD3 only when its name or an explicit override says
        so. The LDM layout is the one a VAE extracted from an SD3 single-file checkpoint comes in.
        """
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

        name = Path(config.path).name
        sd = _read_checkpoint(config.path)
        reject_quantized_side_channel(sd, f"SD3 VAE checkpoint {name}")

        with accelerate.init_empty_weights():
            model = AutoencoderKL(**get_sd3_vae_diffusers_config())

        if _LDM_LAYOUT_MARKER in sd:
            sd = convert_ldm_vae_checkpoint(sd, model.config)
        elif _DIFFUSERS_LAYOUT_MARKER not in sd:
            raise ValueError(f"{name} is in neither the diffusers nor the LDM layout of the SD3 autoencoder.")

        sd = {k: v.to(self._torch_dtype) if v.is_floating_point() else v for k, v in sd.items()}
        load_state_dict_ignoring_extras(model, sd, source="SD3 VAE checkpoint", assign=True)
        model.eval()
        return model

    def _load_wan_vae(self, config: VAE_Checkpoint_Wan_Config) -> AnyModel:
        """Load a Wan 2.2 VAE from a single-file checkpoint.

        Picks the correct ``AutoencoderKLWan`` config based on ``z_dim``. The Wan
        ecosystem ships two distinct VAE architectures:

        * ``z_dim=16`` — the Wan 2.1 / Wan 2.2 A14B VAE. Diffusers' defaults match
          this one (base_dim=96, 8x spatial, no patchify, 3 in/out channels).
        * ``z_dim=48`` — the Wan 2.2-VAE used by TI2V-5B. Larger (base_dim=160,
          decoder_base_dim=256), 16x spatial, patchify with patch_size=2 (so
          in/out channels are 12 = 3 RGB x 2x2 patch), residual blocks, and
          its own latents_mean / latents_std.

        Without overriding those params at construction time, the state dict
        from the TI2V-5B VAE checkpoint won't load (channel and shape mismatches
        throughout the encoder + decoder).

        Reads the same formats and follows the same precision policy as `_load_wan_family_vae`: a
        16-channel file registered under ``wan`` is the VAE Anima also accepts, and must not behave
        differently for the base it was probed as.
        """
        import accelerate
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

        from invokeai.backend.wan.rocm_causal_conv3d import patch_wan_causal_conv3d_for_rocm

        patch_wan_causal_conv3d_for_rocm()
        dtype = _wan_family_dtype(self._torch_dtype)
        sd = _read_checkpoint(config.path)
        reject_quantized_side_channel(sd, f"Wan VAE checkpoint {Path(config.path).name}")

        for k in list(sd.keys()):
            if sd[k].is_floating_point():
                sd[k] = sd[k].to(dtype)

        new_sd_size = sum(t.nelement() * t.element_size() for t in sd.values())
        self._ram_cache.make_room(new_sd_size)

        init_kwargs = _wan_vae_init_kwargs_for(config.latent_channels)
        with accelerate.init_empty_weights():
            model = AutoencoderKLWan(**init_kwargs)

        load_state_dict_ignoring_extras(model, sd, source="Wan VAE checkpoint", assign=True)
        model.eval()
        return model

    def _load_wan_vae_diffusers(self, config: VAE_Diffusers_Wan_Config) -> AnyModel:
        """Load a Wan 2.2 VAE from a flat diffusers folder (AutoencoderKLWan).

        The standalone install ``Wan-AI/Wan2.2-T2V-A14B-Diffusers::vae`` lands as a
        single-class folder (``config.json`` + ``diffusion_pytorch_model.safetensors``,
        no ``model_index.json``). The generic loader rejects this when a
        ``submodel_type`` is requested — we always pass ``SubModelType.VAE`` from
        the model loader invocation since that's how cached entries are keyed.
        Loading ``AutoencoderKLWan`` directly here sidesteps the submodel check.

        Runs in the configured precision except float16 -- see `_wan_family_dtype`.
        """
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

        from invokeai.backend.wan.rocm_causal_conv3d import patch_wan_causal_conv3d_for_rocm

        patch_wan_causal_conv3d_for_rocm()
        return AutoencoderKLWan.from_pretrained(
            config.path,
            torch_dtype=_wan_family_dtype(self._torch_dtype),
            local_files_only=True,
        )

    def _load_wan_family_vae(self, path: str) -> AnyModel:
        """Load the 16-channel Wan 2.1 VAE from a single file in its original (non-diffusers) layout.

        Two registrations reach this: `VAE_Checkpoint_Anima_Config`, and the community `qwen-image`
        redistribution of the same 194-tensor checkpoint.

        Converts and constructs rather than calling `AutoencoderKLWan.from_single_file`, which would
        fetch `Wan-AI/Wan2.1-T2V-14B-Diffusers::vae/config.json` over HTTP at load time and then
        load non-strictly. The fetched config is a strict subset of diffusers' `AutoencoderKLWan`
        defaults with identical values, so `z_dim=16` builds the same 194-tensor module.

        A key the conversion did not produce is the failure that matters: `from_single_file` leaves
        that parameter on the meta device and the model only fails at the first decode, so it is
        checked here instead. Keys the module has no use for are not an error -- a redistribution
        may carry extras -- but they are worth a line in the log.

        Reads safetensors and pickled checkpoints alike, and runs in the configured precision except
        float16, which is unstable on the Wan VAE -- see `_wan_family_dtype`.
        """
        import accelerate
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
        from diffusers.models.autoencoders import AutoencoderKLWan

        from invokeai.backend.wan.rocm_causal_conv3d import patch_wan_causal_conv3d_for_rocm

        patch_wan_causal_conv3d_for_rocm()

        dtype = _wan_family_dtype(self._torch_dtype)
        sd = _read_checkpoint(path)
        reject_quantized_side_channel(sd, f"Wan VAE checkpoint {Path(path).name}")
        sd = convert_wan_vae_to_diffusers(sd)
        for k in list(sd.keys()):
            if sd[k].is_floating_point():
                sd[k] = sd[k].to(dtype)

        self._ram_cache.make_room(sum(t.nelement() * t.element_size() for t in sd.values()))

        with accelerate.init_empty_weights():
            model = AutoencoderKLWan(z_dim=16)

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if missing:
            raise ValueError(
                f"{path} does not convert to a complete Wan 2.1 VAE: {len(missing)} tensors are "
                f"missing, starting with {sorted(missing)[:5]}."
            )
        if unexpected:
            self._logger.warning(f"{path} carries {len(unexpected)} tensors the Wan 2.1 VAE does not use.")

        model.eval()
        return model

    def _load_qwen_image_vae(self, config: VAE_Checkpoint_QwenImage_Config) -> AnyModel:
        """Load a Qwen Image VAE from a single-file checkpoint.

        Two layouts reach this method, and each is recognised by a key it must carry. Files exported
        from the Qwen-Image repo carry the diffusers state-dict keys and are loaded directly,
        because `AutoencoderKLQwenImage` registers no single-file conversion in diffusers. Community
        redistributions carry the original Wan-family layout, which needs converting; loading those
        into `AutoencoderKLQwenImage` with `strict=True` failed with 194 missing keys, which made a
        VAE unusable purely because of the base it happened to be probed as.

        Both tests are positive. "Not the diffusers layout, therefore Wan" sent anything else --
        a truncated download, an unrelated autoencoder -- into a conversion that silently produces
        nothing the module recognises, where identification's own `strict=True` used to raise.
        """
        import accelerate
        from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

        keys = _checkpoint_keys(config.path)

        if _WAN_LAYOUT_MARKER in keys:
            return self._load_wan_family_vae(config.path)

        if _QWEN_IMAGE_LAYOUT_MARKER not in keys:
            raise ValueError(
                f"{config.path} is not a Qwen-Image VAE in either known layout: it carries neither "
                f"`{_QWEN_IMAGE_LAYOUT_MARKER}` (the diffusers export) nor `{_WAN_LAYOUT_MARKER}` "
                f"(the original Wan-family layout)."
            )

        sd = _read_checkpoint(config.path)
        reject_quantized_side_channel(sd, f"Qwen-Image VAE checkpoint {Path(config.path).name}")

        # The same autoencoder as the Wan-family layout above, so the same precision policy.
        dtype = _wan_family_dtype(self._torch_dtype)
        for k in list(sd.keys()):
            if sd[k].is_floating_point():
                sd[k] = sd[k].to(dtype)

        new_sd_size = sum(t.nelement() * t.element_size() for t in sd.values())
        self._ram_cache.make_room(new_sd_size)

        with accelerate.init_empty_weights():
            model = AutoencoderKLQwenImage()

        load_state_dict_ignoring_extras(model, sd, source="Qwen-Image VAE checkpoint", assign=True)
        model.eval()
        return model

    # NOTE: keep the `.eval()` at the end of this method in step with `_load_qwen_image_vae` above —
    # both build the module by hand instead of via `from_pretrained`, which would have done it.
    def _load_sdnq_vae(self, model_path: Path) -> AnyModel:
        """Load SDNQ-quantized VAE with on-the-fly dequantization."""
        # Find the safetensors source. Prefer a single canonical file; otherwise hand the whole
        # directory to sdnq_sd_loader, which merges arbitrarily named / sharded safetensors files.
        model_file = model_path / "diffusion_pytorch_model.safetensors"
        if not model_file.exists():
            model_file = model_path / "model.safetensors"
        source = model_file if model_file.exists() else model_path

        # Load SDNQ state dict
        sd = sdnq_sd_loader(source, compute_dtype=self._torch_dtype)

        # Create empty model from config
        with accelerate.init_empty_weights():
            model = AutoencoderKL.from_config(AutoencoderKL.load_config(model_path, local_files_only=True))

        # Load state dict with SDNQTensor objects. AutoencoderKL has no tied weights, so a complete
        # state dict is expected — a missing key would leave a required parameter on the meta device.
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        raise_on_incomplete_sdnq_load("SDNQ VAE", missing, unexpected)
        model.eval()
        return model
