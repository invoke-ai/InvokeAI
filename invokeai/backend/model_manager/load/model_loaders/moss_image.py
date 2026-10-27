"""Loaders for OpenMOSS MOSS-Image: diffusers pipelines and single-file transformers.

The pipeline's `model_index.json` names `pipeline_moss_image` -- the repository's remote code -- for its transformer
and scheduler, so `GenericDiffusersLoader` cannot resolve them. The transformer class is vendored under
`invokeai.backend.moss_image`, and each submodel is dispatched here explicitly:

- `transformer`  -> vendored `MOSSImageTransformer` (bf16)
- `text_encoder` -> `Qwen35Encoder` (Qwen3.5-9B, language model only; the vision tower stays on disk)
- `tokenizer`    -> the folder's own Qwen2 tokenizer
- `vae`          -> diffusers `AutoencoderKLFlux2` (FLUX.2 encoder, small decoder)

The scheduler is never loaded: `moss_image_denoise` builds the schedule itself.

Both transformer paths build the model under `init_empty_weights` and assign the checkpoint's tensors, reserving RAM
once first. A diffusers `from_pretrained` would load the same weights, but outside the cache's reservation, and it
installs the RoPE frequencies at the weights' bf16 for the first forward to repair.
"""

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import accelerate
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from invokeai.app.util.path_safety import is_plain_filename
from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_MossImage_Config,
    Main_Diffusers_MossImage_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.qwen3_5_encoder import load_qwen3_5_text_encoder
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.moss_image.pipeline_moss_image import MOSSImageTransformer
from invokeai.backend.quantization.fp8_scaled import cast_state_dict, reject_quantized_side_channel
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import reject_incomplete_load

MOSS_IMAGE_FP8_SKIP_PATTERNS: tuple[str, ...] = (
    r"^video_embedder$",
    r"^time_embedding\.0$",
    r"^shared_block_modulation$",
    r"^video_head_modulation$",
    r"^final_linear_video$",
    r"^text_embedder\.proj$",
    r"^layers\.\d+\.proj\.q$",
)
"""The Linears FP8 Storage must leave in the compute dtype.

The vendored forward reads each of these layers' `weight.dtype` *outside* the layer and casts the activation to it
before the call. With an fp8 weight the activation becomes fp8 before the cast hook can widen the weight back, and the
matmul fails ("mat1 and mat2 must have the same dtype"). A block reads `proj.q`'s dtype for the input of all of its
projections, so `q` stays wide and `k`, `v`, `o` and the FFN are stored in fp8. Norms are not Linears, and
`_FP8_DEFAULT_SKIP_PATTERNS` covers them anyway."""

_LAYER_INDEX = re.compile(r"^layers\.(\d+)\.")


def moss_image_transformer_config(sd: dict[str, Any]) -> dict[str, Any]:
    """The `MOSSImageTransformer` constructor arguments a prefix-stripped single-file checkpoint describes.

    Read from the tensors, not assumed: a finetune at another depth or width would otherwise build the released
    geometry and fail on a shape, or -- with a deeper checkpoint -- would never be noticed by a load that drops keys.
    `rope_base` is the one hyperparameter no tensor records; the released value is the class default.
    """
    try:
        dim, latent_channels = sd["video_embedder.weight"].shape
        _, text_dim = sd["text_embedder.proj.weight"].shape
        _, freq_dim = sd["time_embedding.0.weight"].shape
        _, ffn_dim = sd["layers.0.proj.ffn_down.weight"].shape
        (head_dim,) = sd["layers.0.q_norm.weight"].shape
    except KeyError as e:
        raise ValueError(f"MOSS-Image checkpoint lacks {e.args[0]}, which its geometry is read from.") from e
    indices = {int(match.group(1)) for key in sd if (match := _LAYER_INDEX.match(key))}
    num_layers = max(indices) + 1
    if len(indices) != num_layers:
        missing = sorted(set(range(num_layers)) - indices)
        raise ValueError(f"MOSS-Image checkpoint has gaps in its blocks: {missing[:10]} missing of 0-{num_layers - 1}.")
    return {
        "dim": int(dim),
        "ffn_dim": int(ffn_dim),
        "num_heads": int(dim) // int(head_dim),
        "num_layers": num_layers,
        "text_dim": int(text_dim),
        "freq_dim": int(freq_dim),
        "latent_channels": int(latent_channels),
    }


def moss_image_component_shards(directory: Path) -> Mapping[Path, list[str] | None]:
    """The safetensors files a pipeline component's weights come from, with the keys each one holds.

    With a `*.safetensors.index.json`, its `weight_map` decides: only the shards it names are read, and each key only
    from the shard it names, so a stale shard left beside them by an interrupted re-download cannot overwrite or add
    tensors. Without an index the component must be one file, all of whose keys are read (None).
    """
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError(f"{directory} holds {len(indexes)} safetensors indexes; expected one.")
    if not indexes:
        files = sorted(directory.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError(
                f"{directory} has no safetensors index, so its weights must be one file, but it holds {len(files)}."
            )
        return {files[0]: None}

    with open(indexes[0], "r", encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{indexes[0]} has no weight_map.")
    keys_by_shard: dict[Path, list[str]] = {}
    for key, name in weight_map.items():
        # The index is a file on disk like any other; a name that is not a bare filename would read outside the folder.
        if not isinstance(name, str) or not is_plain_filename(name):
            raise ValueError(f"{indexes[0]} maps {key!r} to {name!r}, which is not a file in {directory}.")
        keys_by_shard.setdefault(directory / name, []).append(key)
    missing = sorted(path.name for path in keys_by_shard if not path.is_file())
    if missing:
        raise ValueError(f"{directory} is missing shard(s) its index names: {', '.join(missing)}.")
    return keys_by_shard


def _read_component_state_dict(directory: Path) -> dict[str, torch.Tensor]:
    """A component's whole state dict, read the way `moss_image_component_shards` says."""
    sd: dict[str, torch.Tensor] = {}
    for path, keys in moss_image_component_shards(directory).items():
        if keys is None:
            sd.update(load_file(path))
            continue
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in keys:
                sd[key] = f.get_tensor(key)
    return sd


class _MossImageTransformerLoading(ModelLoader):
    """The transformer load both MOSS-Image formats share, once each has a state dict and a geometry."""

    def _load_transformer(
        self, config: AnyModelConfig, sd: dict[str, torch.Tensor], geometry: dict[str, Any], what: str
    ) -> AnyModel:
        # Strict below would name an orphaned scale key; this names the cause before anything is reserved.
        reject_quantized_side_channel(sd, what)
        dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())

        with accelerate.init_empty_weights():
            model = MOSSImageTransformer(**geometry)
        model._skip_layerwise_casting_patterns = list(MOSS_IMAGE_FP8_SKIP_PATTERNS)
        # The RoPE frequencies are a derived float32 constant the model recomputes on load. The key is kept only for
        # checkpoint compatibility, so an export that leaves it out is still complete.
        sd.setdefault("rope.bands", model.rope.bands)

        skip_patterns = _model_declared_skip_patterns(model)
        reserve_for_load(
            self._ram_cache.make_room,
            sd,
            dtype,
            keep_fp8=False,
            model=model,
            skip_patterns=skip_patterns,
            fp8_layers={},
            nvfp4_payloads={},
        )
        cast_state_dict(sd, dtype, keep_fp8=False, model=model, skip_patterns=skip_patterns)
        model.load_state_dict(sd, strict=True, assign=True)
        # `assign=True` aliases every parameter to its state dict tensor; the dict would keep the whole model alive
        # through the cast below.
        sd.clear()
        reject_incomplete_load(model, what=what)
        return self._apply_fp8_layerwise_casting(model.eval(), config, SubModelType.Transformer)


@ModelLoaderRegistry.register(base=BaseModelType.MossImage, type=ModelType.Main, format=ModelFormat.Diffusers)
class MossImageDiffusersModel(_MossImageTransformerLoading):
    """Loads MOSS-Image submodels from a diffusers pipeline folder (MOSS-Image Base or SFT)."""

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Main_Diffusers_MossImage_Config):
            raise ValueError(f"Expected Main_Diffusers_MossImage_Config, got {type(config).__name__}.")
        model_path = Path(config.path)

        match submodel_type:
            case SubModelType.Transformer:
                transformer_dir = model_path / "transformer"
                with open(transformer_dir / "config.json", "r", encoding="utf-8") as f:
                    declared = json.load(f)
                geometry = {key: value for key, value in declared.items() if not key.startswith("_")}
                sd = _read_component_state_dict(transformer_dir)
                return self._load_transformer(config, sd, geometry, f"MOSS-Image transformer in {model_path.name}")
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config, model_path / "text_encoder")
            case SubModelType.Tokenizer:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(model_path / "tokenizer", local_files_only=True)
            case SubModelType.VAE:
                from diffusers import AutoencoderKLFlux2

                # The FLUX.2 VAE loaders' dtype: never float16, in which the FLUX autoencoders break.
                return AutoencoderKLFlux2.from_pretrained(
                    model_path / "vae", torch_dtype=self._torch_dtype_avoiding_float16(), local_files_only=True
                )

        raise ValueError(
            f"MOSS-Image pipelines provide the transformer, text_encoder, tokenizer and vae submodels, not "
            f"{submodel_type.value if submodel_type else 'None'}."
        )

    def _load_text_encoder(self, config: Main_Diffusers_MossImage_Config, text_encoder_dir: Path) -> AnyModel:
        from invokeai.backend.qwen3_5.qwen3_5_encoder import qwen3_5_text_config_from_config_json

        with open(text_encoder_dir / "config.json", "r", encoding="utf-8") as f:
            text_config = qwen3_5_text_config_from_config_json(json.load(f))
        # Only the shards the index names. The loader reads every language-model key of the files it is given, so a
        # stale shard among them could replace tensors.
        shards = list(moss_image_component_shards(text_encoder_dir))
        # No FP8 Storage cast follows: text encoders never take it (`_FP8_EXCLUDED_SUBMODEL_TYPES`). `keep_fp8` only
        # keeps a raw-fp8 checkpoint packed for the fp8 matmul.
        return load_qwen3_5_text_encoder(
            self,
            shards,
            text_config,
            # MOSS-Image reads the last hidden state, after the final norm.
            final_norm=True,
            keep_fp8=self._keep_fp8_weights(config, SubModelType.TextEncoder),
            what=f"MOSS-Image text encoder in {Path(config.path).name}",
        )


@ModelLoaderRegistry.register(base=BaseModelType.MossImage, type=ModelType.Main, format=ModelFormat.Checkpoint)
class MossImageCheckpointModel(_MossImageTransformerLoading):
    """Loads a MOSS-Image transformer from a single safetensors file, such as the ComfyUI export.

    The file holds the transformer alone, under the diffusers checkpoint's keys. Its text encoder and VAE are separate
    models, chosen on the loader node.
    """

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_MossImage_Config):
            raise ValueError(f"Expected Main_Checkpoint_MossImage_Config, got {type(config).__name__}.")
        if submodel_type is not None and submodel_type is not SubModelType.Transformer:
            raise ValueError(
                f"A single-file MOSS-Image checkpoint holds only the transformer, not {submodel_type.value}. "
                "Select a standalone Qwen3.5 9B encoder and FLUX.2 VAE on the MOSS-Image model loader."
            )
        model_path = Path(config.path)
        sd = load_file(model_path)
        sd = CheckpointPrefix.detect(sd).strip(sd)
        return self._load_transformer(
            config, sd, moss_image_transformer_config(sd), f"MOSS-Image checkpoint {model_path.name}"
        )
