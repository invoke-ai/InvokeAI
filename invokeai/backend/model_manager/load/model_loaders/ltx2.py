"""Loader registrations for LTX-2 (Lightricks) audio-video generation models.

Three records make a generation (see ``configs/main.py`` for the split):

- ``Main`` / ``Checkpoint`` -- the 22B transformer as one official-layout single file: bf16, or Comfy
  ``int8_tensorwise``(+convrot) / ``nvfp4`` with the layers named by a marker or in the header.
  Quantized linears stay packed and dequantize per forward (``Int8ConvrotLinear`` / ``NVFP4Linear``);
  bf16 files take the FP8-storage path when the device policy asks for it.
- ``Main`` / ``Diffusers`` -- the component folder: video VAE, audio VAE, vocoder, text projection
  and connectors, latent upsamplers, each an official-layout single file mapped onto its diffusers
  class here. A folder that also holds a transformer file serves that too.
- ``Gemma4Encoder`` / ``Gemma4Encoder`` -- the Lightricks-tuned Gemma-4-12B text tower with its
  tokenizer, loaded into ``transformers.Gemma4TextModel`` (bf16 or int8-convrot).

Every model class is the installed diffusers' / transformers' own; the pinned configs and key maps
live in ``invokeai.backend.ltx2.component_configs`` and were verified shape-for-shape against the
released 2.5 files.

Cache accounting follows the discipline the loader-seam tests pin: ``get_size_fs`` sizes the file(s)
a submodel is read from so ``_load_and_cache`` reserves the read before it happens, and every path
reserves what a cast will produce *before* widening a tensor.
"""

import json
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import accelerate
import torch

from invokeai.backend.ltx2 import checkpoint_layout as layout
from invokeai.backend.ltx2 import component_configs as cc
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.gemma4_encoder import Gemma4Encoder_Gemma4Encoder_LTX2_Config
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_LTX2_Config, Main_Diffusers_LTX2_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders._single_file_guards import (
    reject_float8_weights,
    reject_formats_declared_in_the_header,
)
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.quantization.fp8_scaled import (
    cast_state_dict,
    parse_quantization_metadata,
    predict_cast_state_dict_size,
    read_safetensors_metadata,
    strip_layer_path_prefix,
)
from invokeai.backend.quantization.int8_convrot import (
    INT8_TENSORWISE_FORMAT,
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    reject_unmarked_int8_weights,
    resolve_quantized_module_paths,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.quantization.nvfp4 import (
    NVFP4_FORMAT,
    install_nvfp4_layers,
    pop_nvfp4_layers,
    predict_nvfp4_install_size,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, reject_incomplete_load

_TRANSFORMER_SUPPORTED_NOTE = (
    "Only unquantized (bf16) LTX-2 transformer files, and Comfy 'int8_tensorwise' (int8-convrot) or 'nvfp4' "
    "files whose layers are named by a marker or in the header, are supported."
)
_ENCODER_SUPPORTED_NOTE = (
    "Only unquantized (bf16) and Comfy 'int8_tensorwise' (int8-convrot) LTX-2 Gemma-4 encoder files are supported."
)
_COMPONENT_SUPPORTED_NOTE = "LTX-2 component files (VAEs, vocoder, connectors, upsamplers) must be unquantized."

# ``AutoTokenizer`` sniffs the folder's ``config.json`` before it reads ``tokenizer_config.json``.
# The release declares ``gemma4_unified_text``, which this transformers version does not register,
# so ``AutoConfig`` raises and the fallback builds the base ``PreTrainedConfig``, whose
# ``model_type`` is "" -- reported at warning level as "You are using a model of type
# `gemma4_unified_text` to instantiate a model of type ``". That sniffed config is then discarded:
# the tokenizer class comes from ``tokenizer_config.json``, which names ``GemmaTokenizer``.
#
# The marker names the released type because the same line reports every unregistered one, with the
# declared type interpolated: the probe in ``configs/gemma4_encoder.py`` also matches a folder on
# ``architectures`` alone, so a repack declaring something else must keep the one message that names
# what it declares. The filter sits on the emitting logger so it runs in ``Logger.handle``, before
# the record reaches transformers' own handler (transformers attaches one to its root logger and,
# outside CI, does not propagate to InvokeAI's), and is scoped to the loading thread because
# verbosity is process-global and loads run alongside generation and the image index worker.
_CONFIG_LOGGER_NAME = "transformers.configuration_utils"
_RELEASE_MODEL_TYPE = "gemma4_unified_text"
_UNREGISTERED_MODEL_TYPE_MARKER = f"model of type `{_RELEASE_MODEL_TYPE}` to instantiate"

_tokenizer_load = threading.local()


class _UnregisteredModelTypeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(_tokenizer_load, "active", False):
            return True
        return _UNREGISTERED_MODEL_TYPE_MARKER not in record.getMessage()


logging.getLogger(_CONFIG_LOGGER_NAME).addFilter(_UnregisteredModelTypeFilter())


@contextmanager
def _quiet_unregistered_model_type() -> Iterator[None]:
    """Drop the config-sniff mismatch warning for a tokenizer load made on this thread."""
    previously_active = getattr(_tokenizer_load, "active", False)
    _tokenizer_load.active = True
    try:
        yield
    finally:
        _tokenizer_load.active = previously_active


_ROLE_LABELS = {
    layout.ROLE_VIDEO_VAE: "video VAE",
    layout.ROLE_AUDIO_VAE: "audio VAE",
    layout.ROLE_VOCODER: "vocoder",
    layout.ROLE_TEXT_PROJECTION: "text projection",
    layout.ROLE_VIDEO_CONNECTOR: "video text connector",
    layout.ROLE_AUDIO_CONNECTOR: "audio text connector",
    layout.ROLE_SPATIAL_UPSAMPLER: "spatial latent upsampler",
    layout.ROLE_TEMPORAL_UPSAMPLER: "temporal latent upsampler",
    layout.ROLE_TRANSFORMER: "transformer",
}

# Which component files a submodel is read from; the connectors join three.
_SUBMODEL_ROLES: dict[SubModelType, tuple[str, ...]] = {
    SubModelType.Transformer: (layout.ROLE_TRANSFORMER,),
    SubModelType.VAE: (layout.ROLE_VIDEO_VAE,),
    SubModelType.AudioVAE: (layout.ROLE_AUDIO_VAE,),
    SubModelType.Vocoder: (layout.ROLE_VOCODER,),
    SubModelType.Connectors: (layout.ROLE_TEXT_PROJECTION, layout.ROLE_VIDEO_CONNECTOR, layout.ROLE_AUDIO_CONNECTOR),
    SubModelType.LatentUpsampler: (layout.ROLE_SPATIAL_UPSAMPLER,),
    SubModelType.TemporalLatentUpsampler: (layout.ROLE_TEMPORAL_UPSAMPLER,),
}

_CONNECTOR_KEY_PREFIXES = (
    "video_embeddings_connector.",
    "audio_embeddings_connector.",
    "text_embedding_projection.",
)


def _component_path(config: Main_Diffusers_LTX2_Config, role: str) -> Path:
    """The file a component folder holds for ``role``, or an actionable error naming what to install."""
    name = config.components.get(role)
    path = Path(config.path) / name if name is not None else None
    if path is None or not path.is_file():
        what = f"has no {_ROLE_LABELS.get(role, role)} file" if name is None else f"is missing its {name}"
        raise ValueError(
            f"This LTX-2 component folder {what}. Install the LTX-2.5 components (the starter model) or add the "
            "missing file to the folder."
        )
    return path


def _read_dense_component(
    path: Path, what: str, logger: Any, key_filter: Callable[[str], bool] | None = None
) -> dict[str, torch.Tensor]:
    """Read an unquantized component file, refusing a quantized one by name before the tensor read.

    With ``key_filter`` only the matching tensors are read (through the safetensors header), for a
    component that shares its file with another -- the official transformer files bundle the two
    connectors beside 38 GB of transformer weights.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    reject_formats_declared_in_the_header(path, what, logger, frozenset(), _COMPONENT_SUPPORTED_NOTE)
    if key_filter is None:
        sd = load_file(path)
    else:
        with safe_open(path, framework="pt", device="cpu") as f:
            sd = {key: f.get_tensor(key) for key in f.keys() if key_filter(key)}
    reject_float8_weights(sd, what, path, _COMPONENT_SUPPORTED_NOTE)
    # An int8 build of a component declares itself nowhere this loader reads (it accepts no
    # quantized components at all); refuse it by name rather than at the strict load.
    reject_unmarked_int8_weights(sd, {}, "LTX-2")
    return sd


def _assign_state_dict(model: torch.nn.Module, sd: dict[str, torch.Tensor], what: str) -> torch.nn.Module:
    """Strict assignment onto a meta-built model; every tensor must land, and nothing may be left over."""
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading {what}: {sorted(unexpected)[:5]}...")
    if missing:
        raise RuntimeError(f"Missing keys loading {what}: {sorted(missing)[:5]}...")
    return model


class _LTX2ComponentLoading:
    """The per-component loaders, shared by the folder loader and (for the transformer) the
    single-file loader. Mixed into ``ModelLoader`` subclasses, so ``self._ram_cache`` / ``_logger`` exist."""

    _ram_cache: Any
    _logger: Any

    def _reserve_for_cast(self, sd: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
        """Reserve what the cast to ``dtype`` will leave resident, before any tensor widens."""
        self._ram_cache.make_room(predict_cast_state_dict_size(sd, dtype, keep_fp8=False))

    def _build_dense(
        self, factory: Callable[[], torch.nn.Module], sd: dict[str, torch.Tensor], what: str, dtype: torch.dtype
    ) -> AnyModel:
        self._reserve_for_cast(sd, dtype)
        cast_state_dict(sd, dtype, keep_fp8=False)
        with accelerate.init_empty_weights():
            model = factory()
        return _assign_state_dict(model, sd, what)

    def _load_video_vae(self, path: Path) -> AnyModel:
        from diffusers import AutoencoderKLLTX2Video

        dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        sd = cc.convert_video_vae_keys(_read_dense_component(path, "LTX-2 video VAE", self._logger))
        return self._build_dense(
            lambda: AutoencoderKLLTX2Video(**cc.LTX2_5_VIDEO_VAE_CONFIG), sd, "LTX-2 video VAE", dtype
        )

    def _load_audio_vae(self, path: Path) -> AnyModel:
        from diffusers import AutoencoderKLLTX2Audio

        # Kept fp32 like the H3 audio VAE: it is small (~0.1 GB) and half-precision artifacts in
        # decoded audio are audible.
        sd = cc.convert_audio_vae_keys(_read_dense_component(path, "LTX-2 audio VAE", self._logger))
        return self._build_dense(
            lambda: AutoencoderKLLTX2Audio(**cc.LTX2_5_AUDIO_VAE_CONFIG), sd, "LTX-2 audio VAE", torch.float32
        )

    def _load_vocoder(self, path: Path) -> AnyModel:
        from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

        # fp32 for the same reason as the audio VAE. The anti-aliasing resampler filter is a
        # non-persistent buffer the constructor computes; buffers are real even under the meta
        # context, so the strict load below never looks for it.
        sd = cc.convert_vocoder_keys(_read_dense_component(path, "LTX-2 vocoder", self._logger))
        return self._build_dense(
            lambda: LTX2VocoderWithBWE(**cc.LTX2_5_VOCODER_CONFIG), sd, "LTX-2 vocoder", torch.float32
        )

    def _load_connectors(self, config: Main_Diffusers_LTX2_Config) -> AnyModel:
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

        dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        sd: dict[str, torch.Tensor] = {}
        for role in _SUBMODEL_ROLES[SubModelType.Connectors]:
            path = _component_path(config, role)
            raw = _read_dense_component(
                path,
                f"LTX-2 {_ROLE_LABELS[role]}",
                self._logger,
                key_filter=lambda key: layout.strip_transformer_prefix(key).startswith(_CONNECTOR_KEY_PREFIXES),
            )
            sd.update(cc.convert_connector_keys(raw))
        return self._build_dense(
            lambda: LTX2TextConnectors(**cc.LTX2_5_CONNECTORS_CONFIG), sd, "LTX-2 text connectors", dtype
        )

    def _load_upsampler(self, path: Path, temporal: bool) -> AnyModel:
        from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

        dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        what = "LTX-2 temporal latent upsampler" if temporal else "LTX-2 spatial latent upsampler"
        params = cc.LTX2_5_TEMPORAL_UPSAMPLER_CONFIG if temporal else cc.LTX2_5_SPATIAL_UPSAMPLER_CONFIG
        sd = _read_dense_component(path, what, self._logger)
        return self._build_dense(lambda: LTX2LatentUpsamplerModel(**params), sd, what, dtype)

    def _load_transformer_from_file(self, path: Path, config: AnyModelConfig) -> AnyModel:
        """The transformer from one official-layout file: bf16, int8-convrot or nvfp4.

        Formats this loader cannot read are refused from the header before the tensor read; every
        reservation is made while the state dict still holds its packed dtypes; nvfp4 layers are
        taken out before either side-channel reader can pair their scales with the wrong weight.
        """
        from diffusers import LTX2VideoTransformer3DModel
        from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers
        from safetensors.torch import load_file

        assert isinstance(self, ModelLoader)
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        what = "LTX-2 transformer"

        reject_formats_declared_in_the_header(
            path, what, self._logger, {INT8_TENSORWISE_FORMAT, NVFP4_FORMAT}, _TRANSFORMER_SUPPORTED_NOTE
        )
        sd = load_file(path)
        metadata = read_safetensors_metadata(path, self._logger)

        # Bare names from here on (the prefix is optional in the official layout, and the header
        # names quantized layers without it). Components a bundled release stores beside the
        # transformer are not this model's: dropped by their official prefixes, before any accounting.
        foreign: list[str] = []
        for key in list(sd):
            bare = layout.strip_transformer_prefix(key)
            if bare.startswith(cc.TRANSFORMER_FOREIGN_PREFIXES):
                foreign.append(key)
                del sd[key]
            elif bare != key:
                sd[bare] = sd.pop(key)
        if foreign:
            self._logger.info(f"{what}: ignoring {len(foreign)} tensor(s) of bundled components (e.g. {foreign[0]}).")

        # Official -> diffusers names, recorded for every key (weights, scales and the nvfp4 payloads
        # about to be popped) so the quantization side channel follows its weights. The converter
        # re-inserts renamed keys at the end of the dict, so positions cannot be paired; sentinels can.
        renamed = convert_ltx2_transformer_to_diffusers({key: key for key in sd})
        renamed, _ = cc.finish_transformer_keys(renamed)
        key_map = {old: new for new, old in renamed.items()}

        def diffusers_module_path(official_path: str) -> str:
            return key_map.get(f"{official_path}.weight", f"{official_path}.weight")[: -len(".weight")]

        # Comfy's nvfp4 builds name their layers in the header. Take them out before either
        # side-channel reader pairs a scale with a weight, and before the undeclared-fp8 refusal:
        # nvfp4's block scales are float8 tensors of their own.
        header_layers = strip_layer_path_prefix(parse_quantization_metadata(metadata))
        nvfp4_payloads = {
            diffusers_module_path(official): payload
            for official, payload in pop_nvfp4_layers(sd, header_layers=header_layers).items()
        }
        reject_float8_weights(sd, what, path, _TRANSFORMER_SUPPORTED_NOTE)

        # ``int8_tensorwise`` has to be recognised before the key conversion: that conversion renames
        # ``.weight`` by substring and carries a sibling ``.weight_scale`` along but not ``.comfy_quant``.
        int8_markers = extract_int8_convrot_markers(sd)
        reject_unmarked_int8_weights(sd, int8_markers, "LTX-2")
        # In place: the dict handed in is the one the cache-room records are taken on.
        for key in set(sd) - set(drop_unconsumed_quantization_sidecars(sd)):
            del sd[key]
        quantized = resolve_quantized_module_paths(int8_markers, key_map)
        for key in list(sd):
            sd[key_map[key]] = sd.pop(key)

        # The keyframe absolute position embedding is a 2.5.1 addition that the nvfp4 release omits;
        # its presence is a fact about the file, not the generation.
        transformer_config = {
            **cc.LTX2_5_TRANSFORMER_CONFIG,
            "use_keyframes_abs_pos_embedding": "keyframes_abs_pos_embedding" in sd,
        }
        with accelerate.init_empty_weights():
            model = LTX2VideoTransformer3DModel(**transformer_config)
        skip_patterns = _model_declared_skip_patterns(model)
        nvfp4_bytes = predict_nvfp4_install_size(model, nvfp4_payloads, model_dtype, skip_patterns)

        if quantized:
            quantized = install_int8_convrot_layers(
                model,
                sd,
                quantized,
                model_dtype,
                architecture="LTX-2",
                reserve=self._ram_cache.make_room,
                skip_patterns=skip_patterns,
                extra_reserved_bytes=nvfp4_bytes,
            )
            self._logger.info(
                f"{what}: kept {len(quantized)} of {len(int8_markers)} layer(s) in int8 "
                "(int8_tensorwise checkpoint, dequantized per forward)"
            )
        else:
            reserve_for_load(
                self._ram_cache.make_room,
                sd,
                model_dtype,
                keep_fp8=False,
                model=model,
                skip_patterns=skip_patterns,
                fp8_layers={},
                nvfp4_payloads=nvfp4_payloads,
            )
            cast_state_dict(sd, model_dtype, keep_fp8=False, model=model, skip_patterns=skip_patterns)

        if nvfp4_payloads:
            packed = install_nvfp4_layers(model, sd, nvfp4_payloads, model_dtype, skip_patterns)
            self._logger.info(f"{what}: kept {packed} of {len(nvfp4_payloads)} nvfp4 layer(s) packed.")

        load_state_dict_ignoring_extras(model, sd, source=f"{what} single file", assign=True, allow_missing=True)
        reject_incomplete_load(model, what=f"{what} single file")
        # ``assign=True`` aliases every param to its ``sd`` tensor; drop the dict's references before
        # the FP8 cast or each original stays reachable while its fp8 copy is allocated.
        sd.clear()

        if not quantized and not nvfp4_payloads:
            model = self._apply_fp8_layerwise_casting(model, config, SubModelType.Transformer)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.LTX2, type=ModelType.Main, format=ModelFormat.Diffusers)
class LTX2FolderModel(_LTX2ComponentLoading, ModelLoader):
    """Loader for an LTX-2 component folder (and its transformer, when the folder has one)."""

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """The file(s) the submodel is read from -- there is no ``<submodel>/`` subfolder to size."""
        if not isinstance(config, Main_Diffusers_LTX2_Config) or submodel_type not in _SUBMODEL_ROLES:
            return 0
        files = {config.components.get(role) for role in _SUBMODEL_ROLES[submodel_type]} - {None}
        return sum((model_path / name).stat().st_size for name in files if (model_path / name).is_file())

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Diffusers_LTX2_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for an LTX-2 folder loader.")
        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading LTX-2 folders.")

        match submodel_type:
            case SubModelType.Transformer:
                if layout.ROLE_TRANSFORMER not in config.components:
                    raise ValueError(
                        "This LTX-2 folder holds only the shared components - it is a components-only install. "
                        "In the LTX-2 Model Loader, select a single-file LTX-2 transformer (bf16 or int8-convrot) "
                        "to use with it."
                    )
                return self._load_transformer_from_file(_component_path(config, layout.ROLE_TRANSFORMER), config)
            case SubModelType.VAE:
                return self._load_video_vae(_component_path(config, layout.ROLE_VIDEO_VAE))
            case SubModelType.AudioVAE:
                return self._load_audio_vae(_component_path(config, layout.ROLE_AUDIO_VAE))
            case SubModelType.Vocoder:
                return self._load_vocoder(_component_path(config, layout.ROLE_VOCODER))
            case SubModelType.Connectors:
                return self._load_connectors(config)
            case SubModelType.LatentUpsampler:
                return self._load_upsampler(_component_path(config, layout.ROLE_SPATIAL_UPSAMPLER), temporal=False)
            case SubModelType.TemporalLatentUpsampler:
                return self._load_upsampler(_component_path(config, layout.ROLE_TEMPORAL_UPSAMPLER), temporal=True)
            case SubModelType.TextEncoder | SubModelType.Tokenizer:
                raise ValueError(
                    "LTX-2 component folders carry no text encoder. The Gemma-4 encoder is its own model record; "
                    "select it in the LTX-2 Model Loader's text encoder field."
                )
            case _:
                raise ValueError(f"Unsupported submodel type {submodel_type} for LTX-2 folders.")


@ModelLoaderRegistry.register(base=BaseModelType.LTX2, type=ModelType.Main, format=ModelFormat.Checkpoint)
class LTX2CheckpointModel(_LTX2ComponentLoading, ModelLoader):
    """Loader for LTX-2 single-file transformers (bf16, Comfy int8-convrot or header-named nvfp4)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_LTX2_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for an LTX-2 checkpoint loader.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                "LTX-2 single-file checkpoints serve only the transformer. The VAEs, vocoder and connectors "
                "come from an installed LTX-2 component folder and the text encoder from the Gemma-4 encoder "
                f"record. Received: {submodel_type.value if submodel_type else 'None'}"
            )
        return self._load_from_singlefile(config)

    def _load_from_singlefile(self, config: Main_Checkpoint_LTX2_Config) -> AnyModel:
        return self._load_transformer_from_file(Path(config.path), config)


@ModelLoaderRegistry.register(base=BaseModelType.LTX2, type=ModelType.Gemma4Encoder, format=ModelFormat.Gemma4Encoder)
class LTX2Gemma4EncoderModel(ModelLoader):
    """Loader for the LTX-2.5 Gemma-4-12B text tower (bf16 or Comfy int8-convrot) and its tokenizer.

    Only the text tower is built (``Gemma4TextModel``, verified key-for-key against the release): the
    unified model's vision/audio embedders are never used, and the LM head is tied and never run.
    Quantized files keep their int8 weights + per-channel scales resident (``Int8ConvrotLinear``).
    """

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """Size the one weight file (the tokenizer is a few MB); the folder may nest it one level down."""
        if not isinstance(config, Gemma4Encoder_Gemma4Encoder_LTX2_Config):
            return 0
        root = model_path / config.subfolder if config.subfolder else model_path
        if submodel_type is SubModelType.TextEncoder:
            weight = root / config.weight_file
            return weight.stat().st_size if weight.is_file() else 0
        return sum(p.stat().st_size for p in root.glob("tokenizer*") if p.is_file())

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Gemma4Encoder_Gemma4Encoder_LTX2_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for an LTX-2 Gemma-4 loader.")
        root = Path(config.path) / config.subfolder if config.subfolder else Path(config.path)
        match submodel_type:
            case SubModelType.Tokenizer:
                from transformers import AutoTokenizer

                # ``fix_mistral_regex=False``: transformers' Mistral-regex heuristic short-circuits on
                # the sibling ``config.json``'s ``transformers_version``, which the release does not
                # carry, so it advises the fix for this folder whatever the model is. Taking the advice
                # prepends the Tekken split regex to Gemma's pre-tokenizer and shatters words
                # ("inconsistent" becomes "in" + "consistent"), which would mis-tokenize every prompt.
                # Passing it explicitly keeps the released pre-tokenizer -- what the reference
                # pipelines encode with -- and states the decision instead of repeating the advice.
                with _quiet_unregistered_model_type():
                    return AutoTokenizer.from_pretrained(root, local_files_only=True, fix_mistral_regex=False)
            case SubModelType.TextEncoder:
                return self._load_text_encoder(root / config.weight_file, root / "config.json")
            case _:
                raise ValueError(
                    "The LTX-2 Gemma-4 encoder serves the text encoder and its tokenizer. "
                    f"Received: {submodel_type.value if submodel_type else 'None'}"
                )

    def _load_text_encoder(self, weight_path: Path, config_path: Path) -> AnyModel:
        from safetensors.torch import load_file
        from transformers import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

        what = "LTX-2 Gemma-4 encoder"
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())

        reject_formats_declared_in_the_header(
            weight_path, what, self._logger, {INT8_TENSORWISE_FORMAT}, _ENCODER_SUPPORTED_NOTE
        )
        sd = load_file(weight_path)
        reject_float8_weights(sd, what, weight_path, _ENCODER_SUPPORTED_NOTE)

        # The unified layout nests the tower under ``model.language_model`` beside vision/audio
        # embedders and the tied head; Lightricks' with-proj files add the connector projections under
        # ``hf_asset__``. None of that is the tower's. ``Gemma4TextModel`` names its own tensors
        # without the ``model.`` prefix.
        dropped = (
            "hf_asset__",
            "model.vision_tower.",
            "model.audio_tower.",
            "model.embed_vision.",
            "model.embed_audio.",
            "lm_head.",
        )
        for key in list(sd):
            if key.startswith(dropped):
                del sd[key]
                continue
            bare = key
            for prefix in ("model.language_model.", "model."):
                if bare.startswith(prefix):
                    bare = bare[len(prefix) :]
                    break
            if bare != key:
                sd[bare] = sd.pop(key)

        int8_markers = extract_int8_convrot_markers(sd)
        reject_unmarked_int8_weights(sd, int8_markers, "LTX-2 Gemma-4")
        for key in set(sd) - set(drop_unconsumed_quantization_sidecars(sd)):
            del sd[key]

        config_dict = json.loads(config_path.read_text(encoding="utf-8"))
        text_config = (
            config_dict.get("text_config") if isinstance(config_dict.get("text_config"), dict) else config_dict
        )
        te_config = Gemma4TextConfig(
            **{k: v for k, v in text_config.items() if k not in ("architectures", "model_type", "transformers_version")}
        )
        with accelerate.init_empty_weights():
            model = Gemma4TextModel(te_config)

        if int8_markers:
            # The shared helper: foreign-scale refusal, reservation before the split, the dense cast
            # and the module swap, in that order.
            install_int8_convrot_layers(
                model, sd, int8_markers, model_dtype, architecture="LTX-2 Gemma-4", reserve=self._ram_cache.make_room
            )
        else:
            self._ram_cache.make_room(predict_cast_state_dict_size(sd, model_dtype, keep_fp8=False))
            cast_state_dict(sd, model_dtype, keep_fp8=False)

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if unexpected:
            raise RuntimeError(f"Unexpected keys loading {what}: {sorted(unexpected)[:5]}...")
        # The rotary inverse-frequency buffers are computed by the constructor, not stored.
        stray = [k for k in missing if not k.startswith("rotary_emb.")]
        if stray:
            raise RuntimeError(f"Missing keys loading {what}: {sorted(stray)[:5]}...")
        sd.clear()
        return model
