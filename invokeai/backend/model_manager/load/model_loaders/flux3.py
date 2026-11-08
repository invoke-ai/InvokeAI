"""Loaders for FLUX.3 Image transformers (Klein and Klein Base).

The Diffusers loader dispatches submodels explicitly instead of subclassing ``GenericDiffusersLoader``, which
reads ``model_index.json``: FLUX.3 ships a ``modular_model_index.json``.

Submodel map of a Diffusers folder (subfolder == ``SubModelType`` value, except the tokenizer):
- ``transformer``  -> ``Flux3Transformer2DModel`` (bf16, QK norms in fp32)
- ``vae``          -> ``AutoencoderKLFlux2`` (the FLUX.2 VAE, byte-identical)
- ``text_encoder`` -> ``Qwen3VLModel`` (Qwen3-VL 4B Instruct), visual tower dropped
- tokenizer        -> ``AutoTokenizer`` from ``text_encoder/``, where the release keeps its tokenizer files

Single-file and GGUF builds are only the transformer, in Black Forest Labs' key layout. The single-file loader
also takes the published quantized builds: BFL's per-row scaled fp8 (`variants/fp8r`) and ModelOpt nvfp4
(`variants/nvfp4`), and Comfy-Org's `int8_convrot` and nvfp4. The file says which it is.
"""

from collections.abc import Callable, Mapping
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import accelerate
import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3VLModel

from invokeai.backend.flux3.checkpoint_quantization import flux3_header_layers, reject_unloadable_quantization
from invokeai.backend.flux3.conversion import (
    convert_flux3_layer_entries,
    convert_flux3_native_to_diffusers,
    convert_flux3_nvfp4_payloads,
    flux3_transformer_config,
)
from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_Flux3_Config,
    Main_Diffusers_Flux3_Config,
    Main_GGUF_Flux3_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.model_manager.util.qwen3_vl import drop_qwen3vl_visual_tower, normalize_qwen3vl_rope_config
from invokeai.backend.quantization.dequantizing_linear import DequantizingLinear
from invokeai.backend.quantization.fp8_scaled import (
    COMFY_QUANT_SUFFIX,
    Fp8ScaledLayer,
    attach_fp8_scales,
    cast_state_dict,
    count_fp8_weights,
    dequantize_fp8_scaled,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    read_safetensors_metadata,
    split_fp8_scaled_layers,
    warn_on_unattached_scales,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader, unpack_ggml_at_load
from invokeai.backend.quantization.int8_convrot import (
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    parse_comfy_quant_marker,
)
from invokeai.backend.quantization.int8_matmul import is_int8_compute_enabled
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.quantization.nvfp4 import (
    install_nvfp4_layers,
    is_modelopt_nvfp4,
    pop_nvfp4_layers,
    relayout_modelopt_nvfp4_payloads,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, reject_incomplete_load

_QK_NORM_WEIGHTS = (".norm_q.weight", ".norm_k.weight")

_logger = InvokeAILogger.get_logger(__name__)


def _keep_qk_norms_in_fp32(model: torch.nn.Module) -> None:
    """Hold the attention QK-norm weights in float32, as diffusers' `_keep_in_fp32_modules` does on a folder load.

    `Flux3AttnProcessor` normalizes a float32 query and key, as the reference implementation does; a bf16
    weight would round the scale that multiplies them.
    """
    for name, module in model.named_modules():
        if name.endswith((".norm_q", ".norm_k")) and getattr(module, "weight", None) is not None:
            module.weight = torch.nn.Parameter(module.weight.detach().to(torch.float32), requires_grad=False)


def _build_transformer(state_dict: dict) -> torch.nn.Module:
    """An empty transformer (on the meta device) shaped by a native FLUX.3 state dict."""
    from diffusers import Flux3Transformer2DModel

    model_config = flux3_transformer_config(state_dict)
    with accelerate.init_empty_weights():
        return Flux3Transformer2DModel(**model_config)


def _reserve_nothing(_bytes: int) -> None:
    """The reservation of a load that happens outside the RAM cache."""


def flux3_transformer_from_native(
    sd: dict,
    dtype: torch.dtype,
    reserve: Optional[Callable[[int], None]] = None,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    keep_fp8: bool = False,
    logger: Optional[Logger] = None,
) -> torch.nn.Module:
    """The transformer of a native single-file FLUX.3 state dict in `dtype`, on the device its tensors are on, the QK
    norms in float32. Empties `sd`.

    `reserve` asks the RAM cache for room once, before anything widens a weight, as `reserve_for_load` does; None asks
    nothing, for a caller that loads outside the cache (the working-memory calibration script). `metadata` is the
    file's safetensors metadata: ComfyUI's nvfp4 build names its layers there, and BFL's says it is a ModelOpt export.
    `keep_fp8` keeps scaled fp8 weights quantized with their scales attached, for the fp8 matmul or FP8 Storage;
    without it they are folded into `dtype`.

    A quantized layer stays quantized where the model has a plain Linear outside its precision-sensitive modules:
    int8 as `Int8ConvrotLinear`, nvfp4 as `NVFP4Linear`, scaled fp8 with `keep_fp8`. Anything else, such as
    `norm_out.linear` (the output adaLN, under the model's `norm` pattern), is decoded into `dtype` with its scale.
    What the loader cannot build is refused first, by the same rules identification applies to the header
    (`flux3/checkpoint_quantization.py`).
    """
    log = logger or _logger
    make_room = reserve or _reserve_nothing
    sd = CheckpointPrefix.detect(sd).strip(sd)
    header_layers = flux3_header_layers(metadata, sd)
    markers = {
        key[: -len(COMFY_QUANT_SUFFIX)]: parse_comfy_quant_marker(value)
        for key, value in sd.items()
        if key.endswith(COMFY_QUANT_SUFFIX)
    }
    reject_unloadable_quantization(sd, header_layers, markers)
    # BFL's nvfp4 build: its layers are named in `header_layers` now and re-laid out once room is reserved for them.
    modelopt = is_modelopt_nvfp4(metadata, sd)

    # Shaped before the nvfp4 layers leave the state dict: `mlp_in`, whose rows give the feedforward width, is one.
    model = _build_transformer(sd)
    nvfp4_payloads = convert_flux3_nvfp4_payloads(pop_nvfp4_layers(sd, header_layers))
    # Before the fp8 hints, which take every `.comfy_quant` marker out of the state dict, the int8 ones included.
    int8_markers = extract_int8_convrot_markers(sd)
    layer_hints: dict[str, Any] = {}
    if int8_markers:
        sd = drop_unconsumed_quantization_sidecars(sd)
    else:
        layer_hints = {**extract_comfy_quant_hints(sd), **header_layers}
    sd = convert_flux3_native_to_diffusers(sd)
    int8_markers = convert_flux3_layer_entries(int8_markers)
    layer_hints = convert_flux3_layer_entries(layer_hints)

    skip_patterns = _model_declared_skip_patterns(model)
    fp8_layers: dict[str, Fp8ScaledLayer] = {}
    if int8_markers:
        # No int8 build carries nvfp4 or scaled fp8 layers beside them; `reject_unloadable_quantization` refused that.
        int8_layers = install_int8_convrot_layers(
            model, sd, int8_markers, dtype, architecture="FLUX.3", reserve=make_room, skip_patterns=skip_patterns
        )
        how = (
            "multiplied in int8 where int8 Compute's kernels cover them, dequantized per forward elsewhere"
            if is_int8_compute_enabled()
            else "dequantized per forward"
        )
        log.info(f"FLUX.3: kept {len(int8_layers)} of {len(int8_markers)} layer(s) in int8 (int8_tensorwise, {how}).")
    else:
        fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)
        reserve_for_load(
            make_room,
            sd,
            dtype,
            keep_fp8=keep_fp8,
            model=model,
            skip_patterns=skip_patterns,
            fp8_layers=fp8_layers,
            nvfp4_payloads=nvfp4_payloads,
        )
        if fp8_layers and not keep_fp8:
            # Kept, they would be dequantized on every forward to save memory nobody asked to save.
            dequantize_fp8_scaled(sd, fp8_layers, dtype)
            fp8_layers = {}
        # The scaled layers the cast would widen are folded with their scale first: `norm_out.linear`, and any
        # layer whose scale `scaled_mm` cannot apply.
        fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, dtype, model=model, skip_patterns=skip_patterns)
        cast_state_dict(sd, dtype, keep_fp8=keep_fp8, model=model, skip_patterns=skip_patterns)
        if nvfp4_payloads:
            if modelopt:
                # After the reservation: the copies it makes are what the reservation counts for the packed layers.
                relayout_modelopt_nvfp4_payloads(nvfp4_payloads)
            packed = install_nvfp4_layers(model, sd, nvfp4_payloads, dtype, skip_patterns)
            log.info(f"FLUX.3: kept {packed} of {len(nvfp4_payloads)} nvfp4 layer(s) packed (dequantized per forward).")

    load_state_dict_ignoring_extras(model, sd, source="FLUX.3 single-file checkpoint", assign=True, allow_missing=True)
    reject_incomplete_load(model, what="FLUX.3 single-file checkpoint")
    sd.clear()
    if fp8_layers:
        warn_on_unattached_scales(log, "FLUX.3", attach_fp8_scales(model, fp8_layers), fp8_layers)
    _keep_qk_norms_in_fp32(model)
    return model


@ModelLoaderRegistry.register(base=BaseModelType.Flux3, type=ModelType.Main, format=ModelFormat.Diffusers)
class Flux3DiffusersModel(ModelLoader):
    """Loads the submodels of a FLUX.3 Diffusers folder."""

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        # The tokenizer lives in `text_encoder/` and has no folder of its own; the generic sizing would fall
        # back to the folder root, which in a full repo download holds BFL's 11 GB single-file package.
        if submodel_type is SubModelType.Tokenizer:
            return 0
        return super().get_size_fs(config, model_path, submodel_type)

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Main_Diffusers_Flux3_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for the FLUX.3 Diffusers loader.")
        if submodel_type is None:
            raise ValueError("A submodel type must be provided when loading a FLUX.3 Diffusers model.")

        model_path = Path(config.path)
        dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())

        match submodel_type:
            case SubModelType.Transformer:
                from diffusers import Flux3Transformer2DModel

                transformer = Flux3Transformer2DModel.from_pretrained(
                    model_path / "transformer", torch_dtype=dtype, local_files_only=True
                )
                return self._apply_fp8_layerwise_casting(transformer, config, submodel_type)
            case SubModelType.VAE:
                from diffusers import AutoencoderKLFlux2

                return AutoencoderKLFlux2.from_pretrained(
                    model_path / "vae", torch_dtype=self._torch_dtype_avoiding_float16(), local_files_only=True
                )
            case SubModelType.TextEncoder:
                encoder_path = model_path / "text_encoder"
                encoder_config = normalize_qwen3vl_rope_config(
                    AutoConfig.from_pretrained(encoder_path, local_files_only=True)
                )
                text_encoder = Qwen3VLModel.from_pretrained(
                    encoder_path,
                    config=encoder_config,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                # FLUX.3 conditions on the language tower's hidden states; the visual tower never runs.
                drop_qwen3vl_visual_tower(text_encoder)
                return text_encoder
            case SubModelType.Tokenizer:
                return AutoTokenizer.from_pretrained(
                    model_path / "text_encoder", local_files_only=True, extra_special_tokens={}
                )
            case _:
                raise ValueError(f"A FLUX.3 Diffusers model has no '{submodel_type.value}' submodel.")


@ModelLoaderRegistry.register(base=BaseModelType.Flux3, type=ModelType.Main, format=ModelFormat.Checkpoint)
class Flux3CheckpointModel(ModelLoader):
    """Loads a FLUX.3 transformer from a single file: bf16 (BFL `model.safetensors`, or its ComfyUI repackage), or one
    of the quantized builds `flux3_transformer_from_native` reads."""

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_Flux3_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for the FLUX.3 single-file loader.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError("A single-file FLUX.3 model holds only the transformer.")

        from safetensors.torch import load_file

        model_path = Path(config.path)
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        model = flux3_transformer_from_native(
            load_file(model_path),
            model_dtype,
            self._ram_cache.make_room,
            metadata=read_safetensors_metadata(model_path, self._logger),
            keep_fp8=self._keep_fp8_weights(config, SubModelType.Transformer),
            logger=self._logger,
        )
        if any(isinstance(module, DequantizingLinear) for module in model.modules()):
            # What FP8 Storage could still cast is the dense remainder: the modulations, the embedders, the output
            # layers and, in the nvfp4 builds, the blocks BFL excludes from quantization. The build left those in bf16
            # on purpose, and the storage cast would round them to unscaled fp8, so it is skipped, as for FLUX.2's int8.
            if self._fp8_storage_asked_for(config, SubModelType.Transformer):
                self._logger.info(
                    "FLUX.3: FP8 Storage skipped - this int8 or nvfp4 build keeps its quantized layers as stored and "
                    "its other layers in the precision the build chose."
                )
            return model
        if kept := count_fp8_weights(model):
            self._logger.info(
                f"FLUX.3: kept {kept} layer(s) in fp8 (fp8 checkpoint, kept for {self._fp8_kept_reason()})."
            )
        return self._apply_fp8_layerwise_casting(model, config, SubModelType.Transformer)


@ModelLoaderRegistry.register(base=BaseModelType.Flux3, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class Flux3GGUFCheckpointModel(ModelLoader):
    """Loads a GGUF-quantized FLUX.3 transformer; the GGML tensors stay packed and dequantize per forward."""

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Main_GGUF_Flux3_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for the FLUX.3 GGUF loader.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError("A GGUF FLUX.3 model holds only the transformer.")

        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        sd = gguf_sd_loader(Path(config.path), compute_dtype=compute_dtype)
        sd = CheckpointPrefix.detect(sd).strip(sd)

        model = _build_transformer(sd)
        sd = convert_flux3_native_to_diffusers(sd)
        # The output adaLN weight comes back unpacked from the [shift, scale] swap; it joins the compute dtype
        # the packed tensors dequantize to.
        adaln = sd.get("norm_out.linear.weight")
        if adaln is not None and not isinstance(adaln, GGMLTensor):
            sd["norm_out.linear.weight"] = adaln.to(compute_dtype)
        # The QK norms are stored unquantized (F32) and stay float32, so they are taken as stored before the
        # unpack below would round them through the bf16 compute dtype.
        for key in [key for key in sd if key.endswith(_QK_NORM_WEIGHTS)]:
            value = sd[key]
            if isinstance(value, GGMLTensor):
                stored = value.quantized_data
                sd[key] = (
                    stored.to(torch.float32)
                    if stored.is_floating_point() and stored.shape == value.shape
                    else value.get_dequantized_tensor().to(torch.float32)
                )
        unpack_ggml_at_load(sd, model, _model_declared_skip_patterns(model), self._ram_cache.make_room)
        load_state_dict_ignoring_extras(model, sd, source="FLUX.3 GGUF checkpoint", assign=True, allow_missing=True)
        reject_incomplete_load(model, what="FLUX.3 GGUF checkpoint")
        _keep_qk_norms_in_fp32(model)
        return model
