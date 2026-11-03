"""Loaders for FLUX.3 Image transformers (Klein and Klein Base).

The Diffusers loader dispatches submodels explicitly instead of subclassing ``GenericDiffusersLoader``, which
reads ``model_index.json``: FLUX.3 ships a ``modular_model_index.json``.

Submodel map of a Diffusers folder (subfolder == ``SubModelType`` value, except the tokenizer):
- ``transformer``  -> ``Flux3Transformer2DModel`` (bf16, QK norms in fp32)
- ``vae``          -> ``AutoencoderKLFlux2`` (the FLUX.2 VAE, byte-identical)
- ``text_encoder`` -> ``Qwen3VLModel`` (Qwen3-VL 4B Instruct), visual tower dropped
- tokenizer        -> ``AutoTokenizer`` from ``text_encoder/``, where the release keeps its tokenizer files

Single-file and GGUF builds are only the transformer, in Black Forest Labs' key layout.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Optional

import accelerate
import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3VLModel

from invokeai.backend.flux3.conversion import convert_flux3_native_to_diffusers, flux3_transformer_config
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
from invokeai.backend.quantization.fp8_scaled import cast_state_dict, reject_quantized_side_channel
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader, unpack_ggml_at_load
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, reject_incomplete_load

_QK_NORM_WEIGHTS = (".norm_q.weight", ".norm_k.weight")


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


def flux3_transformer_from_native(
    sd: dict, dtype: torch.dtype, reserve: Optional[Callable[[int], None]] = None
) -> torch.nn.Module:
    """The transformer of a native single-file FLUX.3 state dict in `dtype`, on the device its tensors are on, the QK
    norms in float32. Empties `sd`. `reserve` asks the RAM cache for room before the cast, as `reserve_for_load` does;
    None asks nothing, for a caller that loads outside the cache (the working-memory calibration script)."""
    sd = CheckpointPrefix.detect(sd).strip(sd)
    # Identification refuses quantized builds already; this catches a record made before it did.
    reject_quantized_side_channel(sd, "a FLUX.3 single-file checkpoint")

    model = _build_transformer(sd)
    sd = convert_flux3_native_to_diffusers(sd)
    skip_patterns = _model_declared_skip_patterns(model)
    if reserve is not None:
        reserve_for_load(
            reserve,
            sd,
            dtype,
            keep_fp8=False,
            model=model,
            skip_patterns=skip_patterns,
            fp8_layers={},
            nvfp4_payloads={},
        )
    cast_state_dict(sd, dtype, keep_fp8=False, model=model, skip_patterns=skip_patterns)
    load_state_dict_ignoring_extras(model, sd, source="FLUX.3 single-file checkpoint", assign=True, allow_missing=True)
    reject_incomplete_load(model, what="FLUX.3 single-file checkpoint")
    sd.clear()
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
    """Loads a FLUX.3 transformer from a single bf16 file (BFL `model.safetensors`, or its ComfyUI repackage)."""

    def _load_model(self, config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_Flux3_Config):
            raise ValueError(f"Unexpected config type {type(config).__name__} for the FLUX.3 single-file loader.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError("A single-file FLUX.3 model holds only the transformer.")

        from safetensors.torch import load_file

        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(TorchDevice.choose_torch_device())
        model = flux3_transformer_from_native(load_file(Path(config.path)), model_dtype, self._ram_cache.make_room)
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
