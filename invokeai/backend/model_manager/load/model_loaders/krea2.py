# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Krea-2 model loading in InvokeAI."""

from pathlib import Path
from typing import Any, Optional

import accelerate
from transformers import AutoConfig, AutoTokenizer

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Krea2_Config, Main_GGUF_Krea2_Config
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
    Qwen3VLEncoder_Checkpoint_Config,
    Qwen3VLEncoder_Qwen3VLEncoder_Config,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.util.devices import TorchDevice


def _normalize_qwen3vl_rope_config(config: Any) -> Any:
    """Mirror Qwen3-VL rope_parameters into rope_scaling for Transformers compatibility."""
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        rope_params = getattr(text_config, "rope_parameters", None)
        if getattr(text_config, "rope_scaling", None) is None and rope_params is not None:
            text_config.rope_scaling = rope_params
    return config


def _strip_comfyui_prefix(sd: dict[str, Any]) -> dict[str, Any]:
    """Strip ComfyUI-style ``model.diffusion_model.`` / ``diffusion_model.`` key prefixes if present."""
    prefix_to_strip = None
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if any(isinstance(k, str) and k.startswith(prefix) for k in sd.keys()):
            prefix_to_strip = prefix
            break
    if not prefix_to_strip:
        return sd
    return {
        (k[len(prefix_to_strip) :] if isinstance(k, str) and k.startswith(prefix_to_strip) else k): v
        for k, v in sd.items()
    }


def _to_plain_tensor(value: Any) -> Any:
    """Dequantize a GGMLTensor to a plain tensor (needed before reshape); pass others through."""
    if hasattr(value, "get_dequantized_tensor"):
        return value.get_dequantized_tensor()
    return value


def _is_native_krea2_format(sd: dict[str, Any]) -> bool:
    """Detect the native/ComfyUI Krea-2 key naming (e.g. GGUF) vs. the diffusers naming."""
    return any(
        isinstance(k, str) and (k.startswith(("blocks.", "txtfusion.", "first.")) or ".mod.lin" in k) for k in sd
    )


def _dequantize_scaled_fp8(sd: dict[str, Any]) -> dict[str, Any]:
    """Dequantize ComfyUI 'scaled fp8' weights: ``dequant = weight.float() * weight_scale``.

    Each quantized layer stores an fp8 ``<name>.weight`` plus a (usually scalar) ``<name>.weight_scale``.
    Returns a new dict with the weights dequantized to float and the ``.weight_scale`` keys removed.
    No-op if there are no scale keys.
    """
    import torch

    scale_keys = [k for k in sd if isinstance(k, str) and k.endswith(".weight_scale")]
    if not scale_keys:
        return sd
    out = dict(sd)
    for scale_key in scale_keys:
        weight_key = scale_key.replace(".weight_scale", ".weight")
        if weight_key in out:
            weight = torch.as_tensor(_to_plain_tensor(out[weight_key])).float()
            scale = torch.as_tensor(_to_plain_tensor(out[scale_key])).float()
            out[weight_key] = weight * scale
        del out[scale_key]
    return out


def _convert_krea2_native_to_diffusers(sd: dict[str, Any]) -> dict[str, Any]:
    """Convert a native/ComfyUI-format Krea-2 state dict (e.g. GGUF) to diffusers Krea2Transformer2DModel keys.

    Top-level module renames::

        blocks.N.*           -> transformer_blocks.N.*
        txtfusion.*          -> text_fusion.*
        first.*              -> img_in.*
        tmlp.0/2.*           -> time_embed.linear_1/2.*
        tproj.1.*            -> time_mod_proj.*
        txtmlp.0/1/3.*       -> txt_in.norm / linear_1 / linear_2.*
        last.linear/norm/modulation -> final_layer.linear / norm.weight / scale_shift_table

    Within every transformer / text-fusion block::

        attn.wq/wk/wv/wo            -> attn.to_q/to_k/to_v/to_out.0
        attn.gate                   -> attn.to_gate
        attn.qknorm.qnorm/knorm.scale -> attn.norm_q/norm_k.weight
        mlp.gate/up/down            -> ff.gate/up/down
        prenorm/postnorm.scale      -> norm1/norm2.weight
        mod.lin (6*H,)              -> scale_shift_table (6, H)

    The original final-block ``last.down``/``last.up`` projections have no counterpart in the diffusers
    ``Krea2FinalLayer`` (a clean AdaLN + linear) and are dropped.
    """
    import torch

    new_sd: dict[str, Any] = {}
    for key, value in sd.items():
        if not isinstance(key, str):
            new_sd[key] = value
            continue
        # Drop original-only final-block projections (no diffusers equivalent).
        if key in ("last.down.weight", "last.up.weight"):
            continue

        k = key
        # Top-level module prefixes.
        if k.startswith("blocks."):
            k = "transformer_blocks." + k[len("blocks.") :]
        elif k.startswith("txtfusion."):
            k = "text_fusion." + k[len("txtfusion.") :]
        elif k.startswith("first."):
            k = "img_in." + k[len("first.") :]
        elif k.startswith("tmlp.0."):
            k = "time_embed.linear_1." + k[len("tmlp.0.") :]
        elif k.startswith("tmlp.2."):
            k = "time_embed.linear_2." + k[len("tmlp.2.") :]
        elif k.startswith("tproj.1."):
            k = "time_mod_proj." + k[len("tproj.1.") :]
        elif k == "txtmlp.0.scale":
            k = "txt_in.norm.weight"
        elif k.startswith("txtmlp.1."):
            k = "txt_in.linear_1." + k[len("txtmlp.1.") :]
        elif k.startswith("txtmlp.3."):
            k = "txt_in.linear_2." + k[len("txtmlp.3.") :]
        elif k == "last.linear.weight":
            k = "final_layer.linear.weight"
        elif k == "last.linear.bias":
            k = "final_layer.linear.bias"
        elif k == "last.norm.scale":
            k = "final_layer.norm.weight"
        elif k == "last.modulation.lin":
            k = "final_layer.scale_shift_table"

        # Within-block sub-module renames (apply to transformer_blocks.* and text_fusion.*).
        k = k.replace(".attn.wq.weight", ".attn.to_q.weight")
        k = k.replace(".attn.wk.weight", ".attn.to_k.weight")
        k = k.replace(".attn.wv.weight", ".attn.to_v.weight")
        k = k.replace(".attn.wo.weight", ".attn.to_out.0.weight")
        k = k.replace(".attn.gate.weight", ".attn.to_gate.weight")
        k = k.replace(".attn.qknorm.qnorm.scale", ".attn.norm_q.weight")
        k = k.replace(".attn.qknorm.knorm.scale", ".attn.norm_k.weight")
        k = k.replace(".mlp.gate.weight", ".ff.gate.weight")
        k = k.replace(".mlp.up.weight", ".ff.up.weight")
        k = k.replace(".mlp.down.weight", ".ff.down.weight")
        k = k.replace(".prenorm.scale", ".norm1.weight")
        k = k.replace(".postnorm.scale", ".norm2.weight")

        # Per-image-block modulation table: flat (6*H,) -> (6, H).
        if k.endswith(".mod.lin"):
            k = k[: -len(".mod.lin")] + ".scale_shift_table"
            value = torch.as_tensor(_to_plain_tensor(value)).reshape(6, -1)

        new_sd[k] = value
    return new_sd


# Default Krea2Transformer2DModel config (from the Krea-2-Turbo transformer/config.json). Used when
# loading a bare single-file checkpoint that has no accompanying config.json.
KREA2_TRANSFORMER_CONFIG = {
    "attention_head_dim": 128,
    "axes_dims_rope": [32, 48, 48],
    "in_channels": 64,
    "intermediate_size": 16384,
    "norm_eps": 1e-05,
    "num_attention_heads": 48,
    "num_key_value_heads": 12,
    "num_layers": 28,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
    "num_text_layers": 12,
    "rope_theta": 1000.0,
    "text_hidden_dim": 2560,
    "text_intermediate_size": 6912,
    "text_num_attention_heads": 20,
    "text_num_key_value_heads": 20,
    "timestep_embed_dim": 256,
}


@ModelLoaderRegistry.register(base=BaseModelType.Krea2, type=ModelType.Main, format=ModelFormat.Diffusers)
class Krea2DiffusersModel(GenericDiffusersLoader):
    """Class to load Krea-2 main models (Krea-2-Turbo) in diffusers format.

    Loads every submodel (transformer, vae, text_encoder, tokenizer, scheduler) from the diffusers
    pipeline folder via the class names declared in model_index.json. The transformer resolves to
    diffusers' ``Krea2Transformer2DModel`` (only available in diffusers main / >=0.39); the VAE to
    ``AutoencoderKLQwenImage`` and the text encoder to ``Qwen3VLModel``.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("CheckpointConfigBase is not implemented for the Krea-2 diffusers loader.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)

        # model_index.json declares the tokenizer as the slow `Qwen2Tokenizer`, which requires
        # vocab.json/merges.txt. Krea-2 ships only a fast tokenizer.json, so load via AutoTokenizer
        # (which resolves to Qwen2TokenizerFast from tokenizer.json).
        #
        # Krea-2's tokenizer_config.json stores `extra_special_tokens` as a list (the special tokens
        # are already baked into tokenizer.json as added tokens). Newer transformers expects a dict and
        # crashes on the list, so override it with an empty dict — the special tokens are still
        # recognized from tokenizer.json.
        if submodel_type is SubModelType.Tokenizer:
            return AutoTokenizer.from_pretrained(
                model_path / submodel_type.value, local_files_only=True, extra_special_tokens={}
            )

        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        # Krea-2 prefers bfloat16; use a safe dtype based on target device capabilities.
        target_device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        extra_kwargs: dict[str, Any] = {}
        if submodel_type is SubModelType.TextEncoder:
            # Krea-2's Qwen3-VL text_encoder config stores rope settings under `rope_parameters`, but the
            # installed transformers' Qwen3VL rotary embedding reads `rope_scaling` (None here) → crash.
            # Patch the config so rope_scaling mirrors rope_parameters before instantiating the model.
            te_config = _normalize_qwen3vl_rope_config(AutoConfig.from_pretrained(model_path, local_files_only=True))
            extra_kwargs["config"] = te_config

        try:
            result: AnyModel = load_class.from_pretrained(
                model_path,
                torch_dtype=dtype,
                variant=variant,
                **extra_kwargs,
            )
        except OSError as e:
            if variant and "no file named" in str(e):
                # try without the variant, just in case the user's preferences changed
                result = load_class.from_pretrained(model_path, torch_dtype=dtype, **extra_kwargs)
            else:
                raise e

        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
        return result


@ModelLoaderRegistry.register(base=BaseModelType.Krea2, type=ModelType.Main, format=ModelFormat.Checkpoint)
class Krea2CheckpointModel(ModelLoader):
    """Class to load Krea-2 transformer models from single-file checkpoints (safetensors).

    Handles plain bf16/fp16 checkpoints as well as ComfyUI 'scaled fp8' checkpoints (fp8 weight +
    ``.weight_scale``), and both the diffusers and native/ComfyUI key naming. Apply the fp8-storage
    setting to keep the (large) transformer fp8-resident; otherwise it loads in full precision.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are supported here.")

        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                f"Only Transformer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
            )
        return self._load_from_singlefile(config)

    def _load_from_singlefile(self, config: AnyModelConfig) -> AnyModel:
        from diffusers import Krea2Transformer2DModel
        from safetensors.torch import load_file

        if not isinstance(config, Main_Checkpoint_Krea2_Config):
            raise TypeError(f"Expected Main_Checkpoint_Krea2_Config, got {type(config).__name__}.")
        model_path = Path(config.path)

        sd = load_file(model_path)
        sd = _strip_comfyui_prefix(sd)
        # ComfyUI 'scaled fp8' checkpoints: fold the per-tensor weight_scale into the weights (→ float).
        sd = _dequantize_scaled_fp8(sd)
        # Native/ComfyUI key naming → diffusers Krea2Transformer2DModel keys.
        if _is_native_krea2_format(sd):
            sd = _convert_krea2_native_to_diffusers(sd)

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        with accelerate.init_empty_weights():
            model = Krea2Transformer2DModel(**KREA2_TRANSFORMER_CONFIG)

        new_sd_size = sum(ten.nelement() * model_dtype.itemsize for ten in sd.values())
        self._ram_cache.make_room(new_sd_size)
        for k in sd.keys():
            sd[k] = sd[k].to(model_dtype)

        model.load_state_dict(sd, assign=True, strict=False)
        _reject_incomplete_load(model, what="Krea-2 single-file checkpoint")
        # Honor the fp8-storage setting (re-quantizes the dequantized weights to fp8-resident on CUDA).
        model = self._apply_fp8_layerwise_casting(model, config, SubModelType.Transformer)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Krea2, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class Krea2GGUFCheckpointModel(ModelLoader):
    """Class to load GGUF-quantized Krea-2 transformer models (single-file).

    GGUF ships only the transformer; the VAE (Qwen-Image), Qwen3-VL encoder, tokenizer and scheduler
    are sourced separately by the Krea-2 model-loader invocation (mix-and-match, like Z-Image/FLUX).
    The GGML tensors stay quantized and are dequantized on-the-fly during inference.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are supported here.")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                f"Only Transformer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
            )
        return self._load_from_gguf(config)

    def _load_from_gguf(self, config: AnyModelConfig) -> AnyModel:
        from diffusers import Krea2Transformer2DModel

        if not isinstance(config, Main_GGUF_Krea2_Config):
            raise TypeError(f"Expected Main_GGUF_Krea2_Config, got {type(config).__name__}.")

        model_path = Path(config.path)
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # GGMLTensor wrappers (kept on CPU; dequantized on-the-fly by the cache during inference).
        sd = gguf_sd_loader(model_path, compute_dtype=compute_dtype)
        sd = _strip_comfyui_prefix(sd)
        # GGUF conversions use the native/ComfyUI compact key naming; remap to diffusers keys.
        if _is_native_krea2_format(sd):
            sd = _convert_krea2_native_to_diffusers(sd)

        with accelerate.init_empty_weights():
            model = Krea2Transformer2DModel(**KREA2_TRANSFORMER_CONFIG)

        model.load_state_dict(sd, assign=True, strict=False)
        # Reject GGUF layouts that don't fully populate the diffusers Krea2Transformer2DModel (city96/
        # ComfyUI GGUFs may use key names needing conversion). Failing here beats a confusing meta-tensor
        # crash mid-inference.
        _reject_incomplete_load(model, what="Krea-2 GGUF checkpoint")
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3VLEncoder, format=ModelFormat.Qwen3VLEncoder)
class Qwen3VLEncoderLoader(ModelLoader):
    """Class to load standalone Qwen3-VL text encoder models for Krea-2 (directory format)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from transformers import Qwen3VLModel

        if not isinstance(config, Qwen3VLEncoder_Qwen3VLEncoder_Config):
            raise ValueError("Only Qwen3VLEncoder_Qwen3VLEncoder_Config models are supported here.")

        model_path = Path(config.path)

        # Support both a full pipeline-style layout (text_encoder/ + tokenizer/) and a standalone
        # download where the encoder files live directly at the root.
        text_encoder_path = model_path / "text_encoder"
        tokenizer_path = model_path / "tokenizer"
        is_standalone = not text_encoder_path.exists() and (model_path / "config.json").exists()
        if is_standalone:
            text_encoder_path = model_path
            tokenizer_path = model_path

        match submodel_type:
            case SubModelType.Tokenizer:
                # extra_special_tokens={} works around Krea-2's list-format tokenizer_config (see
                # Krea2DiffusersModel); harmless for well-formed configs.
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, extra_special_tokens={})
            case SubModelType.TextEncoder:
                target_device = TorchDevice.choose_torch_device()
                model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
                te_config = _normalize_qwen3vl_rope_config(
                    AutoConfig.from_pretrained(text_encoder_path, local_files_only=True)
                )
                return Qwen3VLModel.from_pretrained(
                    text_encoder_path,
                    config=te_config,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )


def _remap_qwen3vl_singlefile_keys(sd: dict[str, Any]) -> dict[str, Any]:
    """Remap ComfyUI single-file Qwen3-VL keys to the transformers ``Qwen3VLModel`` layout.

    ComfyUI/native layout uses a single ``model.`` prefix for both towers; transformers splits them:
    ``model.visual.*`` -> ``visual.*`` and ``model.<rest>`` (layers/embed_tokens/norm) -> ``language_model.<rest>``.
    """
    out: dict[str, Any] = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            out[k] = v
            continue
        # Strip a leading "model." (some checkpoints prefix everything with it), then route by tower.
        key = k[len("model.") :] if k.startswith("model.") else k
        if key.startswith("visual.") or key.startswith("language_model."):
            # Already the transformers layout (e.g. "model.language_model.*" / "model.visual.*").
            out[key] = v
        else:
            # Bare language-model keys (layers.* / embed_tokens / norm) belong under language_model.
            out["language_model." + key] = v
    return out


def _reject_incomplete_load(model: Any, *, what: str) -> None:
    """Raise if a ``load_state_dict(strict=False)`` left required tensors on the meta device.

    ``strict=False`` is used to tolerate benign extra/renamed keys, but it also silently accepts a
    checkpoint that omits required weights — those tensors stay on the meta device and only fail much
    later during inference. Reject such loads here, naming the offending tensors, so an incomplete,
    misidentified, or differently-converted checkpoint fails at load time with an actionable message.

    Both parameters *and persistent buffers* are checked: ``accelerate.init_empty_weights()`` places
    buffers on the meta device too, so a native/GGUF checkpoint that omits a persistent buffer would
    slip past a parameters-only guard and fail mid-inference instead of at load time.
    """
    still_meta = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if getattr(tensor, "is_meta", False)
    ]
    if still_meta:
        raise RuntimeError(
            f"{what} is incomplete: {len(still_meta)} tensor(s) were not provided by the checkpoint "
            f"and remain uninitialized (meta device). First few: {still_meta[:8]}. The file is likely "
            "incomplete, misidentified, or uses a key layout that needs conversion."
        )


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3VLEncoder, format=ModelFormat.Checkpoint)
class Qwen3VLEncoderCheckpointLoader(ModelLoader):
    """Loads a single-file Qwen3-VL encoder checkpoint (e.g. ComfyUI ``qwen3vl_4b_bf16`` / ``_fp8_scaled``).

    The checkpoint bundles the language model + visual tower but no config/tokenizer; those are pulled
    from HuggingFace (``Qwen/Qwen3-VL-4B-Instruct``) with offline-cache fallback. ComfyUI 'scaled fp8'
    weights are dequantized to the compute dtype on load.
    """

    DEFAULT_HF_REPO = "Qwen/Qwen3-VL-4B-Instruct"

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Qwen3VLEncoder_Checkpoint_Config):
            raise ValueError("Only Qwen3VLEncoder_Checkpoint_Config models are supported here.")

        match submodel_type:
            case SubModelType.Tokenizer:
                return self._load_tokenizer()
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config)

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_tokenizer(self) -> AnyModel:
        # A partial offline cache (e.g. config present but vocab/merges missing) raises something other
        # than OSError (e.g. TypeError) deep in the slow-tokenizer path, so catch broadly and re-fetch.
        try:
            return AutoTokenizer.from_pretrained(self.DEFAULT_HF_REPO, local_files_only=True, extra_special_tokens={})
        except Exception:
            return AutoTokenizer.from_pretrained(self.DEFAULT_HF_REPO, extra_special_tokens={})

    def _load_hf_config(self) -> Any:
        try:
            te_config = AutoConfig.from_pretrained(self.DEFAULT_HF_REPO, local_files_only=True)
        except Exception:
            te_config = AutoConfig.from_pretrained(self.DEFAULT_HF_REPO)
        return _normalize_qwen3vl_rope_config(te_config)

    def _load_text_encoder(self, config: Qwen3VLEncoder_Checkpoint_Config) -> AnyModel:
        import torch
        from safetensors.torch import load_file
        from transformers import Qwen3VLModel

        model_path = Path(config.path)
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = load_file(str(model_path))
        # Detect an fp8 source (ComfyUI 'scaled fp8' weight_scale keys, or raw float8 weights) BEFORE
        # dequantizing. An fp8-on-disk encoder is kept fp8-resident with layerwise upcasting below, so
        # it occupies ~half the VRAM of the dequantized bf16 model (the whole point of shipping fp8).
        source_is_fp8 = any(isinstance(k, str) and k.endswith(".weight_scale") for k in sd) or any(
            getattr(t, "dtype", None) in (torch.float8_e4m3fn, torch.float8_e5m2) for t in sd.values()
        )
        # ComfyUI 'scaled fp8': fold weight_scale into the weights, then drop quantization metadata.
        sd = _dequantize_scaled_fp8(sd)
        for k in list(sd.keys()):
            if isinstance(k, str) and (k.endswith(".comfy_quant") or "scale_input" in k):
                del sd[k]
        sd = _remap_qwen3vl_singlefile_keys(sd)

        te_config = self._load_hf_config()
        with accelerate.init_empty_weights():
            model = Qwen3VLModel._from_config(te_config)

        new_sd_size = sum(ten.nelement() * model_dtype.itemsize for ten in sd.values())
        self._ram_cache.make_room(new_sd_size)
        for k in sd.keys():
            sd[k] = sd[k].to(model_dtype)

        model.load_state_dict(sd, assign=True, strict=False)
        _reject_incomplete_load(model, what="Qwen3-VL encoder checkpoint")

        # Keep an fp8 encoder running in fp8 (storage=float8_e4m3fn, per-layer upcast to the compute
        # dtype during forward) on CUDA. `_should_use_fp8` deliberately excludes text encoders (and the
        # config has no fp8_storage toggle), so apply the hook-based casting directly here. This roughly
        # halves the encoder's resident VRAM (~8.9GB bf16 -> ~4.4GB), which avoids partial-load thrashing
        # when it shares the GPU with a large transformer.
        if source_is_fp8 and self._torch_device.type == "cuda":
            self._apply_fp8_to_nn_module(model, storage_dtype=torch.float8_e4m3fn, compute_dtype=model_dtype)
            self._logger.info(
                f"FP8 layerwise casting enabled for Qwen3-VL encoder '{config.name}' "
                f"(storage=float8_e4m3fn, compute={model_dtype})."
            )

        return model
