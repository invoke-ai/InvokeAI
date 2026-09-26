# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Z-Image model loading in InvokeAI."""

from pathlib import Path
from typing import Any, Optional

import accelerate
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.controlnet import ControlNet_Checkpoint_ZImage_Config
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_ZImage_Config,
    Main_GGUF_ZImage_Config,
    Main_SDNQ_Diffusers_ZImage_Config,
    Main_SDNQ_ZImage_Config,
)
from invokeai.backend.model_manager.configs.qwen3_encoder import (
    Qwen3Encoder_Checkpoint_Config,
    Qwen3Encoder_GGUF_Config,
    Qwen3Encoder_Qwen3Encoder_Config,
    Qwen3Encoder_SDNQ_Config,
    Qwen3Encoder_SDNQ_Folder_Config,
)
from invokeai.backend.model_manager.load.load_default import (
    ModelLoader,
    _model_declared_skip_patterns,
    resolve_submodel_path,
)
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.load.quantized_embedding import materialize_quantized_embedding
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.util.llamacpp_keys import (
    convert_llamacpp_decoder_keys,
    is_llamacpp_decoder_state_dict,
)
from invokeai.backend.quantization.fp8_scaled import (
    QKV_SPLIT_SIDECHANNEL_SUFFIXES,
    attach_fp8_scales,
    cast_state_dict,
    dequantize_fp8_scaled,
    expand_weight_scale,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    full_precision_hints_respected,
    is_scale_metadata_key,
    iter_weight_scale_pairs,
    parse_quantization_metadata,
    read_safetensors_metadata,
    reject_quantized_side_channel,
    reject_undecoded_mx_scale,
    split_fp8_scaled_layers,
    split_qkv_sidechannel,
    strip_layer_path_prefix,
    warn_on_unattached_scales,
)
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.quantization.int8_convrot import (
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    reject_unmarked_int8_weights,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.quantization.nvfp4 import (
    NVFP4Payload,
    install_nvfp4_layers,
    pop_nvfp4_layers,
    predict_nvfp4_install_size,
    split_nvfp4_rows,
)
from invokeai.backend.quantization.sdnq.detection import is_sdnq_folder
from invokeai.backend.quantization.sdnq.loaders import raise_on_incomplete_sdnq_load, sdnq_sd_loader
from invokeai.backend.qwen3.qwen3_tokenizer import load_bundled_qwen3_tokenizer
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, log_unexpected_keys


def _remap_z_image_layer_paths(layer_names: Any) -> dict[str, list[str]]:
    """Map native Z-Image layer paths to their diffusers equivalents.

    A probe, where FLUX.2's equivalent was replaced by the conversion's own record. The difference is
    that this converter *raises* on a fused ``qkv`` whose rows are not divisible by three rather than
    leaving the key alone, so a probe and the conversion cannot disagree about it -- and the second
    caller below works on payloads popped out *before* the conversion, which no record of that
    conversion could cover.

    ``_quantization_metadata`` names its layers in the checkpoint's own scheme, but the scales are
    extracted after the state dict has been renamed. Rather than restating the rename rules — which
    would drift — each name is pushed through the real converter as a lone ``<name>.weight`` entry
    and the resulting keys are read back. A fused ``qkv`` maps to *three* diffusers layers, so the
    mapping is one-to-many.
    """
    mapping: dict[str, list[str]] = {}
    for name in layer_names:
        if not isinstance(name, str):
            continue
        try:
            # 3 rows so the qkv split is well-defined; the values themselves are never read.
            converted = _convert_z_image_gguf_to_diffusers({f"{name}.weight": torch.empty(3, 1)})
        except Exception:
            continue
        targets = [k[: -len(".weight")] for k in converted if isinstance(k, str) and k.endswith(".weight")]
        if targets:
            mapping[name] = targets
    return mapping


def _remap_nvfp4_payloads(payloads: dict[str, NVFP4Payload]) -> dict[str, NVFP4Payload]:
    """Move packed nvfp4 layers to their diffusers paths, splitting a fused QKV's tensors the way the converter
    splits its weight: into equal thirds by rows, which the block scales only survive on whole tile rows."""
    path_map = _remap_z_image_layer_paths(payloads.keys())
    remapped: dict[str, NVFP4Payload] = {}
    for name, payload in payloads.items():
        targets = path_map.get(name, [name])
        remapped.update(zip(targets, split_nvfp4_rows(name, payload, len(targets)), strict=True))
    return remapped


def _convert_z_image_gguf_to_diffusers(sd: dict[str, Any]) -> dict[str, Any]:
    """Convert Z-Image GGUF state dict keys to diffusers format.

    The GGUF format uses original model keys that differ from diffusers:
    - qkv.weight (fused) -> to_q.weight, to_k.weight, to_v.weight (split)
    - out.weight -> to_out.0.weight
    - q_norm.weight -> norm_q.weight
    - k_norm.weight -> norm_k.weight
    - x_embedder.* -> all_x_embedder.2-1.*
    - final_layer.* -> all_final_layer.2-1.*
    - norm_final.* -> skipped (diffusers uses non-learnable LayerNorm)
    - x_pad_token, cap_pad_token: [dim] -> [1, dim] (diffusers expects batch dimension)
    """
    new_sd: dict[str, Any] = {}

    for key, value in sd.items():
        if not isinstance(key, str):
            new_sd[key] = value
            continue

        # Handle padding tokens: GGUF has shape [dim], diffusers expects [1, dim]
        if key in ("x_pad_token", "cap_pad_token"):
            if hasattr(value, "shape") and len(value.shape) == 1:
                # GGMLTensor doesn't support unsqueeze, so dequantize first if needed
                if hasattr(value, "get_dequantized_tensor"):
                    value = value.get_dequantized_tensor()
                # Use reshape instead of unsqueeze for better compatibility
                value = torch.as_tensor(value).reshape(1, -1)
            new_sd[key] = value
            continue

        # Handle x_embedder -> all_x_embedder.2-1
        if key.startswith("x_embedder."):
            suffix = key[len("x_embedder.") :]
            new_key = f"all_x_embedder.2-1.{suffix}"
            new_sd[new_key] = value
            continue

        # Handle final_layer -> all_final_layer.2-1
        if key.startswith("final_layer."):
            suffix = key[len("final_layer.") :]
            new_key = f"all_final_layer.2-1.{suffix}"
            new_sd[new_key] = value
            continue

        # Skip norm_final keys - the diffusers model uses LayerNorm with elementwise_affine=False
        # (no learnable weight/bias), but some checkpoints (e.g., FP8) include these as all-zeros
        if key.startswith("norm_final."):
            continue

        # Handle fused QKV weights - need to split
        if ".attention.qkv." in key:
            # Get the layer prefix and suffix
            prefix = key.rsplit(".attention.qkv.", 1)[0]
            suffix = key.rsplit(".attention.qkv.", 1)[1]  # "weight" or "bias"

            if suffix not in ("weight", "bias"):
                # Quantization side-channel for the fused weight. It has to travel with the split,
                # or the recovered scale is keyed on `...attention.qkv`, a module path that no
                # longer exists — `attach_fp8_scales` then finds nothing and the three split
                # weights stay quantized but *unscaled*, i.e. off by 1/weight_scale. The same
                # applies to a `comfy_quant` marker, which is why that suffix is in the list too:
                # without it `extract_int8_convrot_markers` would find no marker for the split
                # weights and reject the checkpoint as having orphaned int8 tensors.
                if suffix in QKV_SPLIT_SIDECHANNEL_SUFFIXES:
                    for name, part in zip(("to_q", "to_k", "to_v"), split_qkv_sidechannel(key, value), strict=True):
                        new_sd[f"{prefix}.attention.{name}.{suffix}"] = part
                    continue
                new_sd[key] = value
                continue

            # Split the fused QKV tensor into Q, K, V
            tensor = value
            if hasattr(tensor, "shape"):
                if tensor.shape[0] % 3 != 0:
                    raise ValueError(
                        f"Cannot split QKV tensor '{key}': first dimension ({tensor.shape[0]}) "
                        "is not divisible by 3. The model file may be corrupted or incompatible."
                    )
                dim = tensor.shape[0] // 3
                q = tensor[:dim]
                k = tensor[dim : 2 * dim]
                v = tensor[2 * dim :]

                new_sd[f"{prefix}.attention.to_q.{suffix}"] = q
                new_sd[f"{prefix}.attention.to_k.{suffix}"] = k
                new_sd[f"{prefix}.attention.to_v.{suffix}"] = v
            continue

        # Handle attention key renaming
        if ".attention." in key:
            new_key = key.replace(".q_norm.", ".norm_q.")
            new_key = new_key.replace(".k_norm.", ".norm_k.")
            new_key = new_key.replace(".attention.out.", ".attention.to_out.0.")
            new_sd[new_key] = value
            continue

        # For all other keys, just copy as-is
        new_sd[key] = value

    return new_sd


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.Diffusers)
class ZImageDiffusersModel(GenericDiffusersLoader):
    """Class to load Z-Image main models (Z-Image-Turbo, Z-Image-Base, Z-Image-Edit)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("CheckpointConfigBase is not implemented for Z-Image models.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        submodel_path = resolve_submodel_path(config, submodel_type, model_path / submodel_type.value)

        # Check if submodel folder has SDNQ quantization - if so, use SDNQ loader
        if self._is_sdnq_folder(submodel_path):
            if submodel_type == SubModelType.TextEncoder:
                return self._load_sdnq_text_encoder(submodel_path)
            elif submodel_type == SubModelType.Transformer:
                return self._load_sdnq_transformer(submodel_path)

        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None

        # Z-Image prefers bfloat16, but use safe dtype based on target device capabilities.
        target_device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
        try:
            result: AnyModel = load_class.from_pretrained(
                submodel_path,
                torch_dtype=dtype,
                variant=variant,
            )
        except OSError as e:
            if variant and "no file named" in str(
                e
            ):  # try without the variant, just in case user's preferences changed
                result = load_class.from_pretrained(submodel_path, torch_dtype=dtype)
            else:
                raise e

        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
        return result

    def _is_sdnq_folder(self, folder_path: Path) -> bool:
        """Check if a folder contains SDNQ-quantized model weights.

        Shared detector, so this dispatch agrees with identification about markerless exports.
        """
        return is_sdnq_folder(folder_path)

    def _load_sdnq_text_encoder(self, text_encoder_path: Path) -> AnyModel:
        """Load SDNQ-quantized text encoder from folder."""
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from invokeai.backend.quantization.sdnq.loaders import sdnq_sd_loader
        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        logger.info(f"Loading SDNQ-quantized text encoder from {text_encoder_path}")

        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the SDNQ state dict
        sd = sdnq_sd_loader(text_encoder_path, compute_dtype=compute_dtype)

        # Determine Qwen model configuration from state dict
        layer_count = 0
        for key in sd.keys():
            if isinstance(key, str) and key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_idx = int(parts[2])
                        layer_count = max(layer_count, layer_idx + 1)
                    except ValueError:
                        pass

        # Get hidden size from embed_tokens weight shape
        embed_weight = sd.get("model.embed_tokens.weight")
        if embed_weight is None:
            raise ValueError("Could not find model.embed_tokens.weight in state dict")

        embed_shape = embed_weight.shape if hasattr(embed_weight, "shape") else embed_weight.tensor_shape
        hidden_size = embed_shape[1]
        vocab_size = embed_shape[0]

        # Detect attention configuration from layer 0 weights
        q_proj_weight = sd.get("model.layers.0.self_attn.q_proj.weight")
        k_proj_weight = sd.get("model.layers.0.self_attn.k_proj.weight")
        gate_proj_weight = sd.get("model.layers.0.mlp.gate_proj.weight")

        if q_proj_weight is None or k_proj_weight is None or gate_proj_weight is None:
            raise ValueError("Could not find attention/mlp weights in state dict")

        q_shape = q_proj_weight.shape if hasattr(q_proj_weight, "shape") else q_proj_weight.tensor_shape
        k_shape = k_proj_weight.shape if hasattr(k_proj_weight, "shape") else k_proj_weight.tensor_shape
        gate_shape = gate_proj_weight.shape if hasattr(gate_proj_weight, "shape") else gate_proj_weight.tensor_shape

        head_dim = 128
        num_attention_heads = q_shape[0] // head_dim
        num_kv_heads = k_shape[0] // head_dim
        intermediate_size = gate_shape[0]

        logger.info(
            f"Qwen3 SDNQ config: layers={layer_count}, hidden={hidden_size}, "
            f"heads={num_attention_heads}, kv_heads={num_kv_heads}"
        )

        qwen_config = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layer_count,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=40960,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            rope_theta=1000000.0,
            use_sliding_window=False,
            attention_bias=False,
            attention_dropout=0.0,
            torch_dtype=compute_dtype,
        )

        with accelerate.init_empty_weights():
            model = Qwen3ForCausalLM(qwen_config)

        # lm_head is tied to embed_tokens (re-shared below), so it is expected to be missing.
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        raise_on_incomplete_sdnq_load(
            "SDNQ Z-Image Qwen3 text encoder", missing, unexpected, allowed_missing={"lm_head.weight"}
        )

        if materialize_quantized_embedding(model.model.embed_tokens, ram_cache=self._ram_cache):
            logger.info("Dequantized embed_tokens weight for embedding lookups")

        # Handle tied weights
        if qwen_config.tie_word_embeddings:
            if model.lm_head.weight.is_meta:
                model.lm_head.weight = model.model.embed_tokens.weight
            else:
                model.tie_weights()

        # Re-initialize meta tensor buffers
        for name, buffer in list(model.named_buffers()):
            if buffer.is_meta:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    buffer_name = parts[1]
                else:
                    parent = model
                    buffer_name = name

                if buffer_name == "inv_freq":
                    base = qwen_config.rope_theta
                    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    parent.register_buffer(buffer_name, inv_freq.to(dtype=compute_dtype), persistent=False)

        # Fail fast if any required parameter is still on the meta device (e.g. a partial export).
        # This mirrors the standalone SDNQ Qwen3 encoder loader's final guard.
        meta_params = [name for name, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(f"SDNQ Z-Image Qwen3 text encoder has parameters left on meta: {meta_params}")

        return model

    def _load_sdnq_transformer(self, transformer_path: Path) -> AnyModel:
        """Load SDNQ-quantized transformer from folder."""
        from diffusers import ZImageTransformer2DModel

        from invokeai.backend.quantization.sdnq.loaders import sdnq_sd_loader
        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        logger.info(f"Loading SDNQ-quantized transformer from {transformer_path}")

        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the SDNQ state dict
        sd = sdnq_sd_loader(transformer_path, compute_dtype=compute_dtype)

        # Check if conversion is needed (original format vs diffusers format)
        needs_conversion = any(k.startswith("x_embedder.") for k in sd.keys() if isinstance(k, str))
        if needs_conversion:
            sd = _convert_z_image_gguf_to_diffusers(sd)

        # Create an empty model with the default Z-Image config
        with accelerate.init_empty_weights():
            model = ZImageTransformer2DModel(
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                in_channels=16,
                dim=3840,
                n_layers=30,
                n_refiner_layers=2,
                n_heads=30,
                n_kv_heads=30,
                norm_eps=1e-05,
                qk_norm=True,
                cap_feat_dim=2560,
                rope_theta=256.0,
                t_scale=1000.0,
                axes_dims=[32, 48, 48],
                axes_lens=[1024, 512, 512],
            )

        load_state_dict_ignoring_extras(model, sd, source="Z-Image transformer", assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.Checkpoint)
class ZImageCheckpointModel(ModelLoader):
    """Class to load Z-Image transformer models from single-file checkpoints (safetensors, etc)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from diffusers import ZImageTransformer2DModel
        from safetensors.torch import load_file

        if not isinstance(config, Main_Checkpoint_ZImage_Config):
            raise TypeError(
                f"Expected Main_Checkpoint_ZImage_Config, got {type(config).__name__}. "
                "Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Load the state dict from safetensors/checkpoint file
        sd = load_file(model_path)

        # Some Z-Image checkpoint files have keys prefixed with "diffusion_model." or
        # "model.diffusion_model." (ComfyUI-style format). Check if we need to strip this prefix.
        sd = CheckpointPrefix.detect(sd).strip(sd)

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Per-layer hints from the safetensors header and/or the per-tensor `.comfy_quant` markers:
        # `full_precision_matrix_mult` for scaled fp8, and for nvfp4 the evidence that a layer follows
        # ComfyUI's conventions. The header names layers in the checkpoint's own scheme, so it is remapped
        # below; the markers ride along through the key conversion instead. The names in the header still
        # carry the checkpoint prefix stripped off `sd` above.
        header_hints = strip_layer_path_prefix(
            parse_quantization_metadata(read_safetensors_metadata(model_path, self._logger))
        )

        # Out of the state dict before anything below reads the quantization side channel: the scaled-fp8
        # extraction pops every `weight_scale` and drops the ones whose weight is not float8, nvfp4's block
        # scales included, and the casts would widen the packed payload.
        nvfp4_payloads = pop_nvfp4_layers(sd, header_layers=header_hints)

        # Check if the state dict is in original format (not diffusers format)
        # Original format has keys like "x_embedder.weight" instead of "all_x_embedder.2-1.weight"
        needs_conversion = any(k.startswith("x_embedder.") for k in sd.keys() if isinstance(k, str))

        if needs_conversion:
            # Convert from original format to diffusers format
            sd = _convert_z_image_gguf_to_diffusers(sd)
            path_map = _remap_z_image_layer_paths(header_hints.keys())
            header_hints = {
                target: hints for name, hints in header_hints.items() for target in path_map.get(name, [name])
            }
            nvfp4_payloads = _remap_nvfp4_payloads(nvfp4_payloads)

        # Create an empty model with the default Z-Image config
        # Z-Image-Turbo uses these default parameters from diffusers
        with accelerate.init_empty_weights():
            model = ZImageTransformer2DModel(
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                in_channels=16,
                dim=3840,
                n_layers=30,
                n_refiner_layers=2,
                n_heads=30,
                n_kv_heads=30,
                norm_eps=1e-05,
                qk_norm=True,
                cap_feat_dim=2560,
                rope_theta=256.0,
                t_scale=1000.0,
                axes_dims=[32, 48, 48],
                axes_lens=[1024, 512, 512],
            )

        # Filter out keys that don't belong to the ZImageTransformer2DModel.
        # Merged checkpoints (e.g. LoRA-baked models) may bundle text encoder weights
        # (text_encoders.*) or other non-transformer keys alongside the transformer weights.
        # This runs *before* the scales are extracted so a bundled encoder's own scale keys are
        # dropped here rather than being recovered as transformer layers that resolve to nothing.
        valid_prefixes = (
            "all_x_embedder.",
            "all_final_layer.",
            "layers.",
            "noise_refiner.",
            "context_refiner.",
            "t_embedder.",
            "cap_embedder.",
            "rope_embedder.",
        )
        valid_exact = {"x_pad_token", "cap_pad_token"}
        keys_to_remove = [k for k in sd.keys() if not (k.startswith(valid_prefixes) or k in valid_exact)]
        for k in keys_to_remove:
            del sd[k]
        # A bundled encoder's nvfp4 layers go the same way.
        nvfp4_payloads = {path: payload for path, payload in nvfp4_payloads.items() if path.startswith(valid_prefixes)}

        # Honor the model's own precision-sensitive list on every path below. Z-Image declares
        # ["t_embedder", "cap_embedder"] because `ZImageTimestepEmbedder.forward` reads
        # `self.mlp[0].weight.dtype` to pick the dtype it casts its activations to, and a quantized
        # weight there breaks that branch whichever scheme it comes from: an fp8 weight turns the
        # activations fp8 and the forward dies in `x.abs()`; an `Int8ConvrotLinear` or `NVFP4Linear`
        # reports an integer dtype, the forward falls through to a `compute_dtype` attribute these
        # modules do not have, and the timestep branch silently runs in float32 into a bf16 model.
        skip_patterns = _model_declared_skip_patterns(model)
        # What the nvfp4 layers will occupy, packed or decoded. Both branches reserve it together with the
        # rest of the state dict: a reservation makes that much room, it does not add to an earlier one.
        nvfp4_bytes = predict_nvfp4_install_size(model, nvfp4_payloads, model_dtype, skip_patterns)

        # Two ComfyUI side-channel formats reach this loader, and a checkpoint carries one or the
        # other: `comfy_quant` names its format per layer, and `int8_tensorwise` never appears in a
        # file that also ships fp8 weight scales. Deciding once, up front, keeps the two casts from
        # having to understand each other -- `cast_unquantized` treats int8 payloads as opaque,
        # `cast_state_dict` reasons about fp8 matmul eligibility, and neither is correct for the
        # other's tensors.
        int8_markers = extract_int8_convrot_markers(sd)

        # Outside the branch on purpose -- see the helper, which explains why.
        reject_unmarked_int8_weights(sd, int8_markers, "Z-Image")

        if int8_markers:
            # Markers are read *after* the key conversion above, which carries them (and their
            # scales) through the fused-QKV split onto the module names the model actually has --
            # so no re-keying is needed here.
            #
            # Filtered in place rather than rebound: the `sd.clear()` below has to reach the same
            # dict the checkpoint was read into, or the originals stay alive through it and peak
            # RAM overshoots the `make_room()` reservation (see
            # test_state_dict_is_released_before_the_fp8_cast).
            kept_sd = drop_unconsumed_quantization_sidecars(sd)
            sd.clear()
            sd.update(kept_sd)
            del kept_sd

            # The nvfp4 layers taken out above are held beside the int8 ones, so the single
            # reservation has to cover both.
            quantized = install_int8_convrot_layers(
                model,
                sd,
                int8_markers,
                model_dtype,
                architecture="Z-Image",
                reserve=self._ram_cache.make_room,
                skip_patterns=skip_patterns,
                extra_reserved_bytes=nvfp4_bytes,
            )
            # What did not stay int8 was widened to the compute dtype, so a load that kept far fewer
            # layers than the file marked is the explanation for a resident size twice what the file
            # suggests. Every other int8 loader reports this; this one did not.
            self._logger.info(
                f"Z-Image: kept {len(quantized)} of {len(int8_markers)} layer(s) in int8 "
                "(int8_tensorwise checkpoint, dequantized per forward)"
            )
            # The fp8 reporting below is keyed on these; an int8 checkpoint keeps neither.
            fp8_layers: dict[str, Any] = {}
            kept = 0
        else:
            # ComfyUI 'scaled fp8' (fp8 weight + .weight_scale/.scale_weight). Until now the loader
            # deleted those scales and cast the weight — silently producing a weight off by
            # 1/weight_scale — and had no way to tell such a checkpoint from a raw fp8 one.
            layer_hints = {**extract_comfy_quant_hints(sd), **header_hints}
            fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)

            # Handle memory management and dtype conversion. Casting fp8 weights here would discard
            # the VRAM saving before the model is even built -- and the tensor cores too, where the
            # fp8 matmul is what kept them. FP8 Storage counts as a consumer alongside the matmul:
            # the checkpoint's own scale is exact, where a layerwise cast of a folded weight has none.
            #
            # The fold itself stays below the reservation rather than moving up here: the prediction
            # takes `scaled_layers=fp8_layers` and would see an empty mapping, charging the layers the
            # split still widens 1 byte/element instead of 2.
            keep_fp8 = self._keep_fp8_weights(config, SubModelType.Transformer)

            # Reserve before anything below widens a weight, not after: without fp8 compute the fold right
            # after this widens every scaled layer, and `split_fp8_scaled_layers` dequantizes its unusable
            # subset through fp32 -- reserving afterwards lets either peak land on an unreserved cache.
            # Without fp8 compute the prediction charges every float at `model_dtype`, folded yet or not.
            # With it, `scaled_layers` is what keeps the prediction honest: the split also widens layers
            # whose scale layout `scaled_mm` cannot apply, and without the mapping the prediction would
            # charge those 1 byte/element and arrive at 2.
            reserve_for_load(
                self._ram_cache.make_room,
                sd,
                model_dtype,
                keep_fp8=keep_fp8,
                model=model,
                skip_patterns=skip_patterns,
                fp8_layers=fp8_layers,
                nvfp4_payloads=nvfp4_payloads,
            )

            if fp8_layers and not keep_fp8:
                # Legacy behavior, but now with the scale actually applied: fold it into the weight.
                dequantize_fp8_scaled(sd, fp8_layers, model_dtype)
                fp8_layers = {}

            # Scaled layers that the cast would dequantize anyway are folded here, scale applied, so
            # `cast_state_dict` never strips a scale it cannot put back.

            fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, model_dtype, model=model, skip_patterns=skip_patterns)
            kept = cast_state_dict(
                sd,
                model_dtype,
                keep_fp8=keep_fp8,
                model=model,
                skip_patterns=skip_patterns,
            )

        if nvfp4_payloads:
            packed = install_nvfp4_layers(model, sd, nvfp4_payloads, model_dtype, skip_patterns)
            decoded = len(nvfp4_payloads) - packed
            self._logger.info(
                f"Z-Image: kept {packed} nvfp4 layer(s) packed"
                + (
                    f" and decoded {decoded} to {model_dtype} (precision-sensitive or not a Linear)."
                    if decoded
                    else "."
                )
            )

        load_state_dict_ignoring_extras(model, sd, source="Z-Image transformer checkpoint", assign=True)
        # `assign=True` aliases every param to its `sd` tensor, so the dict keeps the whole model
        # alive a second time. The FP8 cast below allocates the fp8 copy per param while the
        # `model_dtype` original is still reachable through `sd`, pushing peak RAM to ~1.5x what
        # `make_room()` reserved above (~17.4GB actual vs ~11.5GB reserved for Z-Image). Dropping
        # the dict's references lets each original free as soon as its param is cast.
        sd.clear()

        if fp8_layers:
            attached = attach_fp8_scales(model, fp8_layers)
            self._logger.info(
                f"Z-Image: kept {attached} layer(s) in fp8 (scaled fp8 checkpoint, kept for {self._fp8_kept_reason()})"
            )
            warn_on_unattached_scales(self._logger, "Z-Image", attached, fp8_layers)
            marked = sum(1 for layer in fp8_layers.values() if layer.full_precision_matmul)
            if marked and full_precision_hints_respected():
                self._logger.info(
                    f"Z-Image: {marked} of {len(fp8_layers)} layer(s) are marked full_precision_matrix_mult "
                    "and will dequantize per forward. Set fp8_compute_full_precision_hints=false to run "
                    "them on the fp8 tensor cores instead."
                )
        elif kept:
            self._logger.info(
                f"Z-Image: kept {kept} raw fp8 weight(s) quantized (no weight_scale in the checkpoint); "
                "they will run on the fp8 tensor cores with unit scaling."
            )

        # FP8 *storage* on top. When nothing was kept quantized above, every param is uniform
        # `model_dtype` here, so the layerwise cast has one unambiguous compute dtype to restore to.
        # When weights *were* kept fp8, `_apply_fp8_layerwise_casting` bails out on its own (and
        # says so in the log): its hooks would restore the compute dtype before every forward, which
        # disables the fp8 matmul where there is one and drops the `weight_scale` where there is
        # not -- and saves no VRAM either way.
        model = self._apply_fp8_layerwise_casting(model, config, SubModelType.Transformer)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class ZImageGGUFCheckpointModel(ModelLoader):
    """Class to load GGUF-quantized Z-Image transformer models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from diffusers import ZImageTransformer2DModel

        if not isinstance(config, Main_GGUF_ZImage_Config):
            raise TypeError(
                f"Expected Main_GGUF_ZImage_Config, got {type(config).__name__}. Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the GGUF state dict
        sd = gguf_sd_loader(model_path, compute_dtype=compute_dtype)

        # Some Z-Image GGUF models have keys prefixed with "diffusion_model." or
        # "model.diffusion_model." (ComfyUI-style format). Check if we need to strip this prefix.
        sd = CheckpointPrefix.detect(sd).strip(sd)

        # Convert GGUF format keys to diffusers format
        sd = _convert_z_image_gguf_to_diffusers(sd)

        # Create an empty model with the default Z-Image config
        # Z-Image-Turbo uses these default parameters from diffusers
        with accelerate.init_empty_weights():
            model = ZImageTransformer2DModel(
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                in_channels=16,
                dim=3840,
                n_layers=30,
                n_refiner_layers=2,
                n_heads=30,
                n_kv_heads=30,
                norm_eps=1e-05,
                qk_norm=True,
                cap_feat_dim=2560,
                rope_theta=256.0,
                t_scale=1000.0,
                axes_dims=[32, 48, 48],
                axes_lens=[1024, 512, 512],
            )

        load_state_dict_ignoring_extras(model, sd, source="Z-Image GGUF transformer checkpoint", assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.SDNQQuantized)
class ZImageSDNQCheckpointModel(ModelLoader):
    """Class to load SDNQ-quantized Z-Image transformer models.

    Handles both single-file SDNQ checkpoints (``Main_SDNQ_ZImage_Config``) and full
    diffusers-pipeline folders (``Main_SDNQ_Diffusers_ZImage_Config``), where the
    quantized weights live under ``transformer/`` alongside a ``config.json`` that
    describes the architecture.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, (Main_SDNQ_ZImage_Config, Main_SDNQ_Diffusers_ZImage_Config)):
            raise ValueError(
                "Only Main_SDNQ_ZImage_Config or Main_SDNQ_Diffusers_ZImage_Config models are supported here."
            )

        # Single-file SDNQ checkpoints only carry the transformer.
        if isinstance(config, Main_SDNQ_ZImage_Config):
            if submodel_type == SubModelType.Transformer:
                return self._load_from_singlefile(config)
            raise ValueError(
                f"Single-file SDNQ Z-Image checkpoints only provide the Transformer submodel. "
                f"Received: {submodel_type.value if submodel_type else 'None'}"
            )

        # Full ZImagePipeline folder — dispatch each submodel out of its own subfolder so the
        # model can be used as a 'Qwen3 & VAE source model' for other Z-Image runs.
        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_diffusers_folder(config)
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config)
            case SubModelType.Tokenizer:
                return self._load_tokenizer(config)
            case SubModelType.VAE:
                return self._load_vae(config)

        raise ValueError(
            f"Unsupported submodel type for SDNQ ZImagePipeline: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_text_encoder(self, config: Main_SDNQ_Diffusers_ZImage_Config) -> AnyModel:
        from transformers import AutoConfig, Qwen3ForCausalLM

        te_dir = resolve_submodel_path(config, SubModelType.TextEncoder, Path(config.path) / "text_encoder")
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        te_config = AutoConfig.from_pretrained(te_dir, local_files_only=True)
        with accelerate.init_empty_weights():
            model = Qwen3ForCausalLM(te_config)

        sd = sdnq_sd_loader(te_dir, compute_dtype=compute_dtype)
        # Qwen3ForCausalLM may share lm_head.weight with model.embed_tokens.weight; missing keys
        # for that tie are expected and handled by re-sharing post-load.
        missing = load_state_dict_ignoring_extras(
            model, sd, source="SDNQ Qwen3 text encoder", assign=True, allowed_missing={"lm_head.weight"}
        )
        if missing == ["lm_head.weight"]:
            model.lm_head.weight = model.model.embed_tokens.weight
        return model

    def _load_tokenizer(self, config: Main_SDNQ_Diffusers_ZImage_Config) -> AnyModel:
        tok_dir = resolve_submodel_path(config, SubModelType.Tokenizer, Path(config.path) / "tokenizer")
        return AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)

    def _load_vae(self, config: Main_SDNQ_Diffusers_ZImage_Config) -> AnyModel:
        from diffusers import AutoencoderKL

        vae_dir = resolve_submodel_path(config, SubModelType.VAE, Path(config.path) / "vae")
        return AutoencoderKL.from_pretrained(vae_dir, local_files_only=True)

    def _load_from_singlefile(
        self,
        config: Main_SDNQ_ZImage_Config,
    ) -> AnyModel:
        from diffusers import ZImageTransformer2DModel

        model_path = Path(config.path)

        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = sdnq_sd_loader(model_path, compute_dtype=compute_dtype)

        # Some Z-Image SDNQ models may have keys prefixed with "diffusion_model." or
        # "model.diffusion_model." (ComfyUI-style format). Check if we need to strip this prefix.
        sd = CheckpointPrefix.detect(sd).strip(sd)

        # Check if conversion is needed (original format vs diffusers format)
        needs_conversion = any(k.startswith("x_embedder.") for k in sd.keys() if isinstance(k, str))
        if needs_conversion:
            sd = _convert_z_image_gguf_to_diffusers(sd)

        # Create an empty model with the default Z-Image config
        with accelerate.init_empty_weights():
            model = ZImageTransformer2DModel(
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                in_channels=16,
                dim=3840,
                n_layers=30,
                n_refiner_layers=2,
                n_heads=30,
                n_kv_heads=30,
                norm_eps=1e-05,
                qk_norm=True,
                cap_feat_dim=2560,
                rope_theta=256.0,
                t_scale=1000.0,
                axes_dims=[32, 48, 48],
                axes_lens=[1024, 512, 512],
            )

        load_state_dict_ignoring_extras(model, sd, source="SDNQ Z-Image transformer checkpoint", assign=True)
        return model

    def _load_from_diffusers_folder(
        self,
        config: Main_SDNQ_Diffusers_ZImage_Config,
    ) -> AnyModel:
        from diffusers import ZImageTransformer2DModel

        # When ZImagePipeline is registered with submodels, the transformer submodel's path points
        # into transformer/ directly. The pipeline-level Main config has its own path at the root.
        # Either way, locate the transformer/config.json + safetensors.
        model_path = Path(config.path)
        transformer_path = resolve_submodel_path(
            config,
            SubModelType.Transformer,
            model_path / "transformer" if (model_path / "transformer").is_dir() else model_path,
        )

        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Build the empty model from the on-disk architecture description so we honor non-default
        # axes_lens / dim / etc. that the single-file path hardcodes.
        with accelerate.init_empty_weights():
            model = ZImageTransformer2DModel.from_config(
                ZImageTransformer2DModel.load_config(transformer_path, local_files_only=True)
            )

        sd = sdnq_sd_loader(transformer_path, compute_dtype=compute_dtype)
        # Diffusers-format Z-Image keys already match ZImageTransformer2DModel.state_dict(),
        # so no BFL→diffusers conversion is needed here. The transformer has no tied/shared weights,
        # so we expect a complete state dict — any missing key would leave a required parameter on a
        # meta tensor and fail later during device movement or inference. Fail fast here instead.
        missing, unexpected = model.load_state_dict(sd, assign=True, strict=False)
        raise_on_incomplete_sdnq_load("SDNQ Z-Image transformer", missing, unexpected)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.Qwen3Encoder)
class Qwen3EncoderLoader(ModelLoader):
    """Class to load standalone Qwen3 Encoder models for Z-Image (directory format)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Qwen3Encoder_Qwen3Encoder_Config):
            raise ValueError("Only Qwen3Encoder_Qwen3Encoder_Config models are supported here.")

        model_path = Path(config.path)

        # Support both structures:
        # 1. Full model: model_root/text_encoder/ and model_root/tokenizer/
        # 2. Standalone download: model_root/ contains text_encoder files directly
        text_encoder_path = model_path / "text_encoder"
        tokenizer_path = model_path / "tokenizer"

        # Check if this is a standalone text_encoder download (no nested text_encoder folder)
        is_standalone = not text_encoder_path.exists() and (model_path / "config.json").exists()

        if is_standalone:
            text_encoder_path = model_path
            tokenizer_path = model_path  # Tokenizer files should also be in root

        match submodel_type:
            case SubModelType.Tokenizer:
                # Use local_files_only=True to prevent network requests for validation
                # The tokenizer files should already exist locally in the model directory
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            case SubModelType.TextEncoder:
                # Determine safe dtype based on target device capabilities
                target_device = TorchDevice.choose_torch_device()
                model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
                # Use local_files_only=True to prevent network requests for validation
                return Qwen3ForCausalLM.from_pretrained(
                    text_encoder_path,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.ControlNet, format=ModelFormat.Checkpoint)
class ZImageControlCheckpointModel(ModelLoader):
    """Class to load Z-Image Control adapter models from safetensors checkpoint.

    Z-Image Control models are standalone adapters containing control layers
    (control_layers, control_all_x_embedder, control_noise_refiner) that can be
    combined with a base ZImageTransformer2DModel at runtime for spatial conditioning
    (Canny, HED, Depth, Pose, MLSD).
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are supported here.")

        # ControlNet type models don't use submodel_type - load the adapter directly
        return self._load_control_adapter(config)

    def _load_control_adapter(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from safetensors.torch import load_file

        from invokeai.backend.z_image.z_image_control_adapter import ZImageControlAdapter

        assert isinstance(config, ControlNet_Checkpoint_ZImage_Config)
        model_path = Path(config.path)

        # Load the safetensors state dict
        sd = load_file(model_path)
        # Before the geometry probe, because the shapes it reads are meaningless on a packed weight.
        reject_quantized_side_channel(sd, f"Z-Image ControlNet checkpoint {model_path.name}")

        # Determine number of control blocks from state dict
        # Control blocks are named control_layers.0, control_layers.1, etc.
        control_block_indices = set()
        for key in sd.keys():
            if key.startswith("control_layers."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    control_block_indices.add(int(parts[1]))
        num_control_blocks = len(control_block_indices) if control_block_indices else 6

        # Determine number of refiner layers from state dict
        refiner_indices: set[int] = set()
        for key in sd.keys():
            if key.startswith("control_noise_refiner."):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    refiner_indices.add(int(parts[1]))
        n_refiner_layers = len(refiner_indices) if refiner_indices else 2

        # Determine control_in_dim from embedder weight shape
        # control_in_dim = weight.shape[1] / (f_patch_size * patch_size * patch_size)
        # For patch_size=2, f_patch_size=1: control_in_dim = weight.shape[1] / 4
        control_in_dim = 16  # Default for V1
        embedder_key = "control_all_x_embedder.2-1.weight"
        if embedder_key in sd:
            weight_shape = sd[embedder_key].shape
            # weight_shape[1] = f_patch_size * patch_size * patch_size * control_in_dim
            control_in_dim = weight_shape[1] // 4  # 4 = 1 * 2 * 2

        # Log detected configuration for debugging
        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)
        version = "V2.0" if control_in_dim > 16 else "V1"
        logger.info(
            f"Z-Image ControlNet detected: {version} "
            f"(control_in_dim={control_in_dim}, num_control_blocks={num_control_blocks}, "
            f"n_refiner_layers={n_refiner_layers})"
        )

        # Create an empty control adapter
        dim = 3840
        with accelerate.init_empty_weights():
            model = ZImageControlAdapter(
                num_control_blocks=num_control_blocks,
                control_in_dim=control_in_dim,
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                dim=dim,
                n_refiner_layers=n_refiner_layers,
                n_heads=30,
                n_kv_heads=30,
                norm_eps=1e-05,
                qk_norm=True,
            )

        # Load state dict with strict=False to handle missing keys like x_pad_token
        # Some control adapters may not include x_pad_token in their checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(sd, assign=True, strict=False)
        log_unexpected_keys("Z-Image ControlNet checkpoint", unexpected_keys)

        # Initialize x_pad_token if it was missing from the checkpoint
        if "x_pad_token" in missing_keys:
            import torch.nn as nn

            model.x_pad_token = nn.Parameter(torch.empty(dim))
            nn.init.normal_(model.x_pad_token, std=0.02)

        return model


def _fold_comfy_scaled_weights(sd: dict[str, Any], dtype: torch.dtype) -> int:
    """Fold every ComfyUI-style ``weight_scale`` into its weight, in place. Returns how many.

    ComfyUI stores quantized weights with accompanying scale factors (``layer.weight`` quantized,
    ``layer.weight_scale`` the factor, both spellings), so ``dequantized = weight * weight_scale``.
    See https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md.

    A named function rather than a loop inside the loader so the scale-axis contract below is
    reachable from a test. `expand_weight_scale` handles all three layouts (per-tensor,
    per-output-channel, block-wise); the local loop this replaced left a 1-D per-channel scale
    untouched, and ``(out, in) * (out,)`` then broadcasts on the *last* axis — scaling input
    channels instead of output channels, which is a shape error on a non-square weight and a
    silently wrong weight on a square one.

    The multiply runs in float32 for precision but each result is stored as ``dtype`` immediately,
    so the whole model is never materialized in float32: holding every dequantized weight at fp32
    until the caller's later cast quadruples the per-parameter cost (4 bytes vs 1 on disk) and
    dominates the cold-load RAM peak — enough to swap a 32 GB machine. Same fix as in the FLUX.2
    and Krea-2 loaders.
    """
    folded = 0
    for weight_key, scale_key in list(iter_weight_scale_pairs(sd)):
        # Before the cast: `.float()` on an E8M0 grid turns the exponent bytes into ordinary numbers
        # and loses the only evidence of what they were.
        reject_undecoded_mx_scale(weight_key[: -len(".weight")], sd[scale_key])
        # Float8 needs `.float()`; torch has no direct type promotion for it.
        weight_float = sd[weight_key].float()
        scale = expand_weight_scale(weight_float, sd[scale_key].float(), weight_key)
        sd[weight_key] = (weight_float * scale).to(dtype)
        del weight_float
        folded += 1
    return folded


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.Checkpoint)
class Qwen3EncoderCheckpointLoader(ModelLoader):
    """Class to load single-file Qwen3 Encoder models for Z-Image (safetensors format)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Qwen3Encoder_Checkpoint_Config):
            raise ValueError("Only Qwen3Encoder_Checkpoint_Config models are supported here.")

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_from_singlefile(config)
            case SubModelType.Tokenizer:
                # Single-file checkpoints ship no tokenizer files; use the vendored copy.
                return self._load_bundled_tokenizer()

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_bundled_tokenizer(self) -> AnyModel:
        """Load the Qwen3 tokenizer from the vendored, bundled copy.

        Single-file / GGUF checkpoints do not ship tokenizer files. The Qwen3 BPE
        tokenizer is identical across the 0.6B / 4B / 8B variants, so we load the
        self-contained copy vendored in the package — fully offline, no HuggingFace
        download required.
        """
        return load_bundled_qwen3_tokenizer()

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from safetensors.torch import load_file
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        if not isinstance(config, Qwen3Encoder_Checkpoint_Config):
            raise TypeError(
                f"Expected Qwen3Encoder_Checkpoint_Config, got {type(config).__name__}. "
                "Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the state dict from safetensors file
        sd = load_file(model_path)
        original_key_count = len(sd)

        # Three ComfyUI side channels reach this loader and a file carries one of them. int8 is
        # decided first because it is the one the others cannot be told apart from by structure: an
        # int8 layer ships a `.weight_scale` too, so the scaled-fp8 fold further down would pair
        # every int8 code tensor with its scale and widen it, and the blanket cast after that would
        # turn what survived into bf16 integers. Neither raises.
        int8_markers = extract_int8_convrot_markers(sd)

        # Outside the branch on purpose -- see the helper. An int8 weight whose marker is missing or
        # unparseable is the case that has no structural signature at all.
        reject_unmarked_int8_weights(sd, int8_markers, "Qwen3 encoder")

        if "lm_head" in int8_markers:
            # Same reason the nvfp4 payloads drop it below, but the int8 codes are still in `sd` and
            # so the whole layer has to go: `tie_weights` assigns the embedding Parameter straight
            # over an installed `Int8ConvrotLinear`'s buffers, leaving a module that derotates a
            # bf16 embedding table. Dropping only the marker would instead leave the codes for the
            # cast to widen into bf16 integers. Either way nothing raises.
            del int8_markers["lm_head"]
            for suffix in ("weight", "weight_scale"):
                sd.pop(f"lm_head.{suffix}", None)

        # Comfy's fp4_mixed encoders keep most projections in nvfp4, beside scaled fp8. Take those out before
        # anything below reads the side channel: the fold pairs every `weight_scale` with its weight and would
        # stretch nvfp4's block scales over the packed codes, and the cast further down would widen them.
        nvfp4_payloads = pop_nvfp4_layers(
            sd, header_layers=parse_quantization_metadata(read_safetensors_metadata(model_path, logger))
        )
        # `lm_head` is tied to the embeddings below, which replaces the weight a packed module would hold.
        nvfp4_payloads.pop("lm_head", None)

        if int8_markers:
            # After the nvfp4 pop, not before it: this strips every remaining `.comfy_quant`, and a
            # marker is one of the two things `pop_nvfp4_layers` accepts as naming an nvfp4 layer.
            # Stripped first, a mixed file whose nvfp4 layers are named only by markers is refused
            # as an unnamed-layout foreign file. What is left to drop here is `.input_scale` (W8A8
            # activation scales, which this path has nothing to apply), which would otherwise be
            # cast and charged to the reservation.
            sd = drop_unconsumed_quantization_sidecars(sd)

        # Count the number of layers by looking at layer keys
        layer_count = 0
        for key in sd.keys():
            if isinstance(key, str) and key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_idx = int(parts[2])
                        layer_count = max(layer_count, layer_idx + 1)
                    except ValueError:
                        pass

        # Get vocab size from embed_tokens weight shape
        embed_weight = sd.get("model.embed_tokens.weight")
        if embed_weight is None:
            raise ValueError("Could not find model.embed_tokens.weight in state dict")

        vocab_size = embed_weight.shape[0]
        embed_hidden_size = embed_weight.shape[1]

        # Detect model variant based on embed_tokens hidden size and layer count
        # FLUX 2 Klein / Z-Image uses Qwen3 configurations from ComfyUI:
        # Reference: https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/text_encoders/llama.py
        # - Qwen3-4B: hidden_size=2560, 36 layers, 32 heads, 8 KV heads, intermediate=9728
        # - Qwen3-8B: hidden_size=4096, 36 layers, 32 heads, 8 KV heads, intermediate=12288
        if embed_hidden_size == 2560 and layer_count == 36:
            # Qwen3-4B variant (FLUX 2 Klein / Z-Image)
            logger.info("Detected Qwen3-4B variant (FLUX 2 Klein / Z-Image)")
            hidden_size = 2560
            num_attention_heads = 32
            num_kv_heads = 8
            intermediate_size = 9728
            head_dim = 128
            max_position_embeddings = 40960
        elif embed_hidden_size == 4096 and layer_count == 36:
            # Qwen3-8B variant
            logger.info("Detected Qwen3-8B variant")
            hidden_size = 4096
            num_attention_heads = 32
            num_kv_heads = 8
            intermediate_size = 12288
            head_dim = 128
            max_position_embeddings = 40960
        else:
            # Unknown variant - try to detect from weights
            logger.warning(
                f"Unknown Qwen3 variant: embed_hidden_size={embed_hidden_size}, layers={layer_count}. "
                "Attempting to detect configuration from weights..."
            )

            def output_rows(path: str) -> int | None:
                # A packed layer's weight has left `sd`; its payload knows the rows.
                if path in nvfp4_payloads:
                    return nvfp4_payloads[path].out_features
                weight = sd.get(f"{path}.weight")
                return None if weight is None else weight.shape[0]

            q_rows = output_rows("model.layers.0.self_attn.q_proj")
            k_rows = output_rows("model.layers.0.self_attn.k_proj")
            gate_rows = output_rows("model.layers.0.mlp.gate_proj")

            if q_rows is None or k_rows is None or gate_rows is None:
                raise ValueError("Could not find attention/mlp weights to determine configuration")

            hidden_size = embed_hidden_size
            head_dim = 128
            num_attention_heads = q_rows // head_dim
            num_kv_heads = k_rows // head_dim
            intermediate_size = gate_rows
            max_position_embeddings = 40960

        logger.info(
            f"Qwen3 config: hidden_size={hidden_size}, layers={layer_count}, "
            f"heads={num_attention_heads}, kv_heads={num_kv_heads}, intermediate={intermediate_size}"
        )

        # Create Qwen3 config
        qwen_config = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layer_count,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            rope_theta=1000000.0,
            use_sliding_window=False,
            attention_bias=False,
            attention_dropout=0.0,
            torch_dtype=model_dtype,
        )

        # Use Qwen3ForCausalLM - the correct model class for Z-Image text encoder
        # Use init_empty_weights for fast model creation, then load weights with assign=True. Built before the
        # reservation, which depends on its modules: they decide which nvfp4 layers stay packed.
        with accelerate.init_empty_weights():
            model = Qwen3ForCausalLM(qwen_config)
        skip_patterns = _model_declared_skip_patterns(model)

        nvfp4_bytes = predict_nvfp4_install_size(model, nvfp4_payloads, model_dtype, skip_patterns)

        if int8_markers:
            # The projections stay int8-resident, which is the whole point of the build: 8.8 GiB on
            # disk stays 8.8 GiB, against the 15.3 GiB the bf16 release occupies. The reservation,
            # the split and the cast all have to agree on which layers those are, which is what the
            # shared install is for.
            kept = install_int8_convrot_layers(
                model,
                sd,
                int8_markers,
                model_dtype,
                architecture="Qwen3 encoder",
                reserve=self._ram_cache.make_room,
                skip_patterns=skip_patterns,
                extra_reserved_bytes=nvfp4_bytes,
            )
            logger.info(f"Kept {len(kept)} of {len(int8_markers)} layer(s) in int8 (dequantized per forward)")
        else:
            # Handle memory management before anything below widens a weight: the scaled-fp8 fold turns every
            # quantized layer into the compute dtype, and the base loader reserved only the file size. One
            # reservation for what the state dict ends up holding -- every tensor but the scale metadata at the
            # compute dtype, plus the nvfp4 layers as they will be held -- since `make_room` makes that much room
            # rather than adding to an earlier one.
            # The excluded scale keys are still in `sd` and still resident, so this sum is short by
            # them -- bounded, not open-ended: `_fold_comfy_scaled_weights` refuses an MXFP8 grid by
            # name before folding, so the residue here is per-tensor and per-channel float32 scales,
            # kilobytes on this encoder. That is why this seam keeps its own sum rather than
            # `reserve_for_load`, which sizes a dict the side channel has been *popped* out of.
            new_sd_size = sum(
                tensor.nelement() * model_dtype.itemsize for key, tensor in sd.items() if not is_scale_metadata_key(key)
            )
            self._ram_cache.make_room(new_sd_size + nvfp4_bytes)

            # Handle ComfyUI quantized checkpoints
            # ComfyUI stores quantized weights with accompanying scale factors:
            # - layer.weight: quantized data (FP8)
            # - layer.weight_scale: scale factor (FP32 scalar)
            # Dequantization formula: dequantized = weight.to(dtype) * weight_scale
            # Reference: https://github.com/Comfy-Org/ComfyUI/blob/master/QUANTIZATION.md
            dequantized_count = _fold_comfy_scaled_weights(sd, model_dtype)

            if dequantized_count > 0:
                logger.info(f"Dequantized {dequantized_count} ComfyUI quantized weights")

            # Filter out ComfyUI quantization metadata keys (comfy_quant, weight_scale)
            # These are no longer needed after dequantization
            comfy_metadata_keys = [k for k in sd.keys() if is_scale_metadata_key(k)]
            for k in comfy_metadata_keys:
                del sd[k]
            if comfy_metadata_keys:
                logger.info(f"Filtered out {len(comfy_metadata_keys)} ComfyUI quantization metadata keys")

            # Convert to target dtype
            for k in sd.keys():
                sd[k] = sd[k].to(model_dtype)

        logger.info(f"Loaded state dict with {len(sd)} keys (originally {original_key_count})")

        if nvfp4_payloads:
            packed = install_nvfp4_layers(model, sd, nvfp4_payloads, model_dtype, skip_patterns)
            logger.info(f"Kept {packed} of {len(nvfp4_payloads)} nvfp4 layer(s) packed.")

        # Load the text model weights from checkpoint
        # assign=True replaces meta tensors with real ones from state dict
        load_state_dict_ignoring_extras(
            model, sd, source="Qwen3 text encoder checkpoint", assign=True, allow_missing=True
        )

        # Handle tied weights: lm_head shares weight with embed_tokens when tie_word_embeddings=True
        # This doesn't work automatically with init_empty_weights, so we need to manually tie them
        if qwen_config.tie_word_embeddings:
            model.tie_weights()

        # Re-initialize any remaining meta tensor buffers (like rotary embeddings inv_freq)
        # These are computed from config, not loaded from checkpoint
        for name, buffer in list(model.named_buffers()):
            if buffer.is_meta:
                # Get parent module and buffer name
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    buffer_name = parts[1]
                else:
                    parent = model
                    buffer_name = name

                # Re-initialize the buffer based on expected shape and dtype
                # For rotary embeddings, this is inv_freq which is computed from config
                if buffer_name == "inv_freq":
                    # Compute inv_freq from config (same logic as Qwen3RotaryEmbedding.__init__)
                    # NB: transformers 5.x moved rope_theta into the rope_parameters/rope_scaling dict
                    rope_params = (
                        getattr(qwen_config, "rope_parameters", None)
                        or getattr(qwen_config, "rope_scaling", None)
                        or {}
                    )
                    base = rope_params.get("rope_theta") or getattr(qwen_config, "rope_theta", 1000000.0)
                    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    parent.register_buffer(buffer_name, inv_freq.to(model_dtype), persistent=False)
                else:
                    # For other buffers, log warning
                    logger.warning(f"Re-initializing unknown meta buffer: {name}")

        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.GGUFQuantized)
class Qwen3EncoderGGUFLoader(ModelLoader):
    """Class to load GGUF-quantized Qwen3 Encoder models for Z-Image."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Qwen3Encoder_GGUF_Config):
            raise ValueError("Only Qwen3Encoder_GGUF_Config models are supported here.")

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_from_gguf(config)
            case SubModelType.Tokenizer:
                # GGUF checkpoints ship no tokenizer files; use the vendored copy.
                return self._load_bundled_tokenizer()

        raise ValueError(
            f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_bundled_tokenizer(self) -> AnyModel:
        """Load the Qwen3 tokenizer from the vendored, bundled copy.

        Single-file / GGUF checkpoints do not ship tokenizer files. The Qwen3 BPE
        tokenizer is identical across the 0.6B / 4B / 8B variants, so we load the
        self-contained copy vendored in the package — fully offline, no HuggingFace
        download required.
        """
        return load_bundled_qwen3_tokenizer()

    def _load_from_gguf(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        if not isinstance(config, Qwen3Encoder_GGUF_Config):
            raise TypeError(
                f"Expected Qwen3Encoder_GGUF_Config, got {type(config).__name__}. Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the GGUF state dict - this returns GGMLTensor wrappers (on CPU)
        # We keep them on CPU and let the model cache system handle GPU movement
        # via apply_custom_layers_to_model() and the partial loading cache
        sd = gguf_sd_loader(model_path, compute_dtype=compute_dtype)

        if is_llamacpp_decoder_state_dict(sd):
            logger.info("Detected llama.cpp GGUF format, converting keys to PyTorch format")
            sd = convert_llamacpp_decoder_keys(sd)

        # Determine Qwen model configuration from state dict
        # Count the number of layers by looking at layer keys
        layer_count = 0
        for key in sd.keys():
            if isinstance(key, str) and key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_idx = int(parts[2])
                        layer_count = max(layer_count, layer_idx + 1)
                    except ValueError:
                        pass

        # Get vocab size from embed_tokens weight shape
        embed_weight = sd.get("model.embed_tokens.weight")
        if embed_weight is None:
            raise ValueError("Could not find model.embed_tokens.weight in state dict")

        # Handle GGMLTensor shape access
        embed_shape = embed_weight.shape if hasattr(embed_weight, "shape") else embed_weight.tensor_shape
        if len(embed_shape) != 2:
            raise ValueError(
                f"Expected 2D embed_tokens weight tensor, got shape {embed_shape}. "
                "The model file may be corrupted or incompatible."
            )
        vocab_size = embed_shape[0]

        # Detect attention configuration from layer weights
        # IMPORTANT: Use layer 1 (not layer 0) because some models like FLUX 2 Klein have a special
        # first layer with different dimensions (input projection layer) while the rest of the
        # transformer layers have a different hidden_size. Using a middle layer ensures we get
        # the representative hidden_size for the bulk of the model.
        # Fall back to layer 0 if layer 1 doesn't exist.
        q_proj_weight = sd.get("model.layers.1.self_attn.q_proj.weight")
        k_proj_weight = sd.get("model.layers.1.self_attn.k_proj.weight")
        gate_proj_weight = sd.get("model.layers.1.mlp.gate_proj.weight")

        # Fall back to layer 0 if layer 1 doesn't exist (single-layer model edge case)
        if q_proj_weight is None:
            q_proj_weight = sd.get("model.layers.0.self_attn.q_proj.weight")
            k_proj_weight = sd.get("model.layers.0.self_attn.k_proj.weight")
            gate_proj_weight = sd.get("model.layers.0.mlp.gate_proj.weight")

        if q_proj_weight is None or k_proj_weight is None or gate_proj_weight is None:
            raise ValueError("Could not find attention/mlp weights in state dict to determine configuration")

        # Handle GGMLTensor shape access
        q_shape = q_proj_weight.shape if hasattr(q_proj_weight, "shape") else q_proj_weight.tensor_shape
        k_shape = k_proj_weight.shape if hasattr(k_proj_weight, "shape") else k_proj_weight.tensor_shape
        gate_shape = gate_proj_weight.shape if hasattr(gate_proj_weight, "shape") else gate_proj_weight.tensor_shape

        # Calculate dimensions from actual weights
        # IMPORTANT: Use hidden_size from k_proj input dimension (not q_proj or embed_tokens).
        # Some models (like FLUX 2 Klein) have unusual architectures where:
        # - embed_tokens has a larger dimension (e.g., 2560)
        # - q_proj may have a larger input dimension for query expansion
        # - k_proj/v_proj have the actual transformer hidden_size (e.g., 1280)
        # Using k_proj ensures we get the correct internal hidden_size.
        head_dim = 128  # Standard head dimension for Qwen3 models
        hidden_size = k_shape[1]  # Use k_proj input dim as the hidden_size
        num_attention_heads = q_shape[0] // head_dim
        num_kv_heads = k_shape[0] // head_dim
        intermediate_size = gate_shape[0]

        logger.info(
            f"Qwen3 GGUF Encoder config detected: layers={layer_count}, hidden={hidden_size}, "
            f"heads={num_attention_heads}, kv_heads={num_kv_heads}, intermediate={intermediate_size}, "
            f"head_dim={head_dim}"
        )

        # Create Qwen3 config
        qwen_config = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layer_count,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=40960,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            rope_theta=1000000.0,
            use_sliding_window=False,
            attention_bias=False,
            attention_dropout=0.0,
            torch_dtype=compute_dtype,
        )

        # Use Qwen3ForCausalLM with empty weights, then load GGUF tensors
        with accelerate.init_empty_weights():
            model = Qwen3ForCausalLM(qwen_config)

        # Load the GGUF weights with assign=True
        # GGMLTensor wrappers will be dequantized on-the-fly during inference
        load_state_dict_ignoring_extras(model, sd, source="Qwen3 GGUF text encoder", assign=True, allow_missing=True)

        if materialize_quantized_embedding(model.model.embed_tokens, ram_cache=self._ram_cache):
            logger.info("Dequantized embed_tokens weight for embedding lookups")

        # Handle tied weights - llama.cpp GGUF doesn't include lm_head.weight when embeddings are tied
        # So we need to manually tie them after loading
        if qwen_config.tie_word_embeddings:
            # Check if lm_head.weight is still a meta tensor (wasn't in GGUF state dict)
            if model.lm_head.weight.is_meta:
                # Directly assign embed_tokens weight to lm_head (now dequantized)
                model.lm_head.weight = model.model.embed_tokens.weight
                logger.info("Tied lm_head.weight to embed_tokens.weight (GGUF tied embeddings)")
            else:
                # If lm_head.weight was loaded, use standard tie_weights
                model.tie_weights()

        # Re-initialize any remaining meta tensor buffers (like rotary embeddings inv_freq)
        for name, buffer in list(model.named_buffers()):
            if buffer.is_meta:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    buffer_name = parts[1]
                else:
                    parent = model
                    buffer_name = name

                if buffer_name == "inv_freq":
                    # Compute inv_freq from config - keep on CPU, cache system will move to GPU as needed
                    # NB: transformers 5.x moved rope_theta into the rope_parameters/rope_scaling dict
                    rope_params = (
                        getattr(qwen_config, "rope_parameters", None)
                        or getattr(qwen_config, "rope_scaling", None)
                        or {}
                    )
                    base = rope_params.get("rope_theta") or getattr(qwen_config, "rope_theta", 1000000.0)
                    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    parent.register_buffer(buffer_name, inv_freq.to(dtype=compute_dtype), persistent=False)
                else:
                    logger.warning(f"Re-initializing unknown meta buffer: {name}")

        # Final check: ensure no meta tensors remain in parameters
        meta_params = [(name, p) for name, p in model.named_parameters() if p.is_meta]
        if meta_params:
            meta_names = [name for name, _ in meta_params]
            raise RuntimeError(
                f"Failed to load all parameters from GGUF. The following remain as meta tensors: {meta_names}. "
                "This may indicate missing keys in the GGUF file or a key mapping issue."
            )

        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.SDNQQuantized)
class Qwen3EncoderSDNQLoader(ModelLoader):
    """Class to load SDNQ-quantized Qwen3 Encoder models for Z-Image.

    SDNQ exports carry packed weights only, so the tokenizer comes from the copy vendored in
    `invokeai.backend.qwen3` -- the same one the single-file and GGUF Qwen3 encoders already use.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, (Qwen3Encoder_SDNQ_Config, Qwen3Encoder_SDNQ_Folder_Config)):
            raise ValueError(
                "Only Qwen3Encoder_SDNQ_Config or Qwen3Encoder_SDNQ_Folder_Config models are supported here."
            )

        match submodel_type:
            case SubModelType.TextEncoder:
                return self._load_from_sdnq(config)
            case SubModelType.Tokenizer:
                return load_bundled_qwen3_tokenizer()

        submodel_str = submodel_type.value if submodel_type else "None"
        raise ValueError(f"Only TextEncoder and Tokenizer submodels are supported. Received: {submodel_str}")

    def _load_from_sdnq(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from transformers import Qwen3Config, Qwen3ForCausalLM

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        if not isinstance(config, (Qwen3Encoder_SDNQ_Config, Qwen3Encoder_SDNQ_Folder_Config)):
            raise TypeError(
                f"Expected Qwen3Encoder_SDNQ_Config or Qwen3Encoder_SDNQ_Folder_Config, got {type(config).__name__}."
            )
        model_path = Path(config.path)

        # A folder encoder ships in one of two layouts: weights at the root, or under `text_encoder/`
        # next to a sibling `tokenizer/`. `sdnq_sd_loader` globs one directory and reads the
        # `quantization_config.json` beside it, so pointing it at the root of a nested install finds
        # no shards ("No safetensors files found") and would fall back to a default `group_size`.
        # Resolve the layout with the same helper identification used, so the loader can open every
        # folder identification accepts. Single-file configs carry a file path and are left alone.
        if isinstance(config, Qwen3Encoder_SDNQ_Folder_Config):
            model_path = Qwen3Encoder_SDNQ_Folder_Config.resolve_text_encoder_dir(model_path)

        # Determine safe dtype based on target device capabilities
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Load the SDNQ state dict - this returns SDNQTensor wrappers (on CPU)
        sd = sdnq_sd_loader(model_path, compute_dtype=compute_dtype)

        # Determine Qwen model configuration from state dict
        layer_count = 0
        for key in sd.keys():
            if isinstance(key, str) and key.startswith("model.layers."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_idx = int(parts[2])
                        layer_count = max(layer_count, layer_idx + 1)
                    except ValueError:
                        pass

        # Get hidden size from embed_tokens weight shape
        embed_weight = sd.get("model.embed_tokens.weight")
        if embed_weight is None:
            raise ValueError("Could not find model.embed_tokens.weight in state dict")

        embed_shape = embed_weight.shape if hasattr(embed_weight, "shape") else embed_weight.tensor_shape
        if len(embed_shape) != 2:
            raise ValueError(f"Expected 2D embed_tokens weight tensor, got shape {embed_shape}.")
        hidden_size = embed_shape[1]
        vocab_size = embed_shape[0]

        # Detect attention configuration from layer 0 weights
        q_proj_weight = sd.get("model.layers.0.self_attn.q_proj.weight")
        k_proj_weight = sd.get("model.layers.0.self_attn.k_proj.weight")
        gate_proj_weight = sd.get("model.layers.0.mlp.gate_proj.weight")

        if q_proj_weight is None or k_proj_weight is None or gate_proj_weight is None:
            raise ValueError("Could not find attention/mlp weights in state dict to determine configuration")

        q_shape = q_proj_weight.shape if hasattr(q_proj_weight, "shape") else q_proj_weight.tensor_shape
        k_shape = k_proj_weight.shape if hasattr(k_proj_weight, "shape") else k_proj_weight.tensor_shape
        gate_shape = gate_proj_weight.shape if hasattr(gate_proj_weight, "shape") else gate_proj_weight.tensor_shape

        head_dim = 128  # Standard head dimension for Qwen3 models
        num_attention_heads = q_shape[0] // head_dim
        num_kv_heads = k_shape[0] // head_dim
        intermediate_size = gate_shape[0]

        logger.info(
            f"Qwen3 SDNQ Encoder config detected: layers={layer_count}, hidden={hidden_size}, "
            f"heads={num_attention_heads}, kv_heads={num_kv_heads}, intermediate={intermediate_size}, "
            f"head_dim={head_dim}"
        )

        # Create Qwen3 config
        qwen_config = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=layer_count,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=40960,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
            rope_theta=1000000.0,
            use_sliding_window=False,
            attention_bias=False,
            attention_dropout=0.0,
            torch_dtype=compute_dtype,
        )

        # Use Qwen3ForCausalLM with empty weights, then load SDNQ tensors
        with accelerate.init_empty_weights():
            model = Qwen3ForCausalLM(qwen_config)

        # Load the SDNQ weights with assign=True. lm_head is tied to embed_tokens (re-shared below),
        # so it is expected to be missing; any other missing key (e.g. from a partial or incompatible
        # export) must fail here. Unexpected keys are exporter noise and are only logged at DEBUG
        # (see `raise_on_incomplete_sdnq_load`).
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        raise_on_incomplete_sdnq_load("SDNQ Qwen3 encoder", missing, unexpected, allowed_missing={"lm_head.weight"})

        if materialize_quantized_embedding(model.model.embed_tokens, ram_cache=self._ram_cache):
            logger.info("Dequantized embed_tokens weight for embedding lookups")

        # Handle tied weights
        if qwen_config.tie_word_embeddings:
            if model.lm_head.weight.is_meta:
                model.lm_head.weight = model.model.embed_tokens.weight
                logger.info("Tied lm_head.weight to embed_tokens.weight")
            else:
                model.tie_weights()

        # Re-initialize any remaining meta tensor buffers
        for name, buffer in list(model.named_buffers()):
            if buffer.is_meta:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    buffer_name = parts[1]
                else:
                    parent = model
                    buffer_name = name

                if buffer_name == "inv_freq":
                    base = qwen_config.rope_theta
                    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
                    parent.register_buffer(buffer_name, inv_freq.to(dtype=compute_dtype), persistent=False)
                else:
                    logger.warning(f"Re-initializing unknown meta buffer: {name}")

        # Final check: ensure no meta tensors remain in parameters
        meta_params = [(name, p) for name, p in model.named_parameters() if p.is_meta]
        if meta_params:
            meta_names = [name for name, _ in meta_params]
            raise RuntimeError(
                f"Failed to load all parameters from SDNQ. The following remain as meta tensors: {meta_names}."
            )

        return model
