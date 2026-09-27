# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Krea-2 model loading in InvokeAI."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import accelerate
import torch
from transformers import AutoConfig, AutoTokenizer

from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Krea2_Config, Main_GGUF_Krea2_Config
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import (
    Qwen3VLEncoder_Checkpoint_Config,
    Qwen3VLEncoder_GGUF_Config,
    Qwen3VLEncoder_Qwen3VLEncoder_Config,
)
from invokeai.backend.model_manager.load.load_default import (
    ModelLoader,
    _device_supports_fp8_storage,
    _model_declared_skip_patterns,
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
from invokeai.backend.model_manager.util.llamacpp_keys import convert_llamacpp_decoder_keys
from invokeai.backend.model_manager.util.qwen3_vl import (
    drop_qwen3vl_visual_tower,
    drop_qwen3vl_visual_tower_keys,
    normalize_qwen3vl_rope_config,
    qwen3vl_target_key,
)
from invokeai.backend.quantization.fp8_scaled import (
    INPUT_SCALE_SUFFIXES,
    attach_fp8_scales,
    cast_state_dict,
    dequantize_fp8_scaled,
    detach_layer_sidechannel,
    extract_comfy_quant_hints,
    extract_fp8_scaled_layers,
    full_precision_hints_respected,
    parse_quantization_metadata,
    read_safetensors_metadata,
    reattach_layer_sidechannel,
    should_keep_fp8_weights,
    split_fp8_scaled_layers,
    strip_layer_path_prefix,
    warn_on_unattached_scales,
)
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.quantization.int8_convrot import (
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    reject_unmarked_int8_weights,
    resolve_quantized_module_paths,
)
from invokeai.backend.quantization.load_plan import reserve_for_load
from invokeai.backend.quantization.nvfp4 import (
    WEIGHT_SCALE_2_SUFFIX,
    install_nvfp4_layers,
    pop_nvfp4_layers,
    predict_nvfp4_install_size,
)
from invokeai.backend.qwen3_vl.qwen3_vl_assets import (
    load_bundled_qwen3_vl_config_dict,
    load_bundled_qwen3_vl_tokenizer,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, reject_incomplete_load

# Kept as a module-level alias: this helper moved to model_manager.util.qwen3_vl so the MiniMax H3
# loader can share it without importing across family loaders.
_normalize_qwen3vl_rope_config = normalize_qwen3vl_rope_config


def _to_plain_tensor(value: Any) -> Any:
    """Dequantize a GGMLTensor to a plain tensor (needed before reshape); pass others through."""
    if hasattr(value, "get_dequantized_tensor"):
        return value.get_dequantized_tensor()
    return value


def _put_unique_key(
    dest: dict[Any, Any], key: Any, value: Any, *, source: Any, source_of: dict[Any, Any], what: str
) -> None:
    """Assign ``dest[key] = value``, rejecting a collision produced by a different source key.

    These key normalizers map each source key to exactly one target key. A well-formed checkpoint is
    either fully native or already in the target layout, so two distinct source keys never collapse to
    the same target. A malformed *mixed-layout* checkpoint can, though — e.g. it carries both a native
    ``blocks.0.attn.wq.weight`` and an already-diffusers ``transformer_blocks.0.attn.to_q.weight`` that
    normalize to the same key. Silently overwriting would make the surviving tensor depend on dict
    iteration order, so reject with an actionable message instead of dropping one tensor.
    """
    if key in dest:
        raise RuntimeError(
            f"{what}: source keys {source_of.get(key)!r} and {source!r} both normalize to {key!r}. "
            "The checkpoint appears to mix native and target key layouts; refusing to silently drop "
            "one of the tensors."
        )
    dest[key] = value
    source_of[key] = source


def _is_native_krea2_format(sd: dict[str, Any]) -> bool:
    """Detect the native/ComfyUI Krea-2 key naming (e.g. GGUF) vs. the diffusers naming."""
    return any(
        isinstance(k, str) and (k.startswith(("blocks.", "txtfusion.", "first.")) or ".mod.lin" in k) for k in sd
    )


def _remap_native_layer_paths(layer_names: Any) -> dict[str, str]:
    """Map native/ComfyUI layer paths to their diffusers equivalents.

    ``_quantization_metadata`` names its layers in the checkpoint's own (native) scheme, but the
    scales are extracted after the state dict has been renamed. Rather than restating the rename
    rules - which would drift - each name is pushed through the real converter as a lone
    ``<name>.weight`` entry and the resulting key is read back.

    That assumes the module stores its parameter as ``weight``, which is true of every Linear and so
    of every nvfp4 layer, the one caller left -- and false of Krea-2's norms (``scale``) and
    modulation tables (``lin``). A caller that also has norms to place reads the conversion's own
    ``key_map`` instead of probing it; see the fp8 branch of ``_load_from_singlefile``.
    """
    mapping: dict[str, str] = {}
    for name in layer_names:
        if not isinstance(name, str):
            continue
        try:
            converted = _convert_krea2_native_to_diffusers({f"{name}.weight": torch.empty(0)})
        except Exception:
            continue
        for key in converted:
            if isinstance(key, str) and key.endswith(".weight"):
                mapping[name] = key[: -len(".weight")]
                break
    return mapping


# The original final-block up/down projections have no counterpart in the diffusers
# ``Krea2FinalLayer`` (a clean AdaLN + linear). Named once because two steps act on them: the
# converter drops them, and the single-file loader drops them *before* dequantizing so it never
# spends work - or trips over an exotic scale layout - on tensors that are about to be discarded.
DISCARDED_NATIVE_FINAL_KEYS = ("last.down", "last.up")


def _drop_discarded_native_final_layers(sd: dict[str, Any]) -> dict[str, Any]:
    """Remove the dropped final-block projections together with their quantization metadata.

    An nvfp4 layer's global scale and any activation scale go too: left behind without its weight, the nvfp4 pass
    refuses a global scale as a malformed layer, and an activation scale would linger as an orphan.
    """
    doomed = {
        f"{path}{suffix}"
        for path in DISCARDED_NATIVE_FINAL_KEYS
        for suffix in (".weight", ".weight_scale", WEIGHT_SCALE_2_SUFFIX, ".comfy_quant", *INPUT_SCALE_SUFFIXES)
    }
    if not doomed & set(sd):
        return sd
    return {k: v for k, v in sd.items() if k not in doomed}


def _convert_krea2_native_to_diffusers(sd: dict[str, Any], *, key_map: dict[str, str] | None = None) -> dict[str, Any]:
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
    new_sd: dict[str, Any] = {}
    source_of: dict[Any, Any] = {}
    for key, value in sd.items():
        if not isinstance(key, str):
            _put_unique_key(new_sd, key, value, source=key, source_of=source_of, what="Krea-2 checkpoint")
            continue
        # Drop original-only final-block projections (no diffusers equivalent).
        if key in tuple(f"{p}.weight" for p in DISCARDED_NATIVE_FINAL_KEYS):
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
        elif k.startswith("last.linear."):
            # Prefix rather than an exact match per suffix: a quantized build carries
            # `last.linear.weight_scale` too, and an exact rule leaves it behind under the old name
            # -- which the loader only notices as a missing scale, well after the rename.
            k = "final_layer.linear." + k[len("last.linear.") :]
        elif k == "last.norm.scale":
            k = "final_layer.norm.weight"
        elif k == "last.modulation.lin":
            k = "final_layer.scale_shift_table"
            # Krea2FinalLayer.scale_shift_table is (2, hidden) (scale, shift). Reshape the flat native
            # table just like the per-block (6, hidden) tables below - otherwise load_state_dict(assign=True)
            # installs a wrong-shaped 1-D parameter (which the meta-only completeness guard cannot catch)
            # and the final layer fails at inference.
            value = torch.as_tensor(_to_plain_tensor(value)).reshape(2, -1)

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

        _put_unique_key(new_sd, k, value, source=key, source_of=source_of, what="Krea-2 checkpoint")
        if key_map is not None:
            key_map[key] = k
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

        if submodel_type is SubModelType.TextEncoder:
            # The bundled encoder is the same Qwen3-VL as the standalone one, and this is the path a
            # user gets by default: the loader node falls back to the pipeline's encoder whenever no
            # standalone one is wired up, which is how the "Krea-2 Turbo" starter model is used.
            # Without this, that install keeps the ~0.8 GiB the standalone install no longer pays.
            drop_qwen3vl_visual_tower(result)

        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
        return result


@ModelLoaderRegistry.register(base=BaseModelType.Krea2, type=ModelType.Main, format=ModelFormat.Checkpoint)
class Krea2CheckpointModel(ModelLoader):
    """Class to load Krea-2 transformer models from single-file checkpoints (safetensors).

    Handles plain bf16/fp16 checkpoints, ComfyUI 'scaled fp8' checkpoints (fp8 weight +
    ``.weight_scale``) and ComfyUI 'int8_tensorwise' checkpoints (int8 weight + per-output-channel
    ``.weight_scale`` + a ``.comfy_quant`` marker, optionally convrot-rotated), in both the diffusers
    and native/ComfyUI key naming. Apply the fp8-storage setting to keep the (large) transformer
    fp8-resident; otherwise it loads in full precision.

    The int8 build stays int8-resident: `swap_in_int8_linears` installs `Int8ConvrotLinear`, which
    holds the stored codes and dequantizes per forward, so a 12.0 GiB checkpoint stays 12.0 GiB.
    What it does not get is int8 *compute* -- that needs a kernel InvokeAI does not have yet -- so
    the saving here is resident memory, not speed. ComfyUI's 'nvfp4' build, whose layers the
    safetensors header names, stays packed the same way as `NVFP4Linear`.
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

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = load_file(model_path)
        metadata = read_safetensors_metadata(model_path, self._logger)
        sd = CheckpointPrefix.detect(sd).strip(sd)
        # Discard what the key conversion below would discard anyway, before anything is spent on
        # it. One repack quantizes `last.up` with a blockwise scale grid this decode does not
        # implement; refusing a tensor that is on its way to the bin would be an odd way to fail.
        sd = _drop_discarded_native_final_layers(sd)

        # Comfy's nvfp4 build names its layers in the header, natively. Take them out before either side-channel
        # format below reads a scale: both pair every `weight_scale` with its weight, and the key conversion would
        # rename the packed codes without their global scale. `install_nvfp4_layers` puts them back, packed, under
        # their diffusers paths once the model exists. The key scheme is read once, here, for the layers and for
        # both branches below.
        header_layers = strip_layer_path_prefix(parse_quantization_metadata(metadata))
        native = _is_native_krea2_format(sd)
        nvfp4_payloads = pop_nvfp4_layers(sd, header_layers=header_layers)
        if nvfp4_payloads and native:
            path_map = _remap_native_layer_paths(nvfp4_payloads)
            nvfp4_payloads = {path_map.get(path, path): payload for path, payload in nvfp4_payloads.items()}

        # Two ComfyUI side-channel formats reach this loader and a checkpoint carries one or the
        # other, so the format is decided once here. `int8_tensorwise` has to be recognised before
        # the key conversion below: that conversion renames `.weight` by substring and so carries a
        # sibling `.weight_scale` along but NOT `.comfy_quant`, which would separate an int8 weight
        # from the marker that says it is rotated. Taken down the fp8 path instead, such a weight
        # would be scaled but never un-rotated -- a state dict that loads cleanly and generates
        # noise.
        int8_markers = extract_int8_convrot_markers(sd)

        # Outside the branch on purpose -- see the helper, which explains why. Z-Image has always
        # had this; Krea-2 did not, so an int8 weight whose marker was missing or unparseable was
        # cast to the compute dtype as raw codes and loaded silently.
        reject_unmarked_int8_weights(sd, int8_markers, "Krea-2")

        if int8_markers:
            sd = drop_unconsumed_quantization_sidecars(sd)
            # Native/ComfyUI key naming → diffusers Krea2Transformer2DModel keys.
            key_map: dict[str, str] = {}
            if native:
                sd = _convert_krea2_native_to_diffusers(sd, key_map=key_map)
            quantized = resolve_quantized_module_paths(int8_markers, key_map)
        else:
            # Per-layer `.comfy_quant` markers are read first (they are popped out of `sd` here, before
            # the key conversion). Checkpoints ship the flags in either the header or these markers;
            # without both, a checkpoint using only the per-tensor form has its
            # full_precision_matrix_mult layers silently multiplied in fp8. The header wins on the rare
            # checkpoint carrying both. Header names carry the prefix that was just stripped off the
            # state dict, so strip it from them too or they match nothing.
            layer_hints = {**extract_comfy_quant_hints(sd), **header_layers}
            if native:
                # Take the quantization side channel out before renaming. The converter renames
                # ".weight"-suffixed keys by substring and five more by whole-key equality, so a sibling
                # ".scale_weight", ".input_scale", or any scale on one of the equality-renamed keys
                # (e.g. `last.linear.weight_scale`) would be left behind at its old path while the
                # weight moves — and then silently dropped, leaving the weight unscaled.
                detached = detach_layer_sidechannel(sd)
                key_map = {}
                sd = _convert_krea2_native_to_diffusers(sd, key_map=key_map)
                # The metadata and the detached scales still name layers natively; rename both the way
                # the conversion just renamed their weights, or the scales are orphaned and the per-layer
                # flags (notably full_precision_matrix_mult) match nothing. Read from what the converter
                # did rather than inferred by probing it: a native norm stores its parameter as `scale`,
                # so `blocks.0.prenorm.scale` became `transformer_blocks.0.norm1.weight`, and a probe of
                # `blocks.0.prenorm.weight` matches no norm rule at all.
                #
                # Only a destination ending in `.weight` names a module a scale can hang on.
                # `blocks.0.mod.lin` became `transformer_blocks.0.scale_shift_table`, whose stem is a real
                # module; mapping it would reattach the scale there, count it as placed, and let
                # extraction drop it in silence. Left unmapped, it is reported below instead.
                path_map = {
                    native.rsplit(".", 1)[0]: converted[: -len(".weight")]
                    for native, converted in key_map.items()
                    if "." in native and converted.endswith(".weight")
                }
                orphaned = reattach_layer_sidechannel(sd, detached, path_map)
                if orphaned:
                    # INFO, not DEBUG: a dropped scale leaves its weight off by 1/weight_scale with no
                    # other symptom, so the one line that mentions it must be visible by default.
                    self._logger.info(
                        f"Krea-2: dropped quantization side-channel for {len(orphaned)} module(s) with no "
                        f"diffusers counterpart (e.g. {orphaned[0]})."
                    )
                layer_hints = {path_map.get(name, name): hints for name, hints in layer_hints.items()}

            # ComfyUI 'scaled fp8' checkpoints (fp8 weight + .weight_scale, optionally .input_scale).
            fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)
            keep_fp8 = self._keep_fp8_weights(config, SubModelType.Transformer)

        with accelerate.init_empty_weights():
            model = Krea2Transformer2DModel(**KREA2_TRANSFORMER_CONFIG)
        # Honor the model's own precision-sensitive list on every path below. Krea-2 declares `time_embed` and the
        # `norm*` modules; `time_embed.linear_1/linear_2` are ordinary quantized Linears in a ComfyUI export, and a
        # module that reads its own weight's dtype finds a quantized dtype on a module left quantized.
        skip_patterns = _model_declared_skip_patterns(model)
        # A merged file's bundled submodels are not this model's: their dense weights fall to `strict=False` below,
        # and their packed layers go the same way instead of naming no module. The rest are added to either
        # reservation below as they will be held, since a reservation makes that much room rather than adding to one.
        modules = {name for name, _ in model.named_children()}
        nvfp4_payloads = {path: payload for path, payload in nvfp4_payloads.items() if path.split(".", 1)[0] in modules}
        nvfp4_bytes = predict_nvfp4_install_size(model, nvfp4_payloads, model_dtype, skip_patterns)

        if int8_markers:
            # A merged file's bundled submodels are in `sd` too -- this loader does not
            # prefix-filter them out and never loads them -- so the scale check inside is the one
            # that asks the model, not the dict.
            quantized = install_int8_convrot_layers(
                model,
                sd,
                quantized,
                model_dtype,
                architecture="Krea-2",
                reserve=self._ram_cache.make_room,
                skip_patterns=skip_patterns,
                extra_reserved_bytes=nvfp4_bytes,
            )
            self._logger.info(
                f"Krea-2: kept {len(quantized)} of {len(int8_markers)} layer(s) in int8 "
                "(int8_tensorwise checkpoint, dequantized per forward)"
            )
            # The fp8 reporting below is keyed on these; an int8 checkpoint keeps neither.
            fp8_layers = {}
            kept = 0
        else:
            # Reserve before anything below widens a weight -- the fold and the split both do, and
            # reserving afterwards lets either peak land on a cache that was only ever sized for the
            # file. `scaled_layers` is what keeps the prediction honest where the weights are kept:
            # the split also widens layers whose scale layout `scaled_mm` cannot apply, and without
            # the mapping the prediction would charge those 1 byte/element and arrive at 2. Where
            # they are not kept the prediction charges every float at `model_dtype`, folded yet or
            # not, so the number is the same on either side of the fold -- what changes is when the
            # room exists.
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
                # Neither consumer asked for them: keeping them quantized would halve VRAM but
                # dequantize on every forward, so fold the scales into the weights.
                dequantize_fp8_scaled(sd, fp8_layers, model_dtype)
                fp8_layers = {}

            # Scaled layers the cast would dequantize anyway (skip patterns, non-Linear weights) are
            # folded by the split, with their scale applied. Left to `cast_state_dict` they would be
            # cast *without* it and `attach_fp8_scales` would then skip them for no longer being fp8 —
            # a weight silently off by 1/weight_scale. Krea-2's `time_embed.linear_1/linear_2` are
            # ordinary quantized Linears in a ComfyUI export and match the model's `time_embed` pattern.
            fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, model_dtype, model=model, skip_patterns=skip_patterns)
            # A checkpoint with raw fp8 weights (fp8 tensors and no weight_scale) yields no fp8_layers at
            # all, but its weights are still usable on the tensor cores, so the same `keep_fp8` covers
            # both kinds.
            kept = cast_state_dict(
                sd,
                model_dtype,
                keep_fp8=keep_fp8,
                model=model,
                skip_patterns=skip_patterns,
            )

        if nvfp4_payloads:
            packed = install_nvfp4_layers(model, sd, nvfp4_payloads, model_dtype, skip_patterns)
            self._logger.info(f"Krea-2: kept {packed} of {len(nvfp4_payloads)} nvfp4 layer(s) packed.")

        load_state_dict_ignoring_extras(
            model, sd, source="Krea-2 single-file checkpoint", assign=True, allow_missing=True
        )
        _reject_incomplete_load(model, what="Krea-2 single-file checkpoint")
        # `assign=True` aliases every param to its `sd` tensor. Drop the dict's references before
        # the FP8 cast, or each param's `model_dtype` original stays reachable while its fp8 copy is
        # allocated, overshooting the `make_room()` reservation above by ~50%.
        sd.clear()

        if kept and not fp8_layers:
            # Raw fp8: no scales to attach, but say so — otherwise the tensor-core path is invisible,
            # and the only other fp8 log line (the scaled one below) never fires for this checkpoint.
            self._logger.info(
                f"Krea-2: kept {kept} raw fp8 weight(s) quantized (no weight_scale in the checkpoint); "
                "they will run on the fp8 tensor cores with unit scaling."
            )

        if fp8_layers:
            attached = attach_fp8_scales(model, fp8_layers)
            self._logger.info(
                f"Krea-2: kept {attached} layer(s) in fp8 (scaled fp8 checkpoint, kept for {self._fp8_kept_reason()})"
            )
            warn_on_unattached_scales(self._logger, "Krea-2", attached, fp8_layers)
            # Marked layers dequantize on every forward instead of using the tensor cores, which is
            # the single biggest lever on how much fp8_compute actually buys for a given checkpoint.
            # Surface it: otherwise "fp8_compute is on but barely faster" has no visible cause.
            marked = sum(1 for layer in fp8_layers.values() if layer.full_precision_matmul)
            if marked:
                if full_precision_hints_respected():
                    self._logger.info(
                        f"Krea-2: {marked} of {len(fp8_layers)} layer(s) are marked "
                        "full_precision_matrix_mult and will dequantize per forward. Set "
                        "fp8_compute_full_precision_hints=false to run them on the fp8 tensor cores "
                        "instead (faster, but overrides the checkpoint producer's instruction)."
                    )
                else:
                    self._logger.info(
                        f"Krea-2: ignoring the full_precision_matrix_mult marker on {marked} layer(s) "
                        "(fp8_compute_full_precision_hints=false)."
                    )
            # `fp8_storage` exists to *create* fp8 weights from full-precision ones. Here they are
            # already fp8 -- kept either by the matmul or by that setting itself -- so the cast has
            # nothing to add. Say so, otherwise a user who enabled it cannot tell whether it took.
            default_settings = getattr(config, "default_settings", None)
            if default_settings is not None and getattr(default_settings, "fp8_storage", None):
                self._logger.info(
                    "Krea-2: fp8_storage is satisfied by the checkpoint itself - its weights are already "
                    "fp8 and are kept that way, so the layerwise cast is skipped."
                )
            # The layerwise-casting path exists to *produce* fp8 weights from full-precision ones. The
            # checkpoint already is fp8, and its hooks would cast back to the compute dtype without
            # applying weight_scale, so it must not run here.
            return model

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
        sd = CheckpointPrefix.detect(sd).strip(sd)
        # GGUF conversions use the native/ComfyUI compact key naming; remap to diffusers keys.
        if _is_native_krea2_format(sd):
            sd = _convert_krea2_native_to_diffusers(sd)

        with accelerate.init_empty_weights():
            model = Krea2Transformer2DModel(**KREA2_TRANSFORMER_CONFIG)

        load_state_dict_ignoring_extras(model, sd, source="Krea-2 GGUF checkpoint", assign=True, allow_missing=True)
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
                model = Qwen3VLModel.from_pretrained(
                    text_encoder_path,
                    config=te_config,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                # After the load rather than before it: `from_pretrained` builds the module tree
                # itself, so there is no point to intervene at. The tower's weights are therefore
                # read and then freed -- resident size drops, load peak does not.
                drop_qwen3vl_visual_tower(model)
                return model

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )


def _remap_qwen3vl_singlefile_keys(
    sd: dict[str, Any],
    *,
    key_map: dict[str, str] | None = None,
    what: str = "Qwen3-VL encoder checkpoint",
) -> dict[str, Any]:
    """Remap ComfyUI single-file Qwen3-VL keys to the transformers ``Qwen3VLModel`` layout.

    `key_map` records old key -> new key when given, which `resolve_quantized_module_paths` needs to
    carry `comfy_quant` markers onto the module paths the model actually has. `what` names the
    container in a collision message, so a GGUF is not reported as a checkpoint.
    """
    out: dict[str, Any] = {}
    source_of: dict[Any, Any] = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            _put_unique_key(out, k, v, source=k, source_of=source_of, what=what)
            continue
        new_key = qwen3vl_target_key(k)
        _put_unique_key(out, new_key, v, source=k, source_of=source_of, what=what)
        if key_map is not None:
            key_map[k] = new_key
    return out


def _reject_incomplete_load(model: Any, *, what: str) -> None:
    """Krea-2's alias for the shared meta-device completeness sweep.

    ``strict=False`` is used to tolerate benign extra/renamed keys, but it also silently accepts a
    checkpoint that omits required weights — those tensors stay on the meta device and only fail much
    later during inference. Reject such loads here, naming the offending tensors, so an incomplete,
    misidentified, or differently-converted checkpoint fails at load time with an actionable message.
    """
    reject_incomplete_load(model, what=what)


_Qwen3VLSingleFileConfig = TypeVar(
    "_Qwen3VLSingleFileConfig", Qwen3VLEncoder_Checkpoint_Config, Qwen3VLEncoder_GGUF_Config
)


class _Qwen3VLEncoderSingleFileLoader(ModelLoader, Generic[_Qwen3VLSingleFileConfig]):
    """Shared plumbing for the two single-file Qwen3-VL encoder loaders (safetensors and GGUF).

    Neither container ships a config or tokenizer, so both take them from the assets vendored in
    `invokeai.backend.qwen3_vl` -- the file itself says nothing about which Qwen3-VL it is beyond
    its shapes, so the config's recorded variant selects the config. Only the weight decoding
    differs, which is what the subclasses supply.
    """

    # Not a ClassVar: it is parameterized per subclass, which is what lets `_load_text_encoder`
    # narrow its argument to that subclass's config without violating the base's signature.
    CONFIG_CLASS: type[_Qwen3VLSingleFileConfig]

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, self.CONFIG_CLASS):
            raise ValueError(f"Only {self.CONFIG_CLASS.__name__} models are supported here.")

        match submodel_type:
            case SubModelType.Tokenizer:
                return load_bundled_qwen3_vl_tokenizer()
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config)

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )

    @abstractmethod
    def _load_text_encoder(self, config: _Qwen3VLSingleFileConfig) -> AnyModel:
        """Decode this container's weights into the Qwen3-VL module tree."""

    def _load_te_config(self, config: _Qwen3VLSingleFileConfig) -> Any:
        """Build the architecture config for this variant from the vendored copy.

        Normalization stays here rather than being frozen into the vendored file: whether
        `rope_parameters` has to be mirrored onto `rope_scaling` is a property of the installed
        transformers, not of the release. It is a no-op on transformers 5.5.4, which already
        populates both; `TestNormalizeQwen3vlRopeConfig` covers the versions where it is not.
        """
        from transformers import Qwen3VLConfig

        config_dict = load_bundled_qwen3_vl_config_dict(config.variant)
        return _normalize_qwen3vl_rope_config(Qwen3VLConfig.from_dict(config_dict))


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3VLEncoder, format=ModelFormat.Checkpoint)
class Qwen3VLEncoderCheckpointLoader(_Qwen3VLEncoderSingleFileLoader[Qwen3VLEncoder_Checkpoint_Config]):
    """Loads a single-file Qwen3-VL encoder checkpoint (e.g. ComfyUI ``qwen3vl_4b_bf16`` / ``_fp8_scaled``).

    The checkpoint bundles the language model + visual tower; the tower is dropped on the way in (see
    ``drop_qwen3vl_visual_tower_keys``). ComfyUI 'scaled fp8' weights are dequantized to the compute
    dtype on load.
    """

    CONFIG_CLASS = Qwen3VLEncoder_Checkpoint_Config

    def _load_text_encoder(self, config: Qwen3VLEncoder_Checkpoint_Config) -> AnyModel:
        import torch
        from safetensors.torch import load_file
        from transformers import Qwen3VLModel

        model_path = Path(config.path)
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = load_file(str(model_path))
        # Ahead of every pass that reads tensors, so the tower is never dequantized, cast, copied or
        # reserved for. (The file *header* is still read below for quantization metadata, which can
        # therefore still name visual layers; those hints match no surviving key and are ignored.)
        sd = drop_qwen3vl_visual_tower_keys(sd)
        # Same one-or-the-other split as the transformer above, and here it also guards the fp8
        # *detection*: an int8 layer ships a `.weight_scale` too, so probing for scales without
        # ruling out int8 first would keep the encoder "fp8-resident" over weights that were never
        # fp8.
        int8_markers = extract_int8_convrot_markers(sd)

        # Outside the branch on purpose -- see the helper, which explains why. Z-Image has always
        # had this; Krea-2 did not, so an int8 weight whose marker was missing or unparseable was
        # cast to the compute dtype as raw codes and loaded silently.
        reject_unmarked_int8_weights(sd, int8_markers, "Qwen3-VL encoder")

        if int8_markers:
            sd = drop_unconsumed_quantization_sidecars(sd)
            key_map: dict[str, str] = {}
            sd = _remap_qwen3vl_singlefile_keys(sd, key_map=key_map)
            quantized = resolve_quantized_module_paths(int8_markers, key_map)
            source_is_fp8 = False
            fp8_layers = {}
            keep_matmul_fp8 = False
            # An int8 encoder stays int8-resident, so there is nothing for the fp8 matmul path to
            # keep. The storage pass below is off for a different reason: `_apply_fp8_to_nn_module`
            # casts only `_FP8_SUPPORTED_PYTORCH_LAYERS`, and `Int8ConvrotLinear` is not one of them
            # (nor does it own parameters), so it would walk past every quantized layer and convert
            # only what this path already left dense -- work with no saving on the layers that matter.
            use_fp8_storage = False
        else:
            metadata = read_safetensors_metadata(model_path, self._logger)
            # Per-layer markers must be read before extract_fp8_scaled_layers() drops them, and before
            # the key remap, which would not carry a ".comfy_quant" suffix to a sensible destination.
            layer_hints = {**extract_comfy_quant_hints(sd), **parse_quantization_metadata(metadata)}

            # Remap BEFORE pulling the scales out. The remap rewrites whole keys, so each
            # ".weight_scale" travels with its ".weight" and the recovered layer paths already match the
            # model's module paths — which is what attach_fp8_scales() resolves them against.
            sd = _remap_qwen3vl_singlefile_keys(sd)
            layer_hints = {qwen3vl_target_key(path): hints for path, hints in layer_hints.items()}

            # ComfyUI 'scaled fp8' (fp8 weight + .weight_scale). Only the language-model linears are
            # quantized in the checkpoints seen so far.
            fp8_layers = extract_fp8_scaled_layers(sd, layer_hints=layer_hints)
            source_is_fp8 = bool(fp8_layers) or any(
                getattr(t, "dtype", None) in (torch.float8_e4m3fn, torch.float8_e5m2) for t in sd.values()
            )
            # Resolved once. `device_supports_fp8_matmul` deliberately does not cache an inconclusive
            # probe, so two calls can disagree: a transient failure here followed by a success below
            # would leave the scales already folded while the raw Linears stay quantized, i.e. the
            # encoder silently on the storage path with the matmul log line never printed.
            keep_matmul_fp8 = should_keep_fp8_weights(target_device)
            # Resolved here, next to the matmul probe and for the same reason: the fold below is
            # irreversible, so both consumers have to be known before it runs. Asking only about the
            # matmul folded the scales away and then let the storage cast re-quantize the result to
            # *unscaled* fp8 -- on a scaled checkpoint that is ~3% of the weights lost to underflow,
            # for the byte count the file already had.
            use_fp8_storage = source_is_fp8 and _device_supports_fp8_storage(self._torch_device, self._logger)

        te_config = self._load_te_config(config)
        with accelerate.init_empty_weights():
            model = Qwen3VLModel._from_config(te_config)
        # Its weights were dropped from the state dict above, so the module has to go too -- left in
        # place it would stay on the meta device and `_reject_incomplete_load` would rightly refuse.
        drop_qwen3vl_visual_tower(model)

        if int8_markers:
            # `Qwen3VLModel` declares no precision-sensitive modules, but read them rather than
            # assume: the split is also what keeps a marker on a non-Linear (a "quantize
            # everything" repack's 1-D norms) out of `swap_in_int8_linears`.
            skip_patterns = _model_declared_skip_patterns(model)

            quantized = install_int8_convrot_layers(
                model,
                sd,
                quantized,
                model_dtype,
                architecture="Qwen3-VL encoder",
                reserve=self._ram_cache.make_room,
                skip_patterns=skip_patterns,
            )
            self._logger.info(
                f"Qwen3-VL encoder: kept {len(quantized)} of {len(int8_markers)} layer(s) in int8 "
                "(int8_tensorwise checkpoint, dequantized per forward)"
            )
        else:
            # Same ordering contract as every other loader in this series: split *before* the cast.
            # A per-key `if dtype is not FP8_DTYPE` cast looks equivalent and is not — it keeps every
            # fp8 tensor quantized, including the ones that must not stay:
            #
            #  - a 1-D fp8 norm (checkpoints that "quantize everything" ship these) would keep its
            #    scale as an unused buffer on a non-Linear and the forward would compute on the raw
            #    fp8 codes, i.e. off by 1/weight_scale with nothing logged;
            #  - an e5m2 scaled weight would be cast *without* its scale, since only e4m3fn is spared;
            #  - a block-wise scale would reach `scaled_mm_linear` unchecked and raise mid-generation.
            #
            # `split_fp8_scaled_layers` applies exactly those filters and dequantizes the affected
            # layers *with* their scale, so what remains is what the matmul can actually consume.
            # The storage path below re-quantizes to fp8 anyway, so casting the raw fp8 Linears to
            # `model_dtype` here would double both this reservation and the host-RAM peak (~4.4 -> ~8.9
            # GiB on the 4B encoder) for a round trip that ends where it started -- e4m3fn is a subset
            # of bf16, so it is value-exact. Keep them for either consumer.
            keep_fp8 = keep_matmul_fp8 or use_fp8_storage
            # Reserve before anything below widens a weight: the fold widens every scaled layer and
            # the split dequantizes its unusable subset through fp32, so reserving afterwards lets
            # either peak land on a cache that was only ever sized for the file. `scaled_layers`
            # keeps the prediction in step with the split's own scale-layout filter; where the
            # weights are not kept it charges every float at `model_dtype`, folded yet or not, so the
            # number is the same on either side of the fold.
            reserve_for_load(
                self._ram_cache.make_room,
                sd,
                model_dtype,
                keep_fp8=keep_fp8,
                model=model,
                fp8_layers=fp8_layers,
                nvfp4_payloads={},
            )
            if fp8_layers and not keep_fp8:
                # Neither consumer wants them packed: fold the scales into the weights. Both consumers
                # were resolved above, next to the matmul probe, because this fold is irreversible.
                dequantize_fp8_scaled(sd, fp8_layers, model_dtype)
                fp8_layers = {}
            fp8_layers = split_fp8_scaled_layers(sd, fp8_layers, model_dtype, model=model)
            # No `skip_patterns` here on purpose: this model declares none, and the storage pass below
            # applies `_FP8_DEFAULT_SKIP_PATTERNS` itself. Those two lists used to have to not intersect
            # on a layer class with no fp8-capable wrapper -- a `pos_embed` or a `patch_embed.proj`
            # would arrive fp8 from the state dict, be skipped by the cast pass, and then raise on the
            # first forward. `_apply_fp8_to_nn_module` now restores the compute dtype on the modules it
            # skips, so the two lists are independent again.
            cast_state_dict(sd, model_dtype, keep_fp8=keep_fp8, model=model)

        load_state_dict_ignoring_extras(
            model, sd, source="Qwen3-VL encoder checkpoint", assign=True, allow_missing=True
        )
        _reject_incomplete_load(model, what="Qwen3-VL encoder checkpoint")

        if fp8_layers:
            # Keep the weights quantized and let CustomLinear run their matmuls on the fp8 tensor
            # cores. Same resident VRAM as the layerwise-casting path below, but without paying a
            # fp8->bf16 round trip on every forward.
            attached = attach_fp8_scales(model, fp8_layers)
            warn_on_unattached_scales(self._logger, f"Qwen3-VL encoder '{config.name}'", attached, fp8_layers)
            # This pass was originally here to cast the bf16 visual tower, worth ~0.4GB resident. The
            # tower is now dropped before the load, and on qwen3vl_4b_fp8_scaled -- where every Linear
            # is already scaled -- what is left for it to cast is nothing: measured 0.000 GiB saved,
            # conditioning identical (cosine 1.000000).
            #
            # It stays for two reasons that survive the tower, and one that is weaker than it looks.
            # The weak one first: a build that scales only *some* of its Linears would still shrink --
            # but this pass quantizes those to *unscaled* fp8, and per the `nn.Embedding` note below
            # that costs real accuracy, so on such a build it is a trade, not a free win. The two that
            # hold: this is the only call on this branch that records the real compute dtype via
            # `set_fp8_compute_dtype`, without which `get_model_compute_dtype` falls back to scanning
            # for a non-fp8 float parameter -- which happens to work only because the embedding is
            # excluded here and stays bf16. And it installs the backstop that restores the compute
            # dtype on the modules its skip list and the default one both name, which is the documented
            # guard against those two lists ever overlapping. Two exclusions:
            #
            #  - anything carrying a `weight_scale`: those keep their scale and go through
            #    _scaled_mm; the cast hooks would upcast them without it, i.e. a wrong weight.
            #  - `nn.Embedding`: the token embedding table is this model's *input* representation
            #    and it is large (389M params). Measured on qwen3vl_4b_fp8_scaled, quantizing it
            #    doubles the encoder's error against bf16 (relative L2 0.0079 -> 0.0163) to save
            #    371MiB. That is a bad trade for a model whose entire job is text fidelity. (The
            #    old fp8_storage path did cast it — this is strictly more accurate than before.)
            self._apply_fp8_to_nn_module(
                model,
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=model_dtype,
                skip=lambda _name, module: getattr(module, "weight_scale", None) is not None
                or isinstance(module, torch.nn.Embedding),
            )
            self._logger.info(
                f"Qwen3-VL encoder '{config.name}': visual tower dropped; kept {attached} scaled "
                f"layer(s) quantized ({self._fp8_kept_reason()}, storage=float8_e4m3fn, "
                f"compute={model_dtype})."
            )
            # The layerwise-casting path below exists to *produce* fp8 weights from full-precision
            # ones. These already are fp8, and its hooks would cast them to the compute dtype without
            # applying weight_scale — a silently wrong weight. It must not run here.
            return model

        # Keep an fp8 encoder running in fp8 (storage=float8_e4m3fn, per-layer upcast to the compute
        # dtype during forward) on devices that support fp8 storage. `_should_use_fp8` deliberately
        # excludes text encoders (and the config has no fp8_storage toggle), so apply the hook-based
        # casting directly here. This roughly halves the encoder's resident VRAM (~8.9GB bf16 ->
        # ~4.4GB), which avoids partial-load thrashing when it shares the GPU with a large transformer.
        if use_fp8_storage:
            # `model.dtype` now reports the float8 storage dtype; `_apply_fp8_to_nn_module` records
            # the real compute dtype so callers can recover it via `get_model_compute_dtype`.
            self._apply_fp8_to_nn_module(model, storage_dtype=torch.float8_e4m3fn, compute_dtype=model_dtype)
            self._logger.info(
                f"FP8 layerwise casting enabled for Qwen3-VL encoder '{config.name}' "
                f"(storage=float8_e4m3fn, compute={model_dtype})."
            )
        else:
            # Otherwise nothing above has said anything, and an encoder that just became ~0.8GiB
            # smaller than its file should say why.
            self._logger.info(f"Qwen3-VL encoder '{config.name}': visual tower dropped (never used for text).")

        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3VLEncoder, format=ModelFormat.GGUFQuantized)
class Qwen3VLEncoderGGUFLoader(_Qwen3VLEncoderSingleFileLoader[Qwen3VLEncoder_GGUF_Config]):
    """Loads a llama.cpp GGUF Qwen3-VL encoder (the language tower; the visual tower is a separate
    ``mmproj-*.gguf`` this loader neither needs nor accepts).

    The model is still built as a full ``Qwen3VLModel`` from the variant's HuggingFace config rather
    than as a hand-configured ``Qwen3ForCausalLM``. That matters for correctness, not tidiness:
    Qwen3-VL's language tower uses interleaved mRoPE with ``rope_theta`` 5e6 over a 262144-token
    context, none of which is inferable from tensor shapes. Synthesizing a plain Qwen3 config instead
    would silently substitute 1D RoPE at a different base -- a model that loads, runs, and conditions
    wrongly. Building the real architecture also keeps this encoder byte-identical to the safetensors
    one for both consumers, so neither the Krea-2 nor the Ideogram 4 invocation needs a GGUF branch.
    """

    CONFIG_CLASS = Qwen3VLEncoder_GGUF_Config

    def _load_text_encoder(self, config: Qwen3VLEncoder_GGUF_Config) -> AnyModel:
        from transformers import Qwen3VLModel

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Before the file is read, not after. This is the one step that can need the network, and
        # `gguf_sd_loader` copies every tensor into RAM -- so resolving the config first turns a
        # cold-offline failure from "read 2.5 GB, then raise" into an immediate one.
        te_config = self._load_te_config(config)

        sd = gguf_sd_loader(Path(config.path), compute_dtype=model_dtype)
        # Unconditional: identification requires the llama.cpp `token_embd.weight`, so this file is
        # llama.cpp-named by construction. (The converter passes unrecognized keys through anyway.)
        sd = convert_llamacpp_decoder_keys(sd)
        # Reuse the single-file remap so both containers land on identical module paths.
        sd = _remap_qwen3vl_singlefile_keys(sd, what="Qwen3-VL encoder GGUF")

        with accelerate.init_empty_weights():
            model = Qwen3VLModel._from_config(te_config)
        # Nothing to load into it: llama.cpp keeps the visual tower in a companion mmproj file.
        drop_qwen3vl_visual_tower(model)

        load_state_dict_ignoring_extras(model, sd, source="Qwen3-VL encoder GGUF", assign=True, allow_missing=True)

        materialize_quantized_embedding(model.language_model.embed_tokens, ram_cache=self._ram_cache)

        _reject_incomplete_load(model, what="Qwen3-VL encoder GGUF")

        self._logger.info(
            f"Qwen3-VL encoder '{config.name}': loaded {config.variant.value} from GGUF "
            f"(quantized weights kept, compute={model_dtype}); visual tower omitted."
        )
        return model
