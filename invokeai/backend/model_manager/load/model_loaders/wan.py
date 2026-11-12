"""Loader registrations for Wan 2.2 image-generation models.

Currently covers:
- Main: Diffusers format (T2V-A14B with dual experts via Transformer +
  Transformer2 submodels, plus TI2V-5B).
- Main: GGUFQuantized and single-file Checkpoint (safetensors) transformers.
  Both are transformer-only — one file per A14B expert — and rely on a
  standalone VAE + T5 encoder for the rest of the pipeline.
- WanT5Encoder: standalone UMT5-XXL encoder folder (``text_encoder/`` +
  ``tokenizer/`` subdirs, or a flat ``text_encoder/`` folder), or one llama.cpp
  GGUF file.
- VAE: handled in ``vae.py`` (registered for type=VAE generically).
"""

from itertools import chain
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from invokeai.backend.model_manager.checkpoint_prefix import CheckpointPrefix
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_Wan_Config,
    Main_GGUF_Wan_Config,
    _is_benign_extra_key,
    _is_native_wan_layout,
)
from invokeai.backend.model_manager.configs.wan_t5_encoder import WanT5Encoder_GGUF_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader, _model_declared_skip_patterns
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.comfyui_state_dict_utils import (
    _strip_quantization_metadata,
    fold_comfyui_fp8_scale,
    reject_layers_the_fold_cannot_decode,
)
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.model_on_disk import safetensors_meta_state_dict
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.util.llamacpp_t5 import (
    convert_llamacpp_t5_keys,
    infer_t5_encoder_config,
    load_t5_tokenizer_from_gguf,
    make_feed_forward_gguf_safe,
)
from invokeai.backend.quantization.fp8_scaled import (
    COMFY_QUANT_SUFFIX,
    is_scale_metadata_key,
    iter_weight_scale_pairs,
    parse_quantization_metadata,
    predict_cast_state_dict_size,
    read_safetensors_metadata,
    strip_layer_path_prefix,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader, unpack_ggml_at_load
from invokeai.backend.quantization.gguf.utils import TORCH_COMPATIBLE_QTYPES
from invokeai.backend.quantization.int8_convrot import (
    drop_unconsumed_quantization_sidecars,
    extract_int8_convrot_markers,
    install_int8_convrot_layers,
    reject_unmarked_int8_weights,
    resolve_quantized_module_paths,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.state_dict_loading import load_state_dict_ignoring_extras, reject_incomplete_load


@ModelLoaderRegistry.register(base=BaseModelType.Wan, type=ModelType.Main, format=ModelFormat.Diffusers)
class WanDiffusersModel(GenericDiffusersLoader):
    """Loader for Wan 2.2 diffusers-format models (T2V-A14B and TI2V-5B).

    Forces bfloat16 for the transformer and VAE — fp16 is unstable on Wan VAE
    (same issue affects the Flux VAE). Resolves the appropriate Hugging Face
    class for each submodel via the parent loader's ``get_hf_load_class``.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            # Defensive: the registry keys on format, so single-file configs are
            # routed to WanGGUFCheckpointModel / WanCheckpointModel, not here.
            raise TypeError(f"{type(config).__name__} is a single-file config; it does not belong to this loader.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading Wan main pipelines.")

        if submodel_type is SubModelType.VAE:
            from invokeai.backend.wan.rocm_causal_conv3d import patch_wan_causal_conv3d_for_rocm

            patch_wan_causal_conv3d_for_rocm()

        model_path = Path(config.path)
        if submodel_type is SubModelType.Tokenizer:
            from invokeai.backend.wan.tokenizer import load_wan_tokenizer

            return load_wan_tokenizer(model_path / submodel_type.value)

        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        def _load_with_variant_fallback(dtype_kwarg: dict[str, torch.dtype]) -> AnyModel:
            # Some Wan repos ship without a fp16 variant suffix on every submodel.
            # If the requested variant isn't on disk, fall back to the default weights.
            try:
                return load_class.from_pretrained(
                    model_path,
                    **dtype_kwarg,
                    variant=variant,
                    local_files_only=True,
                )
            except OSError as e:
                if variant and "no file named" in str(e):
                    return load_class.from_pretrained(model_path, **dtype_kwarg, local_files_only=True)
                raise

        # bfloat16 across the board: matches Diffusers WanPipeline reference and
        # avoids the fp16 instability seen in the Wan VAE.
        try:
            result: AnyModel = _load_with_variant_fallback({"dtype": torch.bfloat16})
        except TypeError:
            # Older diffusers releases use torch_dtype instead of dtype.
            result = _load_with_variant_fallback({"torch_dtype": torch.bfloat16})
        _copy_tensors_left_on_the_mapping(result, torch.bfloat16)

        # This override exists to force bfloat16, and in doing so it replaced the parent's whole
        # `_load_model` -- including the cast at `GenericDiffusersLoader._load_model`, which every
        # other subclass either keeps or reapplies. So FP8 Storage was offered for Wan diffusers folders
        # and silently did nothing. The weights here are the same `WanTransformer3DModel` the
        # single-file path casts.
        return self._apply_fp8_layerwise_casting(result, config, submodel_type)


def _copy_tensors_left_on_the_mapping(model: Any, dtype: torch.dtype) -> None:
    """Give memory of their own to the tensors ``from_pretrained`` kept off ``dtype`` that still read the file's.

    ``from_pretrained`` memory-maps each shard and casts tensor by tensor; a tensor whose cast is a no-op stays a
    view of the mapping, and a view keeps its whole shard mapped. At ``dtype`` those views are the model's own
    weights, and mapping them is how a bf16 folder loads without a copy, so they stay. Off ``dtype`` they are what
    ``_keep_in_fp32_modules`` (norms, modulation tables, the time embedder) leaves in float32 when the file is
    float32: a few MB per shard, each holding a multi-GB shard whose other tensors were all cast away. The A14B
    experts ship float32, and one would hold 80 GB of working set against its 27 GB bf16 size. Copied, the mapping is
    released with its last view, for a transient the size of the copies.

    The test is ``resizable()``, which is False for a storage torch did not allocate itself -- here, the views of the
    mapping; it does not mean "mapped" in general. It skips what the load already copied: a bf16 file whose
    ``_keep_in_fp32_modules`` are stored in bf16 has them upcast by the load, and a second copy would buy nothing.
    """
    if not isinstance(model, torch.nn.Module):
        return
    with torch.no_grad():
        for tensor in chain(model.parameters(), model.buffers()):
            if tensor.device.type == "cpu" and tensor.dtype != dtype and not tensor.untyped_storage().resizable():
                tensor.data = tensor.data.clone()


# Native (upstream) -> Diffusers key rename rules.
#
# Mirrors diffusers.loaders.single_file_utils.convert_wan_transformer_to_diffusers
# (T2V subset; we don't ship VACE / motion / face-adapter conversion). Order
# matters — `cross_attn`/`self_attn` must come before `.q. .k. .v. .o.` so the
# attention blocks are renamed before the projection suffix swap. The norm2/3
# swap uses a placeholder to avoid collisions during the substring rewrite.
_WAN_NATIVE_TO_DIFFUSERS_RENAMES: tuple[tuple[str, str], ...] = (
    ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
    ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
    ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
    ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
    ("time_projection.1", "condition_embedder.time_proj"),
    ("cross_attn", "attn2"),
    ("self_attn", "attn1"),
    (".o.", ".to_out.0."),
    (".q.", ".to_q."),
    (".k.", ".to_k."),
    (".v.", ".to_v."),
    (".k_img.", ".add_k_proj."),
    (".v_img.", ".add_v_proj."),
    (".norm_k_img.", ".norm_added_k."),
    ("head.modulation", "scale_shift_table"),
    ("head.head", "proj_out"),
    ("modulation", "scale_shift_table"),
    ("ffn.0", "ffn.net.0.proj"),
    ("ffn.2", "ffn.net.2"),
    # norm2 <-> norm3 swap via placeholder
    ("norm2", "norm__placeholder"),
    ("norm3", "norm2"),
    ("norm__placeholder", "norm3"),
    # I2V-only keys (harmless on T2V)
    ("img_emb.proj.0", "condition_embedder.image_embedder.norm1"),
    ("img_emb.proj.1", "condition_embedder.image_embedder.ff.net.0.proj"),
    ("img_emb.proj.3", "condition_embedder.image_embedder.ff.net.2"),
    ("img_emb.proj.4", "condition_embedder.image_embedder.norm2"),
)


def _rename_wan_native_key(key: str) -> str:
    """The diffusers name of one native upstream Wan key."""
    for needle, replacement in _WAN_NATIVE_TO_DIFFUSERS_RENAMES:
        key = key.replace(needle, replacement)
    return key


def _convert_wan_native_to_diffusers(state_dict: dict) -> dict:
    """Rename native upstream Wan keys (ComfyUI / QuantStack) to diffusers names.

    Pure substring replacement — no tensor manipulation — so it's safe to apply
    to a dict of GGMLTensors. Returns a new dict; the input is not mutated.
    """
    converted: dict = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            converted[key] = value
            continue
        converted[_rename_wan_native_key(key)] = value
    return converted


def _unwrap_unquantized_to_compute_dtype(state_dict: dict) -> dict:
    """Replace non-quantized GGMLTensor entries with plain tensors at compute_dtype.

    Why: QuantStack-style GGUFs store biases (and other small tensors) as F16,
    while Wan's ``patch_embedding`` is an ``nn.Conv3d``. ``conv3d`` isn't in
    GGMLTensor's dispatch table, so PyTorch reads the wrapper's underlying F16
    storage directly and crashes against bf16 latents
    (``Input type (c10::BFloat16) and bias type (c10::Half) should be the same``).

    For compatible qtypes (F16/F32/BF16) we just pre-cast to compute_dtype here —
    they're not quantized, there's no benefit to keeping them wrapped, and
    unwrapping them sidesteps the missing-op problem entirely. Genuinely
    quantized tensors (Q4_K, Q6_K, etc.) stay wrapped — their on-demand
    dequantization through the linear/addmm dispatch path still works.
    """
    unwrapped: dict = {}
    for key, value in state_dict.items():
        if isinstance(value, GGMLTensor) and value._ggml_quantization_type in TORCH_COMPATIBLE_QTYPES:
            # GGMLTensor.get_dequantized_tensor() already casts to compute_dtype.
            unwrapped[key] = value.get_dequantized_tensor()
        else:
            unwrapped[key] = value
    return unwrapped


def _drop_benign_extra_keys(sd: dict, source: str, logger: Any) -> None:
    """Remove packaging weights the transformer has no use for, in place.

    Done up front rather than left to ``load_state_dict(strict=False)`` because
    everything between here and there costs real memory: the read, the fp8 dequant pass,
    the cast to the compute dtype, and the RAM-cache reservation. An all-in-one checkpoint
    bundles a full VAE and UMT5-XXL text encoder — several GB, upcast to bf16 and reserved
    in the cache — only for ``load_state_dict`` to discard them. The single-file loader
    runs this over the header, so what it drops is never read.
    """
    dropped = [key for key in sd if isinstance(key, str) and _is_benign_extra_key(key)]
    if not dropped:
        return
    modules = sorted({key.split(".")[0] for key in dropped})
    for key in dropped:
        del sd[key]
    logger.info(
        f"{source}: ignored {len(dropped)} bundled/merged weights not part of the transformer "
        f"({', '.join(modules[:8])}). The VAE and text encoder come from the separately-wired models."
    )


def _raise_for_incompatible_keys(incompatible_keys: Any, source: str) -> None:
    """Fail loudly on anything ``load_state_dict(strict=False)`` quietly discarded.

    Missing keys are the obvious error. Unexpected keys matter just as much here and
    are far easier to miss: several Wan 2.2 derivatives are supersets of the plain
    transformer — Fun-Camera adds ``control_adapter.*`` (6 keys), S2V adds
    ``audio_injector``/``cond_encoder``/``frame_packer`` (165 keys), Animate adds
    ``face_adapter``/``motion_encoder`` (127 keys). They match the probe, build a
    correctly-shaped ``WanTransformer3DModel``, report zero missing keys, and then
    generate with the entire branch they were built around silently absent.

    ``configs.main._find_unsupported_wan_variant_marker`` turns away the families we
    know by name; this is the generic backstop, so a derivative nobody has enumerated
    yet produces an error instead of quietly degraded output.

    Benign extras — bundled VAE/text-encoder weights and merged-LoRA residue — have
    already been removed by ``_drop_benign_extra_keys``, so anything reaching here is
    genuinely unplaceable.
    """
    if incompatible_keys.missing_keys:
        raise RuntimeError(f"{source} is missing model parameters: {sorted(incompatible_keys.missing_keys)[:10]}")

    unexpected = [key for key in incompatible_keys.unexpected_keys if isinstance(key, str)]
    if unexpected:
        # Report the distinct top-level module names rather than hundreds of keys.
        modules = sorted({key.split(".")[0] for key in unexpected})
        raise RuntimeError(
            f"{source} has {len(unexpected)} weights that WanTransformer3DModel has nowhere to put "
            f"(modules: {', '.join(modules[:8])}). This is a Wan variant with extra conditioning "
            "branches — Animate, S2V, Fun-Camera and similar — which InvokeAI cannot run faithfully; "
            "loading it anyway would silently ignore that conditioning."
        )


def _tensor_shape(tensor: Any) -> tuple[int, ...]:
    """Logical shape of a tensor, unwrapping GGMLTensor's packed storage.

    A GGMLTensor's ``.shape`` describes the packed quantized blob, not the weight,
    so the logical dimensions live on ``.tensor_shape``.
    """
    shape = tensor.tensor_shape if isinstance(tensor, GGMLTensor) else tensor.shape
    return tuple(int(dim) for dim in shape)


def _build_wan_transformer_config(sd: dict, source: str) -> dict:
    """Derive ``WanTransformer3DModel`` constructor kwargs from a state dict.

    The state dict must already be prefix-stripped and in the diffusers key
    layout. Shared by the GGUF and single-file checkpoint loaders so a community
    release is described by its own weights rather than by a hard-coded table of
    known repos.

    ``source`` only flavours the error messages.
    """
    num_layers = 0
    for key in sd.keys():
        if isinstance(key, str) and key.startswith("blocks."):
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    num_layers = max(num_layers, int(parts[1]) + 1)
                except ValueError:
                    pass

    def require(key: str) -> tuple[int, ...]:
        tensor = sd.get(key)
        if tensor is None:
            raise RuntimeError(f"{source} is missing {key} after prefix strip and key conversion")
        return _tensor_shape(tensor)

    # Patch embedding gives us in_channels (16/36=A14B, 48=TI2V-5B) and inner dim.
    patch_shape = require("patch_embedding.weight")
    inner_dim = patch_shape[0]
    in_channels = patch_shape[1]

    # Wan uses head_dim=128 throughout the family; num_heads = inner_dim / 128.
    attention_head_dim = 128
    num_attention_heads = inner_dim // attention_head_dim

    ffn_dim = require("blocks.0.ffn.net.0.proj.weight")[0]

    text_w = sd.get("condition_embedder.text_embedder.linear_1.weight")
    text_dim = _tensor_shape(text_w)[1] if text_w is not None else 4096

    # out_channels is read from proj_out.weight directly rather than assumed
    # equal to in_channels: I2V-A14B has in_channels=36 (16 noise + 16
    # ref-image latents + 4 mask, concatenated by the denoise loop) but
    # out_channels=16 (only the noise prediction comes back). proj_out is
    # ``nn.Linear(inner_dim, out_channels * prod(patch_size))`` and
    # patch_size is (1, 2, 2) → prod = 4 for the Wan 2.2 family.
    out_channels = require("proj_out.weight")[0] // 4

    # No fallback for num_layers. It cannot be zero here: that would mean no key starts
    # with `blocks.`, and `require("blocks.0.ffn.net.0.proj.weight")` above has already
    # raised. An earlier revision carried a variant-keyed default (40 for A14B, 30 for
    # TI2V-5B) that was unreachable, and it was the only thing the `variant` argument
    # was used for — so the config is now derived entirely from the weights, which is
    # the point of this helper.

    return {
        "patch_size": (1, 2, 2),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "num_layers": num_layers,
        "attention_head_dim": attention_head_dim,
        "num_attention_heads": num_attention_heads,
        "ffn_dim": ffn_dim,
        "text_dim": text_dim,
    }


@ModelLoaderRegistry.register(base=BaseModelType.Wan, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class WanGGUFCheckpointModel(ModelLoader):
    """Loader for GGUF-quantized Wan 2.2 transformer models.

    The community typically distributes Wan A14B as two files (one per expert
    — high-noise + low-noise). Each file is loaded independently here; the
    pairing happens at the WanModelLoaderInvocation layer. TI2V-5B ships as a
    single file.

    Mirrors the QwenImage GGUF loader pattern: ``gguf_sd_loader`` -> strip the
    ComfyUI ``model.diffusion_model.`` / ``diffusion_model.`` prefix if present
    -> auto-detect arch from state-dict shapes -> ``init_empty_weights`` +
    ``load_state_dict(strict=False, assign=True)``.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_GGUF_Wan_Config):
            raise TypeError(f"Expected Main_GGUF_Wan_Config, got {type(config).__name__}.")

        if submodel_type != SubModelType.Transformer:
            raise ValueError(
                "Only the Transformer submodel is available from a GGUF Wan checkpoint. "
                "Pair with a standalone Wan VAE and Wan T5 encoder for the other components."
            )

        return self._load_from_singlefile(config)

    def _load_from_singlefile(self, config: Main_GGUF_Wan_Config) -> AnyModel:
        import accelerate
        from diffusers import WanTransformer3DModel

        from invokeai.backend.util.logging import InvokeAILogger

        model_path = Path(config.path)
        target_device = TorchDevice.choose_torch_device()
        compute_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = gguf_sd_loader(model_path, compute_dtype=compute_dtype)

        # Strip ComfyUI-style prefixes if present.
        sd = CheckpointPrefix.detect(sd).strip(sd)

        _drop_benign_extra_keys(sd, "GGUF state dict", InvokeAILogger.get_logger(self.__class__.__name__))

        # QuantStack and other community releases ship the native upstream Wan key
        # layout (text_embedding.0, self_attn/cross_attn, ffn.0/2, head.head, ...);
        # diffusers' WanTransformer3DModel expects condition_embedder.*, attn1/attn2,
        # ffn.net.*, proj_out. Convert in place if needed.
        if _is_native_wan_layout(sd):
            sd = _convert_wan_native_to_diffusers(sd)

        # Pre-cast non-quantized tensors (F16/F32/BF16 biases, scale_shift_table,
        # patch_embedding.weight, etc.) to compute_dtype. This avoids dtype
        # mismatches in conv3d at the input (patch_embedding is the only Conv3d
        # in WanTransformer3DModel; conv3d isn't in GGMLTensor's dispatch table
        # so the wrapper's underlying storage dtype reaches PyTorch directly).
        sd = _unwrap_unquantized_to_compute_dtype(sd)

        model_config = _build_wan_transformer_config(sd, source="GGUF state dict")

        with accelerate.init_empty_weights():
            model = WanTransformer3DModel(**model_config)

        incompatible_keys = model.load_state_dict(sd, strict=False, assign=True)
        _raise_for_incompatible_keys(incompatible_keys, source="GGUF state dict")
        return model


def _predict_folded_cast_size(sd: dict, dtype: torch.dtype) -> int:
    """Bytes the state dict occupies once the scale fold and the cast to ``dtype`` have run.

    The fold consumes and strips the scale tensors, and it widens every weight that has a sibling
    scale to ``dtype`` whatever its stored dtype, so those are charged at ``dtype`` too.
    """
    kept = {k: v for k, v in sd.items() if not is_scale_metadata_key(k)}
    folded = {weight_key for weight_key, _ in iter_weight_scale_pairs(sd)}
    total = predict_cast_state_dict_size({k: v for k, v in kept.items() if k not in folded}, dtype, keep_fp8=False)
    return total + sum(kept[k].nelement() * dtype.itemsize for k in folded)


def _read_folded_and_cast(sd: dict[str, torch.Tensor], read: Callable[[str], torch.Tensor], dtype: torch.dtype) -> int:
    """Fill ``sd``'s header entries from ``read`` one tensor at a time, each converted as it arrives.

    A ComfyUI weight scale is folded into its weight and dropped with the rest of the quantization metadata;
    every other float tensor is cast to ``dtype``. Converted on arrival, the peak is the finished state dict plus
    the tensor in flight rather than the whole source. Returns the number of folds.
    """
    scales = dict(iter_weight_scale_pairs(sd))
    _strip_quantization_metadata(sd)
    for key in sd:
        tensor = read(key)
        if key in scales:
            tensor = fold_comfyui_fp8_scale(tensor, read(scales[key]), key, dtype)
        elif tensor.is_floating_point():
            tensor = tensor.to(dtype)
        sd[key] = tensor
    return len(scales)


@ModelLoaderRegistry.register(base=BaseModelType.Wan, type=ModelType.Main, format=ModelFormat.Checkpoint)
class WanCheckpointModel(ModelLoader):
    """Loader for single-file Wan 2.2 transformer checkpoints (safetensors).

    This is what CivitAI fine-tunes and ComfyUI-oriented Hugging Face repos ship.
    Handles the full matrix of community conventions: the optional
    ``model.diffusion_model.`` key prefix, the native upstream key layout as well
    as the diffusers one, ComfyUI ``fp8_scaled`` weights (dequantized to the
    compute dtype at load time), and plain ``float8_e4m3fn`` weights with no
    scales (cast the same way as any other non-bf16 dtype). ComfyUI ``int8_tensorwise`` (convrot) builds keep
    their marked layers in int8 as ``Int8ConvrotLinear`` and dequantize them per forward; layers the model
    reads outside a matmul (``condition_embedder``) are widened at load instead.

    Like the GGUF loader, one file is one expert; A14B pairing happens at the
    WanModelLoaderInvocation layer.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_Wan_Config):
            raise TypeError(f"Expected Main_Checkpoint_Wan_Config, got {type(config).__name__}.")

        if submodel_type != SubModelType.Transformer:
            raise ValueError(
                "Only the Transformer submodel is available from a single-file Wan checkpoint. "
                "Pair with a standalone Wan VAE and Wan T5 encoder for the other components."
            )

        return self._load_from_singlefile(config)

    def _load_from_singlefile(self, config: Main_Checkpoint_Wan_Config) -> AnyModel:
        import accelerate
        from diffusers import WanTransformer3DModel
        from safetensors import safe_open

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        model_path = Path(config.path)
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Everything up to the read -- the prefix, what is dropped, how the file is quantized, the reservation -- is
        # decided on the header's `meta` tensors, so a bundled VAE or text encoder is never read and the cache makes
        # room before the first payload arrives.
        sd = safetensors_meta_state_dict(model_path)
        prefix = CheckpointPrefix.detect(sd)
        file_keys = prefix.strip({key: key for key in sd})
        sd = prefix.strip(sd)
        _drop_benign_extra_keys(sd, "Wan checkpoint", logger)

        # pread gives every tensor memory of its own. Through the default mmap each one is a slice of one file-wide
        # mapping, and a tensor whose cast is a no-op (bf16 on disk) would keep the whole file mapped.
        with safe_open(str(model_path), framework="pt", device="cpu", backend="pread") as f:

            def read(key: str) -> torch.Tensor:
                return f.get_tensor(file_keys[key])

            # The markers are the one payload the decisions below look at.
            for key in [k for k in sd if k.endswith(COMFY_QUANT_SUFFIX)]:
                sd[key] = read(key)

            # Markers and header entries name their layers in the file's native key space, so they are read before
            # any rename. The refusal sits outside the "found markers" branch on purpose: an int8 weight no marker
            # claims must be refused, not cast to the compute dtype as raw codes.
            header_layers = strip_layer_path_prefix(
                parse_quantization_metadata(read_safetensors_metadata(model_path, logger))
            )
            int8_markers = extract_int8_convrot_markers(sd, header_layers=header_layers)
            reject_unmarked_int8_weights(sd, int8_markers, f"Wan 2.2 ({model_path.name})", header_layers=header_layers)

            dequantized = 0
            if int8_markers:
                # Still the header's tensors: `install_int8_convrot_layers` reads them once it has reserved.
                sd = drop_unconsumed_quantization_sidecars(sd)
            else:
                reject_layers_the_fold_cannot_decode(sd, f"Wan checkpoint {model_path.name}")
                # Before the first read: the reservation is the state dict after the fold and the cast.
                self._ram_cache.make_room(_predict_folded_cast_size(sd, model_dtype))
                dequantized = _read_folded_and_cast(sd, read, model_dtype)
                if dequantized > 0:
                    logger.info(f"Dequantized {dequantized} ComfyUI-quantized weights")

            # Community releases ship the native upstream Wan key layout
            # (text_embedding.0, self_attn/cross_attn, ffn.0/2, head.head, ...);
            # diffusers' WanTransformer3DModel expects condition_embedder.*,
            # attn1/attn2, ffn.net.*, proj_out. Convert if needed.
            key_map: dict[str, str] = {}
            if _is_native_wan_layout(sd):
                key_map = {key: _rename_wan_native_key(key) for key in sd if isinstance(key, str)}
                sd = _convert_wan_native_to_diffusers(sd)
                file_keys = _convert_wan_native_to_diffusers(file_keys)

            model_config = _build_wan_transformer_config(sd, source="checkpoint state dict")

            with accelerate.init_empty_weights():
                model = WanTransformer3DModel(**model_config)

            if int8_markers:
                # Casts only the dense tensors: the blanket cast below would also cast the fp32 scales. The skip
                # patterns match diffusers names, hence after the rename; `condition_embedder` reads its dtype from
                # a parameter, which an int8 layer does not have, so it is widened here.
                quantized = install_int8_convrot_layers(
                    model,
                    sd,
                    resolve_quantized_module_paths(int8_markers, key_map),
                    model_dtype,
                    architecture="Wan 2.2",
                    reserve=self._ram_cache.make_room,
                    skip_patterns=_model_declared_skip_patterns(model),
                    read=read,
                )
                # Metadata the fp8 branch strips: a stray `scaled_fp8` flag, or a weight scale beside a dense weight.
                # Left in, `_raise_for_incompatible_keys` would blame it on an Animate/S2V variant. Dropped after the
                # install, whose foreign-scale check has to see (and refuse) a scale that sits on an fp8 weight.
                for key in [k for k in sd if is_scale_metadata_key(k) and k.rsplit(".", 1)[0] not in quantized]:
                    del sd[key]
                logger.info(
                    f"Wan 2.2: kept {len(quantized)} of {len(int8_markers)} layer(s) in int8 "
                    "(int8_tensorwise checkpoint, dequantized per forward)"
                )

        incompatible_keys = model.load_state_dict(sd, strict=False, assign=True)
        _raise_for_incompatible_keys(incompatible_keys, source="Wan checkpoint")

        # A checkpoint that shipped per-tensor scales must not take the cast. The fold above consumed
        # those scales and `_strip_quantization_metadata` dropped them, so re-encoding the result as
        # *unscaled* fp8 would flush the low end of every tensor to zero -- 3.4% of the weights on
        # `flux-2-klein-4b-fp8`, for the byte count the file already had. `configs/factory.py` enables
        # FP8 Storage by itself for a float8 denoiser, on the stated grounds that the loaders keep such
        # a file in its own scaled form; this loader does not do that yet (the scaled-fp8 treatment the
        # FLUX and Krea-2 paths received never reached here), so it declines rather than quietly
        # trading quality the user did not offer. A plain fp8 checkpoint with no scales is unaffected:
        # its round trip through the compute dtype and back is exact.
        if dequantized > 0:
            if self._should_use_fp8(config, SubModelType.Transformer):
                logger.info(
                    f"FP8 Storage not applied to {model_path.name}: its {dequantized} scaled weight(s) were folded "
                    "with the per-tensor scales the file shipped, and re-encoding them as unscaled fp8 would lose "
                    "the low end of every tensor. Keeping them packed instead is the fix, and this loader does not "
                    "implement it yet."
                )
            return model

        # `assign=True` aliases every param to its `sd` tensor; drop the dict's references before the
        # FP8 cast or each original stays reachable while its fp8 copy is allocated (see `ltx2.py`,
        # which learned this the same way).
        sd.clear()

        # FP8 Storage, which this path ignored until now although the control was offered for it and
        # identification switched it on by itself for any float8 denoiser -- a 14B transformer loaded
        # at bf16 size while the record said it was halved.
        return self._apply_fp8_layerwise_casting(model, config, SubModelType.Transformer)


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.WanT5Encoder, format=ModelFormat.WanT5Encoder)
class WanT5EncoderLoader(ModelLoader):
    """Loader for the standalone Wan UMT5-XXL encoder.

    Accepts two on-disk layouts:
    1. Parent dir with ``text_encoder/`` (and typically ``tokenizer/``) subdirs —
       what ``Wan-AI/Wan2.2-T2V-A14B::text_encoder+tokenizer`` produces.
    2. A flat ``text_encoder/`` folder with ``config.json`` declaring
       ``model_type: umt5`` directly at the root. In this case the tokenizer
       is loaded from the same folder.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is None:
            raise ValueError("A submodel type (Tokenizer or TextEncoder) must be provided.")

        root = Path(config.path)
        nested_text_encoder = root / "text_encoder"
        nested_tokenizer = root / "tokenizer"

        if submodel_type == SubModelType.TextEncoder:
            from transformers import UMT5EncoderModel

            target = nested_text_encoder if nested_text_encoder.exists() else root
            return UMT5EncoderModel.from_pretrained(
                str(target),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        if submodel_type == SubModelType.Tokenizer:
            from invokeai.backend.wan.tokenizer import load_wan_tokenizer

            # Prefer a sibling tokenizer/ directory; fall back to the encoder dir
            # itself, which is normal for "flat" downloads.
            target = (
                nested_tokenizer
                if nested_tokenizer.exists()
                else (nested_text_encoder if nested_text_encoder.exists() else root)
            )
            return load_wan_tokenizer(target)

        raise ValueError(
            f"Unsupported submodel type for WanT5Encoder: {submodel_type.value if submodel_type else 'None'}"
        )


def _umt5_compute_dtype(device: torch.device) -> torch.dtype:
    """bfloat16, as the folder loader always uses, or float32 where the device lacks it -- never float16.

    UMT5's feed-forward activations overflow float16; transformers guards a float16 load by keeping
    ``wo`` in float32, which a packed ``wo`` cannot be.
    """
    dtype = TorchDevice.choose_bfloat16_safe_dtype(device)
    return torch.float32 if dtype == torch.float16 else dtype


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.WanT5Encoder, format=ModelFormat.GGUFQuantized)
class WanT5EncoderGGUFLoader(ModelLoader):
    """Loader for a UMT5-XXL encoder in one llama.cpp GGUF file (``city96/umt5-xxl-encoder-gguf``).

    Quantized Linear weights stay packed ``GGMLTensor``s, which the model cache's GGUF-aware layers
    dequantize per forward. Unpacked once at load is what a module cannot read packed -- the token
    embedding and every block's relative attention bias (``nn.Embedding`` indexes its weight; UMT5 has
    a bias per block where T5 v1.1 has one), and the RMS norms -- and Linear weights the file stores
    unquantized (F32/F16/BF16), which unpacking does not enlarge and which packed would be converted
    on every forward. The tokenizer is built from the file's metadata.
    """

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        # The tokenizer comes from the header, not from the tensors the file size counts.
        if submodel_type is SubModelType.Tokenizer:
            return 0
        return super().get_size_fs(config, model_path, submodel_type)

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, WanT5Encoder_GGUF_Config):
            raise TypeError(f"Expected WanT5Encoder_GGUF_Config, got {type(config).__name__}.")

        match submodel_type:
            case SubModelType.Tokenizer:
                return load_t5_tokenizer_from_gguf(Path(config.path))
            case SubModelType.TextEncoder:
                return self._load_text_encoder(config)

        raise ValueError(
            f"Unsupported submodel type for a GGUF Wan T5 encoder: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_text_encoder(self, config: WanT5Encoder_GGUF_Config) -> AnyModel:
        import accelerate
        from transformers import UMT5Config, UMT5EncoderModel

        compute_dtype = _umt5_compute_dtype(TorchDevice.choose_torch_device())
        sd = convert_llamacpp_t5_keys(gguf_sd_loader(Path(config.path), compute_dtype=compute_dtype))

        with accelerate.init_empty_weights():
            model = UMT5EncoderModel(infer_t5_encoder_config(sd, UMT5Config))

        # No model-declared skip patterns, unlike the transformer loaders: UMT5 declares
        # `_keep_in_fp32_modules = ["wo"]`, which would unpack every feed-forward output projection --
        # nearly a fifth of the encoder's weights. transformers keeps those in float32 only for a
        # float16 load, against an overflow this loader never computes in.
        source = "Wan UMT5 GGUF encoder"
        unpack_ggml_at_load(sd, model, (), self._ram_cache.make_room)
        load_state_dict_ignoring_extras(
            model, sd, source=source, assign=True, allowed_missing={"encoder.embed_tokens.weight"}
        )
        model.encoder.embed_tokens.weight = model.shared.weight

        make_feed_forward_gguf_safe(model)
        reject_incomplete_load(model, what=source)
        return model
