"""Loader registrations for Wan 2.2 image-generation models.

Currently covers:
- Main: Diffusers format (T2V-A14B with dual experts via Transformer +
  Transformer2 submodels, plus TI2V-5B).
- Main: GGUFQuantized and single-file Checkpoint (safetensors) transformers.
  Both are transformer-only — one file per A14B expert — and rely on a
  standalone VAE + T5 encoder for the rest of the pipeline.
- WanT5Encoder: standalone UMT5-XXL encoder folder (``text_encoder/`` +
  ``tokenizer/`` subdirs, or a flat ``text_encoder/`` folder).
- VAE: handled in ``vae.py`` (registered for type=VAE generically).
"""

from pathlib import Path
from typing import Any, Optional

import torch

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_Wan_Config,
    Main_GGUF_Wan_Config,
    _is_native_wan_layout,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.comfyui_state_dict_utils import (
    _dequantize_comfyui_fp8,
    _strip_comfyui_prefix,
    _strip_quantization_metadata,
)
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader
from invokeai.backend.quantization.gguf.utils import TORCH_COMPATIBLE_QTYPES
from invokeai.backend.util.devices import TorchDevice


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

        return result


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
        new_key = key
        for needle, replacement in _WAN_NATIVE_TO_DIFFUSERS_RENAMES:
            new_key = new_key.replace(needle, replacement)
        converted[new_key] = value
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


# Top-level modules that legitimately ride along in a single-file Wan checkpoint
# without being part of the transformer. Dropping these is correct: the pipeline
# sources its VAE and text encoder from separately-wired models, and the EMA copy
# is not the weight set we generate with.
#
# ``vae`` / ``text_encoders`` / ``clip`` / ``cond_stage_model`` / ``first_stage_model``
# are the "all-in-one" packaging convention — one file holding transformer + VAE +
# CLIP so ComfyUI's ``Load Checkpoint`` node can supply all three. The
# Phr00t/WAN2.2-14B-Rapid-AllInOne family and its ~110 GGUF conversions
# (befox/WAN2.2-14B-Rapid-AllInOne-GGUF) ship this way and loaded fine before the
# unexpected-key check existed, so refusing them would be a regression.
_BENIGN_EXTRA_MODULES = frozenset(
    {
        "vae",
        "first_stage_model",
        "text_encoders",
        "cond_stage_model",
        "clip",
        "model_ema",
    }
)

# Trailing segments marking a merged-in LoRA's leftover adapter tensors. The main-model
# probe deliberately admits checkpoints that retain these — see
# ``configs.main._has_wan_transformer_block_weights``, which uses a *positive* structural
# test precisely so merged-LoRA mains aren't turned away — so the loader has to admit
# them too, or the two halves disagree about the same file.
#
# Kept in step with the suffix set ``LoRA_LyCORIS_Wan_Config`` matches on
# (``configs/lora.py``): kohya, PEFT, DoRA and LoKr. Matched against the *last* path
# segment rather than as a substring anywhere in the key, so a future conditioning branch
# that merely contains "lora_a" in a module name still trips the backstop instead of
# being silently discarded by it.
_MERGED_LORA_SEGMENTS = frozenset(
    {
        "alpha",
        "dora_scale",
        "lora_magnitude_vector",
        "lokr_w1",
        "lokr_w2",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2_a",
        "lokr_w2_b",
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
        "oft_blocks",
    }
)
_MERGED_LORA_PENULTIMATE = frozenset({"lora_a", "lora_b", "lora_down", "lora_up", "lora_mid", "lora_magnitude"})


def _is_benign_extra_key(key: str) -> bool:
    """True if an unexpected key is packaging rather than an unsupported branch."""
    parts = key.lower().split(".")
    if parts[0] in _BENIGN_EXTRA_MODULES:
        return True
    if parts[-1] in _MERGED_LORA_SEGMENTS:
        return True
    # `...to_q.lora_down.weight` — the marker is the segment before the tensor name.
    return len(parts) >= 2 and parts[-2] in _MERGED_LORA_PENULTIMATE


def _drop_benign_extra_keys(sd: dict, source: str, logger: Any) -> None:
    """Remove packaging weights the transformer has no use for, in place.

    Done up front rather than left to ``load_state_dict(strict=False)`` because
    everything between here and there costs real memory: the fp8 dequant pass, the
    blanket cast to the compute dtype, and the RAM-cache reservation all run over the
    whole dict. An all-in-one checkpoint bundles a full VAE and UMT5-XXL text encoder —
    several GB, upcast to bf16 and reserved in the cache — only for
    ``load_state_dict`` to discard them one line later.
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
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if any(isinstance(k, str) and k.startswith(prefix) for k in sd.keys()):
                sd = {
                    (k[len(prefix) :] if isinstance(k, str) and k.startswith(prefix) else k): v for k, v in sd.items()
                }
                break

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


@ModelLoaderRegistry.register(base=BaseModelType.Wan, type=ModelType.Main, format=ModelFormat.Checkpoint)
class WanCheckpointModel(ModelLoader):
    """Loader for single-file Wan 2.2 transformer checkpoints (safetensors).

    This is what CivitAI fine-tunes and ComfyUI-oriented Hugging Face repos ship.
    Handles the full matrix of community conventions: the optional
    ``model.diffusion_model.`` key prefix, the native upstream key layout as well
    as the diffusers one, ComfyUI ``fp8_scaled`` weights (dequantized to the
    compute dtype at load time), and plain ``float8_e4m3fn`` weights with no
    scales (cast the same way as any other non-bf16 dtype).

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
        from safetensors.torch import load_file

        from invokeai.backend.util.logging import InvokeAILogger

        logger = InvokeAILogger.get_logger(self.__class__.__name__)

        model_path = Path(config.path)
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        sd = load_file(str(model_path))
        sd = _strip_comfyui_prefix(sd)
        _drop_benign_extra_keys(sd, "Wan checkpoint", logger)

        dequantized = _dequantize_comfyui_fp8(sd, model_dtype)
        if dequantized > 0:
            logger.info(f"Dequantized {dequantized} ComfyUI-quantized weights")
        # Drop the scale tensors themselves — they've been folded into the weights
        # above and are not parameters of WanTransformer3DModel. load_state_dict
        # runs with strict=False and would ignore them anyway, but dropping them
        # here keeps the dtype cast and the RAM-cache reservation below honest.
        _strip_quantization_metadata(sd)

        # Community releases ship the native upstream Wan key layout
        # (text_embedding.0, self_attn/cross_attn, ffn.0/2, head.head, ...);
        # diffusers' WanTransformer3DModel expects condition_embedder.*,
        # attn1/attn2, ffn.net.*, proj_out. Convert if needed.
        if _is_native_wan_layout(sd):
            sd = _convert_wan_native_to_diffusers(sd)

        model_config = _build_wan_transformer_config(sd, source="checkpoint state dict")

        with accelerate.init_empty_weights():
            model = WanTransformer3DModel(**model_config)

        # Cast every float tensor to the compute dtype. Dequantized fp8_scaled
        # weights are already there; this catches plain fp16/fp32/fp8 checkpoints
        # and makes the cache reservation below reflect the post-cast sizes.
        for key in list(sd.keys()):
            if sd[key].is_floating_point():
                sd[key] = sd[key].to(model_dtype)

        new_sd_size = sum(t.nelement() * t.element_size() for t in sd.values())
        self._ram_cache.make_room(new_sd_size)

        incompatible_keys = model.load_state_dict(sd, strict=False, assign=True)
        _raise_for_incompatible_keys(incompatible_keys, source="Wan checkpoint")
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.WanT5Encoder, format=ModelFormat.WanT5Encoder)
class WanT5EncoderLoader(ModelLoader):
    """Loader for the standalone Wan UMT5-XXL encoder.

    Accepts two on-disk layouts:
    1. Parent dir with ``text_encoder/`` (and typically ``tokenizer/``) subdirs —
       what ``Wan-AI/Wan2.2-T2V-A14B::text_encoder+tokenizer`` produces.
    2. A flat ``text_encoder/`` folder with ``config.json`` declaring
       ``model_type: umt5`` directly at the root. In this case the tokenizer
       is loaded from the same folder via ``AutoTokenizer.from_pretrained``.
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
            from transformers import AutoTokenizer

            # Prefer a sibling tokenizer/ directory; fall back to the encoder dir
            # itself, which is normal for "flat" downloads.
            target = (
                nested_tokenizer
                if nested_tokenizer.exists()
                else (nested_text_encoder if nested_text_encoder.exists() else root)
            )
            return AutoTokenizer.from_pretrained(str(target), local_files_only=True)

        raise ValueError(
            f"Unsupported submodel type for WanT5Encoder: {submodel_type.value if submodel_type else 'None'}"
        )
