"""Loader registrations for Wan 2.2 image-generation models.

Currently covers:
- Main: Diffusers format (T2V-A14B with dual experts via Transformer +
  Transformer2 submodels, plus TI2V-5B). Phase 4 will add a GGUFQuantized loader.
- WanT5Encoder: standalone UMT5-XXL encoder folder (``text_encoder/`` +
  ``tokenizer/`` subdirs, or a flat ``text_encoder/`` folder).
- VAE: handled in ``vae.py`` (registered for type=VAE generically).
"""

from pathlib import Path
from typing import Optional

import torch

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_GGUF_Wan_Config, _is_native_wan_layout
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
    WanVariantType,
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
            raise NotImplementedError("Single-file checkpoint format is not yet supported for Wan models.")

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

        # Auto-detect architecture from the state dict.
        num_layers = 0
        for key in sd.keys():
            if isinstance(key, str) and key.startswith("blocks."):
                parts = key.split(".")
                if len(parts) >= 2:
                    try:
                        num_layers = max(num_layers, int(parts[1]) + 1)
                    except ValueError:
                        pass

        # Patch embedding gives us in_channels (16=A14B, 48=TI2V-5B) and inner dim.
        patch_w = sd.get("patch_embedding.weight")
        if patch_w is None:
            raise RuntimeError("GGUF state dict missing patch_embedding.weight after prefix strip")
        patch_shape = patch_w.tensor_shape if isinstance(patch_w, GGMLTensor) else patch_w.shape
        inner_dim = int(patch_shape[0])
        in_channels = int(patch_shape[1])

        # Wan uses head_dim=128 throughout the family; num_heads = inner_dim / 128.
        attention_head_dim = 128
        num_attention_heads = inner_dim // attention_head_dim

        ffn_w = sd.get("blocks.0.ffn.net.0.proj.weight")
        if ffn_w is None:
            raise RuntimeError("GGUF state dict missing blocks.0.ffn.net.0.proj.weight after prefix strip")
        ffn_shape = ffn_w.tensor_shape if isinstance(ffn_w, GGMLTensor) else ffn_w.shape
        ffn_dim = int(ffn_shape[0])

        text_w = sd.get("condition_embedder.text_embedder.linear_1.weight")
        text_dim = 4096
        if text_w is not None:
            text_shape = text_w.tensor_shape if isinstance(text_w, GGMLTensor) else text_w.shape
            text_dim = int(text_shape[1])

        # out_channels is read from proj_out.weight directly rather than assumed
        # equal to in_channels: I2V-A14B has in_channels=36 (16 noise + 16
        # ref-image latents + 4 mask, concatenated by the denoise loop) but
        # out_channels=16 (only the noise prediction comes back). proj_out is
        # ``nn.Linear(inner_dim, out_channels * prod(patch_size))`` and
        # patch_size is (1, 2, 2) → prod = 4 for the Wan 2.2 family.
        proj_out_w = sd.get("proj_out.weight")
        if proj_out_w is None:
            raise RuntimeError("GGUF state dict missing proj_out.weight after prefix strip")
        proj_out_shape = proj_out_w.tensor_shape if isinstance(proj_out_w, GGMLTensor) else proj_out_w.shape
        out_channels = int(proj_out_shape[0]) // 4

        # Layer count fallback (only triggers if the auto-count loop above
        # found zero blocks, which shouldn't happen for a valid GGUF). T2V/I2V
        # A14B have 40 layers; TI2V-5B has 30.
        layer_count_fallback = 30 if config.variant == WanVariantType.TI2V_5B else 40

        model_config: dict = {
            "patch_size": (1, 2, 2),
            "in_channels": in_channels,
            "out_channels": out_channels,
            "num_layers": num_layers if num_layers > 0 else layer_count_fallback,
            "attention_head_dim": attention_head_dim,
            "num_attention_heads": num_attention_heads,
            "ffn_dim": ffn_dim,
            "text_dim": text_dim,
        }

        with accelerate.init_empty_weights():
            model = WanTransformer3DModel(**model_config)

        model.load_state_dict(sd, strict=False, assign=True)
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
