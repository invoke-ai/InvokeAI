# Wan 2.2 LoRA prefix constants and key-shape detection helpers.
#
# Wan LoRAs come in three shapes in the wild:
#
# 1. **Diffusers PEFT** (HF naming), with or without a "transformer." prefix:
#      blocks.0.attn1.to_q.lora_A.weight
#      transformer.blocks.0.attn1.to_q.lora_A.weight
#
# 2. **Native upstream PEFT** (ComfyUI / Wan-AI checkpoint naming) with
#    "diffusion_model." or "transformer." prefix:
#      diffusion_model.blocks.0.self_attn.q.lora_A.weight
#      transformer.blocks.0.cross_attn.k.lora_A.weight
#
# 3. **Kohya**, with the standard ``lora_unet_blocks_<idx>_<submodule>`` shape,
#    in either diffusers naming (``attn1_to_q``) or native naming (``self_attn_q``):
#      lora_unet_blocks_0_attn1_to_q.lora_down.weight
#      lora_unet_blocks_0_self_attn_q.lora_down.weight
#
# The detection helpers below are shared with ``configs/lora.py`` so the probe
# and the conversion code agree on what counts as a Wan LoRA. They keep this
# file circular-import-free.

import re

from invokeai.backend.model_manager.taxonomy import WanLoRAVariantType

# Prefix for Wan transformer LoRA layers in the ModelPatchRaw layer dict.
# Same convention as Anima / QwenImage — the LayerPatcher uses this prefix to
# resolve patches against the loaded transformer's parameter paths.
WAN_LORA_TRANSFORMER_PREFIX = "lora_transformer-"


# Diffusers Wan-specific submodules: attn1/attn2 (self/cross attention with
# to_q/to_k/to_v/to_out.0 children) and ffn.net (gated FFN). These are unique
# to WanTransformer3DModel — none of FLUX (double_blocks/single_blocks),
# QwenImage (transformer_blocks.X.attn), Z-Image (diffusion_model.layers),
# or Anima/Cosmos (mlp + adaln_modulation) produce this combination.
_WAN_DIFFUSERS_SUBMODULES = r"(attn1\.|attn2\.|ffn\.net\.)"

# Native upstream Wan submodules. self_attn / cross_attn collide with Anima's
# Cosmos DiT naming, so we look for the bare ``.q``/``.k``/``.v``/``.o``
# projection suffix (no ``_proj`` tail) AND/OR the ``ffn.<digit>`` MLP layout —
# Anima uses ``mlp`` instead, so this is mutually exclusive.
_WAN_NATIVE_SUBMODULES = r"(self_attn\.[qkvo](\.|$)|cross_attn\.[qkvo](\.|$)|ffn\.\d+\.)"

# Anti-patterns: keys that would indicate Anima/Cosmos (mlp / adaln_modulation /
# the ``q_proj`` projection naming Cosmos uses on its attention blocks),
# QwenImage (transformer_blocks), Flux (double_blocks / single_blocks), or
# Z-Image (diffusion_model.layers). If any of these are present, the LoRA is
# NOT Wan.
_ANIMA_ANTI_RE = re.compile(r"blocks[\._]\d+[\._](mlp|adaln_modulation)")
# Anima Cosmos attention uses ``q_proj`` / ``k_proj`` / ``v_proj`` / ``output_proj``
# under self_attn/cross_attn. Wan native uses just ``q``/``k``/``v``/``o`` — so
# the ``_proj`` suffix on a self_attn/cross_attn child is a definitive Anima tell,
# in both Kohya (``self_attn_q_proj``) and PEFT (``self_attn.q_proj``) forms.
_ANIMA_ATTN_ANTI_RE = re.compile(r"(self_attn|cross_attn)[\._]([qkv]_proj|output_proj)")
_QWEN_ANTI_RE = re.compile(r"(^|\.)transformer_blocks\.\d+\.")
_FLUX_ANTI_RE = re.compile(r"(^|\.|_)(double_blocks|single_blocks|single_transformer_blocks)[\._]\d+")
_Z_IMAGE_ANTI_RE = re.compile(r"diffusion_model\.layers\.\d+\.")


# Kohya format: lora_unet_blocks_<idx>_(attn1_to_X | ffn_N | (self|cross)_attn_X
# where X is a single q/k/v/o letter). The strict alphabet on the attention
# child keeps us from matching Anima's ``cross_attn_q_proj`` (which has an
# additional ``_proj`` segment).
_KOHYA_WAN_RE = re.compile(
    r"lora_unet_blocks_\d+_"
    r"(attn[12]_(to_[qkv]|to_out_0|norm_[qk])"
    r"|(self_attn|cross_attn)_[qkvo](_|\.|$)"
    r"|ffn_(\d+|net_\d+_proj|net_\d+))"
)

# PEFT format: <prefix>.blocks.<idx>.<wan_submodule>
# Prefix may be empty, "transformer.", "diffusion_model.", or "base_model.model.transformer."
_PEFT_WAN_DIFFUSERS_RE = re.compile(
    r"(?:^|(?:diffusion_model|transformer|base_model\.model\.transformer)\.)blocks\.\d+\."
    + _WAN_DIFFUSERS_SUBMODULES
)
_PEFT_WAN_NATIVE_RE = re.compile(
    r"(?:^|(?:diffusion_model|transformer|base_model\.model\.transformer)\.)blocks\.\d+\."
    + _WAN_NATIVE_SUBMODULES
)


def has_wan_kohya_keys(str_keys: list[str]) -> bool:
    """Kohya-style keys naming Wan submodules (attn1/attn2/self_attn/cross_attn/ffn)."""
    return any(_KOHYA_WAN_RE.search(k) is not None for k in str_keys)


def has_wan_peft_keys(str_keys: list[str]) -> bool:
    """Diffusers PEFT keys naming Wan submodules in either diffusers or native layout."""
    for k in str_keys:
        if _PEFT_WAN_DIFFUSERS_RE.search(k) is not None:
            return True
        if _PEFT_WAN_NATIVE_RE.search(k) is not None:
            return True
    return False


def detect_wan_lora_variant(state_dict: dict) -> WanLoRAVariantType | None:
    """Inspect a Wan LoRA state dict and guess which model family it targets.

    A14B has inner_dim=5120; TI2V-5B has inner_dim=3072. Every transformer
    block's ``attn1.to_q`` (or native ``self_attn.q``) LoRA pair has weights
    shaped against the inner dim — ``lora_up.weight`` is ``[inner_dim, rank]``
    and ``lora_down.weight`` is ``[rank, inner_dim]``. The larger dim of
    either is the inner dim.

    Returns:
        ``WanLoRAVariantType.A14B`` if inner_dim == 5120,
        ``WanLoRAVariantType.Wan5B`` if inner_dim == 3072,
        ``None`` if no recognisable attn weight is found or inner_dim is
        ambiguous (e.g. LoRA that only patches FFN at non-standard rank).
    """
    # Probe several common key shapes — diffusers PEFT (lora_A/lora_B),
    # native Kohya naming (lora_up/lora_down), with or without a
    # diffusion_model/transformer prefix, in diffusers or native attn
    # naming. The first matching tensor is enough.
    candidate_suffixes = (
        # diffusers PEFT
        ".attn1.to_q.lora_A.weight",
        ".attn1.to_q.lora_B.weight",
        ".self_attn.q.lora_A.weight",
        ".self_attn.q.lora_B.weight",
        # native (Kohya) PEFT
        ".attn1.to_q.lora_up.weight",
        ".attn1.to_q.lora_down.weight",
        ".self_attn.q.lora_up.weight",
        ".self_attn.q.lora_down.weight",
    )
    kohya_substrings = (
        "_attn1_to_q.lora_up.weight",
        "_attn1_to_q.lora_down.weight",
        "_self_attn_q.lora_up.weight",
        "_self_attn_q.lora_down.weight",
    )

    for key, tensor in state_dict.items():
        if not isinstance(key, str):
            continue
        match_suffix = any(key.endswith(suffix) for suffix in candidate_suffixes)
        match_kohya = any(needle in key for needle in kohya_substrings)
        if not (match_suffix or match_kohya):
            continue
        shape = getattr(tensor, "shape", None)
        if shape is None or len(shape) < 2:
            continue
        inner_dim = max(int(shape[0]), int(shape[1]))
        if inner_dim == 5120:
            return WanLoRAVariantType.A14B
        if inner_dim == 3072:
            return WanLoRAVariantType.Wan5B
        # Any other inner_dim is uncharted — bail rather than guess.
        return None

    return None


def has_non_wan_architecture_keys(str_keys: list[str]) -> bool:
    """True if any key indicates a non-Wan architecture (Anima, Qwen, Flux, Z-Image).

    Used as an exclusion guard — a Wan LoRA should never carry these patterns,
    so finding them is grounds to reject the Wan probe.
    """
    for k in str_keys:
        if _ANIMA_ANTI_RE.search(k) is not None:
            return True
        if _ANIMA_ATTN_ANTI_RE.search(k) is not None:
            return True
        if _QWEN_ANTI_RE.search(k) is not None:
            return True
        if _FLUX_ANTI_RE.search(k) is not None:
            return True
        if _Z_IMAGE_ANTI_RE.search(k) is not None:
            return True
    return False
