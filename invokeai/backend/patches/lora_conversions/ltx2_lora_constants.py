# LTX-2 LoRA prefix constants and key-shape detection helpers.
#
# LTX-2 LoRAs are published against the official Lightricks single-file transformer
# layout with plain PEFT ``lora_A`` / ``lora_B`` suffixes and no ``.alpha`` tensors
# (alpha == rank by convention, i.e. ``W_eff = W + lora_B @ lora_A`` with no extra
# scaling). Measured on the 2.5 distilled accelerator (1660 layers, rank 450):
#
#     diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight
#     diffusion_model.transformer_blocks.0.audio_attn1.to_q.lora_A.weight
#     diffusion_model.transformer_blocks.0.audio_to_video_attn.to_k.lora_A.weight
#     diffusion_model.transformer_blocks.0.video_to_audio_attn.to_v.lora_A.weight
#     diffusion_model.av_ca_a2v_gate_adaln_single.linear.lora_A.weight
#     diffusion_model.adaln_single.emb.timestep_embedder.linear_1.lora_A.weight
#     diffusion_model.patchify_proj.lora_A.weight
#
# The detection helpers below are shared with ``configs/lora.py`` so the probe and the
# conversion code agree on what counts as an LTX-2 LoRA. They keep this file
# circular-import-free.

import re

# Prefix for LTX-2 transformer LoRA layers in the ModelPatchRaw layer dict. Same convention
# as Wan / H3 / Anima / QwenImage — the LayerPatcher uses this prefix to resolve patches
# against the loaded transformer's parameter paths.
LTX2_LORA_TRANSFORMER_PREFIX = "lora_transformer-"

# Optional prefixes seen on PEFT-style keys. Anchored to the key start so detection admits
# exactly what the converter's single prefix-strip can handle — an unanchored alternative
# would also match nested prefixes like ``diffusion_model.transformer.blocks...``, which the
# probe would then admit but the converter would map onto nonexistent module paths (the LoRA
# would silently apply zero layers).
_PEFT_PREFIX_RE = r"^(?:(?:model\.diffusion_model|diffusion_model|transformer|base_model\.model(?:\.transformer)?)\.)?"

# Submodules unique to LTX-2 among the supported architectures, in the checkpoint's native
# naming. All of them exist only because LTX-2 is a *dual-stream* video+audio transformer:
# every block carries a second audio attention/feed-forward tower plus two cross-modal
# attentions, and the modulation parameters that gate them are named per direction.
#
# Deliberately NOT used as markers:
# - ``transformer_blocks.\d+.attn1`` alone — Wan's diffusers naming and QwenImage share it;
# - ``patchify_proj`` alone — earlier LTX-Video releases use it too, so it identifies the
#   family but not this architecture.
# Requiring an audio-side or cross-modal marker is what makes this exclusive.
#
# A LoRA without such a marker (video stream only) is deliberately not identified as LTX-2, even though the
# video stream is 4096 wide: LTX-Video 13B (``LTXVideoTransformer3DModel``: 48 blocks, 32 heads of 128, 4096-wide
# text context) names and shapes ``transformer_blocks.N.attn1/attn2.to_q/to_k/to_v/to_out.0`` identically, as do
# Cosmos-Predict1 7B LoRAs, so no shape separates them and a wrong claim would apply silently to the wrong model.
_LTX2_BLOCK_RE = re.compile(
    _PEFT_PREFIX_RE + r"transformer_blocks\.\d+\.(?:audio_attn[12]|audio_to_video_attn|video_to_audio_attn|audio_ff)\."
)
# The cross-modal modulation heads, which live at the top level rather than per block.
_LTX2_CROSS_MODAL_RE = re.compile(
    _PEFT_PREFIX_RE + r"(?:av_ca_(?:a2v_gate|v2a_gate|audio_scale_shift|video_scale_shift)_adaln_single|"
    r"audio_adaln_single|audio_prompt_adaln_single|audio_patchify_proj|audio_proj_out)\."
)
# The same heads under the diffusers ``LTX2VideoTransformer3DModel`` names, as a PEFT ``PeftModel``
# around that class saves them. The runtime transformer uses these names, so the converter must carry
# them through untouched. ``time_embed`` / ``proj_in`` / ``prompt_adaln`` are accepted the same way
# but are not markers: LTX-Video's diffusers transformer has ``time_embed`` and ``proj_in`` too.
_LTX2_DIFFUSERS_TOP_LEVEL_MARKER_RE = re.compile(
    _PEFT_PREFIX_RE + r"(?:av_cross_attn_(?:video_scale_shift|video_a2v_gate|audio_scale_shift|audio_v2a_gate)|"
    r"audio_time_embed|audio_prompt_adaln|audio_proj_in)\."
)
# Every top-level module the converter must leave alone, matched against prefix-stripped layer paths
# (``proj_in`` itself is a layer, so this compares the first path component rather than a ``name.`` prefix).
_LTX2_DIFFUSERS_TOP_LEVEL_MODULES = frozenset(
    {
        "time_embed",
        "audio_time_embed",
        "prompt_adaln",
        "audio_prompt_adaln",
        "proj_in",
        "audio_proj_in",
        "av_cross_attn_video_scale_shift",
        "av_cross_attn_video_a2v_gate",
        "av_cross_attn_audio_scale_shift",
        "av_cross_attn_audio_v2a_gate",
    }
)

# Any of these indicates a different architecture; an LTX-2 LoRA never carries them.
#
# Note what is absent: ``transformer_blocks\.\d+`` cannot be an anti-pattern here because
# LTX-2 uses it natively. QwenImage is excluded by its own modulation/MLP naming instead.
_NON_LTX2_ANTI_RES = (
    # Wan / Anima attention naming; LTX-2's diffusers ``av_cross_attn_*`` heads are not it.
    re.compile(r"(self_attn|(?<!av_)cross_attn)[\._]"),
    re.compile(r"blocks\.\d+\.(?:attn\.qkv_proj|adaln_proj\.linear)"),  # MiniMax H3 fused attention
    re.compile(r"final_layer\.adaln_proj\.linear"),  # MiniMax H3 final layer
    re.compile(r"(^|\.|_)(double_blocks|single_blocks|single_transformer_blocks)[\._]\d+"),  # FLUX
    re.compile(r"(img_mod|txt_mod|img_mlp|txt_mlp|img_attn|txt_attn)[\._]"),  # QwenImage / FLUX-style
    re.compile(r"diffusion_model\.layers\.\d+\."),  # Z-Image
    re.compile(r"mlp[\._]layer|adaln_modulation"),  # Anima / Cosmos
)

# LyCORIS variants the LTX-2 pipeline does NOT support. The conversion maps official layer
# paths onto diffusers module paths one-for-one and hands the LayerPatcher an ordinary
# low-rank pair; LoKR/LoHA factorizations and DoRA's per-row magnitudes are a different patch
# shape entirely. Rejecting them at probe time turns a would-be generation-time failure into
# a clear install-time "not supported".
_UNSUPPORTED_LTX2_VARIANT_SUFFIXES = (
    ".lokr_w1",
    ".lokr_w2",
    ".lokr_w1_a",
    ".lokr_w2_a",
    ".hada_w1_a",
    ".hada_w2_a",
    # Both DoRA spellings. LTX-2 LoRAs are published in the PEFT layout, so the PEFT/ai-toolkit
    # magnitude names are the ones this family will actually meet -- and DoRA is the only variant
    # that reaches this guard at all, because it carries `lora_A`/`lora_B` and so passes the generic
    # "has a LoRA suffix" test that LoKR and LoHA files fail on their own. Checking only the kohya
    # spelling let a DoRA file install and then fail inside the denoise, after the 22B transformer
    # had loaded -- exactly the deferral this guard exists to prevent.
    ".dora_scale",
    ".lora_magnitude_vector.weight",
    ".magnitude",
)


def has_unsupported_ltx2_lora_variant_keys(str_keys: list[str]) -> bool:
    """True if the state dict carries LyCORIS-variant tensors (LoKR/LoHA/DoRA) that the LTX-2
    conversion pipeline cannot apply."""
    return any(k.endswith(suffix) for k in str_keys for suffix in _UNSUPPORTED_LTX2_VARIANT_SUFFIXES)


def has_ltx2_lora_keys(str_keys: list[str]) -> bool:
    """PEFT-style keys naming LTX-2-exclusive submodules in the checkpoint's native or diffusers layout.

    Either an audio-side / cross-modal tower inside a block, or one of the top-level
    cross-modal modulation heads under either naming. Both exist only on a dual-stream video+audio transformer.
    """
    return any(
        _LTX2_BLOCK_RE.search(k) or _LTX2_CROSS_MODAL_RE.search(k) or _LTX2_DIFFUSERS_TOP_LEVEL_MARKER_RE.search(k)
        for k in str_keys
    )


def is_ltx2_diffusers_top_level_path(layer_path: str) -> bool:
    """True if a prefix-stripped layer path already names a top-level module in diffusers form."""
    return layer_path.split(".", 1)[0] in _LTX2_DIFFUSERS_TOP_LEVEL_MODULES


def has_non_ltx2_architecture_keys(str_keys: list[str]) -> bool:
    """True if any key indicates a non-LTX-2 architecture (Wan, H3, FLUX, QwenImage, Z-Image,
    Anima).

    Used as an exclusion guard — an LTX-2 LoRA never carries these.
    """
    return any(anti.search(k) for k in str_keys for anti in _NON_LTX2_ANTI_RES)
