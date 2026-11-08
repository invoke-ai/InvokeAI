"""Representative key layout of Comfy-Org's repackaged scaled-fp8 Wan 2.2.

Captured on 2026-09-26 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`Comfy-Org/Wan_2.2_ComfyUI_Repackaged/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors`.

Subsetting rule: `blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Captured over an HTTP range read. The file is ~14 GB and no Wan fp8 build is on this machine, which
is the case the capture tool exists for.

**What makes it worth capturing is the native key names.** Three Wan probes read them --
`_is_native_wan_layout` decides whether the diffusers conversion runs, `_detect_wan_variant_from_state_dict`
reads `patch_embedding.weight`'s channel count and width to pick T2V-A14B over I2V-A14B over TI2V-5B,
and `_has_wan_transformer_block_weights` separates a main model from an I2V LoRA that ships its own
`patch_embedding`. Until this capture all three were exercised only against a synthetic dict written
from the same understanding as the probes themselves.

It also declares itself through **neither** hint transport: no `_quantization_metadata` header block
and no `.comfy_quant` markers, only the bare `scaled_fp8` marker tensor and the sibling scale keys.
`WanCheckpointModel` therefore enters the fold structurally, on the key names alone. (Neither
transport is the common case among these fixtures, not a peculiarity of this one.)

The scales are spelled `.scale_weight`/`.scale_input` rather than `.weight_scale`/`.input_scale`.
That is *not* new here -- `flux1_transformer_scaled_fp8_keys.py` records the same spelling and
`test_flux1_scaled_fp8_keys.py` already drives a reader over it -- so no cell in this file claims it.

Every scale is per-tensor (0-d), so this layout does *not* exercise the per-output-channel axis that
`expand_weight_scale` gets right and the copy it replaced got wrong -- the FLUX.1 int8 fixture beside
this one is where that lives.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "blocks.0.cross_attn.k.bias": ([5120], "F16"),
    "blocks.0.cross_attn.k.scale_input": ([], "F32"),
    "blocks.0.cross_attn.k.scale_weight": ([], "F32"),
    "blocks.0.cross_attn.k.weight": ([5120, 5120], "F8_E4M3"),
    "blocks.0.cross_attn.norm_k.weight": ([5120], "F16"),
    "blocks.0.cross_attn.norm_q.weight": ([5120], "F16"),
    "blocks.0.cross_attn.o.bias": ([5120], "F16"),
    "blocks.0.cross_attn.o.scale_input": ([], "F32"),
    "blocks.0.cross_attn.o.scale_weight": ([], "F32"),
    "blocks.0.cross_attn.o.weight": ([5120, 5120], "F8_E4M3"),
    "blocks.0.cross_attn.q.bias": ([5120], "F16"),
    "blocks.0.cross_attn.q.scale_input": ([], "F32"),
    "blocks.0.cross_attn.q.scale_weight": ([], "F32"),
    "blocks.0.cross_attn.q.weight": ([5120, 5120], "F8_E4M3"),
    "blocks.0.cross_attn.v.bias": ([5120], "F16"),
    "blocks.0.cross_attn.v.scale_input": ([], "F32"),
    "blocks.0.cross_attn.v.scale_weight": ([], "F32"),
    "blocks.0.cross_attn.v.weight": ([5120, 5120], "F8_E4M3"),
    "blocks.0.ffn.0.bias": ([13824], "F16"),
    "blocks.0.ffn.0.scale_input": ([], "F32"),
    "blocks.0.ffn.0.scale_weight": ([], "F32"),
    "blocks.0.ffn.0.weight": ([13824, 5120], "F8_E4M3"),
    "blocks.0.ffn.2.bias": ([5120], "F16"),
    "blocks.0.ffn.2.scale_input": ([], "F32"),
    "blocks.0.ffn.2.scale_weight": ([], "F32"),
    "blocks.0.ffn.2.weight": ([5120, 13824], "F8_E4M3"),
    "blocks.0.modulation": ([1, 6, 5120], "F16"),
    "blocks.0.norm3.bias": ([5120], "F16"),
    "blocks.0.norm3.weight": ([5120], "F16"),
    "blocks.0.self_attn.k.bias": ([5120], "F16"),
    "blocks.0.self_attn.k.scale_input": ([], "F32"),
    "blocks.0.self_attn.k.scale_weight": ([], "F32"),
    "blocks.0.self_attn.k.weight": ([5120, 5120], "F8_E4M3"),
    "blocks.0.self_attn.norm_k.weight": ([5120], "F16"),
    "blocks.0.self_attn.norm_q.weight": ([5120], "F16"),
    "blocks.0.self_attn.o.bias": ([5120], "F16"),
    "blocks.0.self_attn.o.scale_input": ([], "F32"),
    "blocks.0.self_attn.o.scale_weight": ([], "F32"),
    "blocks.0.self_attn.o.weight": ([5120, 5120], "F8_E4M3"),
    "blocks.0.self_attn.q.bias": ([5120], "F16"),
    "blocks.0.self_attn.q.scale_input": ([], "F32"),
    "blocks.0.self_attn.q.scale_weight": ([], "F32"),
    "blocks.0.self_attn.q.weight": ([5120, 5120], "F8_E4M3"),
    "blocks.0.self_attn.v.bias": ([5120], "F16"),
    "blocks.0.self_attn.v.scale_input": ([], "F32"),
    "blocks.0.self_attn.v.scale_weight": ([], "F32"),
    "blocks.0.self_attn.v.weight": ([5120, 5120], "F8_E4M3"),
    "head.head.bias": ([64], "F16"),
    "head.head.scale_input": ([], "F32"),
    "head.head.scale_weight": ([], "F32"),
    "head.head.weight": ([64, 5120], "F8_E4M3"),
    "head.modulation": ([1, 2, 5120], "F16"),
    "patch_embedding.bias": ([5120], "F16"),
    "patch_embedding.weight": ([5120, 16, 1, 2, 2], "F16"),
    "scaled_fp8": ([0], "F8_E4M3"),
    "text_embedding.0.bias": ([5120], "F16"),
    "text_embedding.0.scale_input": ([], "F32"),
    "text_embedding.0.scale_weight": ([], "F32"),
    "text_embedding.0.weight": ([5120, 4096], "F8_E4M3"),
    "text_embedding.2.bias": ([5120], "F16"),
    "text_embedding.2.scale_input": ([], "F32"),
    "text_embedding.2.scale_weight": ([], "F32"),
    "text_embedding.2.weight": ([5120, 5120], "F8_E4M3"),
    "time_embedding.0.bias": ([5120], "F16"),
    "time_embedding.0.scale_input": ([], "F32"),
    "time_embedding.0.scale_weight": ([], "F32"),
    "time_embedding.0.weight": ([5120, 256], "F8_E4M3"),
    "time_embedding.2.bias": ([5120], "F16"),
    "time_embedding.2.scale_input": ([], "F32"),
    "time_embedding.2.scale_weight": ([], "F32"),
    "time_embedding.2.weight": ([5120, 5120], "F8_E4M3"),
    "time_projection.1.bias": ([30720], "F16"),
    "time_projection.1.scale_input": ([], "F32"),
    "time_projection.1.scale_weight": ([], "F32"),
    "time_projection.1.weight": ([30720, 5120], "F8_E4M3"),
}
