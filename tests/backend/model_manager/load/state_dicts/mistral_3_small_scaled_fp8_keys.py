"""Representative key layout of the scaled-fp8 Mistral-3-Small encoder FLUX.2 ships.

Captured on 2026-09-26 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`Comfy-Org/flux2-dev/split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors`.

Subsetting rule: `model.layers` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Two things make this the encoder cell rather than a second transformer one.

It arrives already normalised: no `text_encoder.`/`language_model.` wrapper, so every key is the
plain `model.*` form that `_strip_known_prefixes` exists to produce, and `model.` itself comes off
later in `_convert_for_bare_mistral_model` because bare `MistralModel` has no LM head and no `model.`
level. The header names its layers in that same namespace, which is the agreement the hint re-key
depends on -- `MISTRAL_KEY_PREFIXES` is a named constant precisely because the header is written
*before* the strip, so a second copy of the list could drift out of step with it.

And it is the one captured layout that ships the tokenizer: `tekken_model`, 19 MB of it, as a uint8
tensor sitting beside the weights, next to a zero-element `scaled_fp8` marker that names no layer.
Both have to be told apart from a scale key -- one is metadata to strip, the other is the tokenizer.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "model.embed_tokens.weight": ([131072, 5120], "BF16"),
    "model.layers.0.input_layernorm.weight": ([5120], "BF16"),
    "model.layers.0.mlp.down_proj.input_scale": ([], "F32"),
    "model.layers.0.mlp.down_proj.weight": ([5120, 32768], "F8_E4M3"),
    "model.layers.0.mlp.down_proj.weight_scale": ([], "F32"),
    "model.layers.0.mlp.gate_proj.input_scale": ([], "F32"),
    "model.layers.0.mlp.gate_proj.weight": ([32768, 5120], "F8_E4M3"),
    "model.layers.0.mlp.gate_proj.weight_scale": ([], "F32"),
    "model.layers.0.mlp.up_proj.input_scale": ([], "F32"),
    "model.layers.0.mlp.up_proj.weight": ([32768, 5120], "F8_E4M3"),
    "model.layers.0.mlp.up_proj.weight_scale": ([], "F32"),
    "model.layers.0.post_attention_layernorm.weight": ([5120], "BF16"),
    "model.layers.0.self_attn.k_proj.input_scale": ([], "F32"),
    "model.layers.0.self_attn.k_proj.weight": ([1024, 5120], "F8_E4M3"),
    "model.layers.0.self_attn.k_proj.weight_scale": ([], "F32"),
    "model.layers.0.self_attn.o_proj.input_scale": ([], "F32"),
    "model.layers.0.self_attn.o_proj.weight": ([5120, 4096], "F8_E4M3"),
    "model.layers.0.self_attn.o_proj.weight_scale": ([], "F32"),
    "model.layers.0.self_attn.q_proj.input_scale": ([], "F32"),
    "model.layers.0.self_attn.q_proj.weight": ([4096, 5120], "F8_E4M3"),
    "model.layers.0.self_attn.q_proj.weight_scale": ([], "F32"),
    "model.layers.0.self_attn.v_proj.input_scale": ([], "F32"),
    "model.layers.0.self_attn.v_proj.weight": ([1024, 5120], "F8_E4M3"),
    "model.layers.0.self_attn.v_proj.weight_scale": ([], "F32"),
    "scaled_fp8": ([0], "F8_E4M3"),
    "tekken_model": ([19399895], "U8"),
}

# The `_quantization_metadata` header block, layer names exactly as the producer wrote them.
layer_hints: dict[str, dict[str, object]] = {
    "model.layers.0.mlp.down_proj": {"format": "float8_e4m3fn"},
    "model.layers.0.mlp.gate_proj": {"format": "float8_e4m3fn"},
    "model.layers.0.mlp.up_proj": {"format": "float8_e4m3fn"},
    "model.layers.0.self_attn.k_proj": {"format": "float8_e4m3fn"},
    "model.layers.0.self_attn.o_proj": {"format": "float8_e4m3fn"},
    "model.layers.0.self_attn.q_proj": {"format": "float8_e4m3fn"},
    "model.layers.0.self_attn.v_proj": {"format": "float8_e4m3fn"},
}
