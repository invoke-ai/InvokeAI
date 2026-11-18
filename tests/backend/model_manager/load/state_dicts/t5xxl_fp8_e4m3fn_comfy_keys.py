"""Key layout of ComfyUI's raw fp8 T5-XXL encoder (`t5xxl_fp8_e4m3fn`).

Captured on 2026-10-11 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`comfyanonymous/flux_text_encoders/t5xxl_fp8_e4m3fn.safetensors`.

Subsetting rule: `encoder.block` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

The fp16 layout with *every* tensor cast to float8 and no scale anywhere -- the layer norms, the
position bias and both embeddings included. Only the Linear weights may stay fp8 at load; the rest
have to be widened, or the first norm hands fp8 activations to the next layer.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "encoder.block.0.layer.0.SelfAttention.k.weight": ([4096, 4096], "F8_E4M3"),
    "encoder.block.0.layer.0.SelfAttention.o.weight": ([4096, 4096], "F8_E4M3"),
    "encoder.block.0.layer.0.SelfAttention.q.weight": ([4096, 4096], "F8_E4M3"),
    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": ([32, 64], "F8_E4M3"),
    "encoder.block.0.layer.0.SelfAttention.v.weight": ([4096, 4096], "F8_E4M3"),
    "encoder.block.0.layer.0.layer_norm.weight": ([4096], "F8_E4M3"),
    "encoder.block.0.layer.1.DenseReluDense.wi_0.weight": ([10240, 4096], "F8_E4M3"),
    "encoder.block.0.layer.1.DenseReluDense.wi_1.weight": ([10240, 4096], "F8_E4M3"),
    "encoder.block.0.layer.1.DenseReluDense.wo.weight": ([4096, 10240], "F8_E4M3"),
    "encoder.block.0.layer.1.layer_norm.weight": ([4096], "F8_E4M3"),
    "encoder.embed_tokens.weight": ([32128, 4096], "F8_E4M3"),
    "encoder.final_layer_norm.weight": ([4096], "F8_E4M3"),
    "shared.weight": ([32128, 4096], "F8_E4M3"),
}
