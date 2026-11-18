"""Key layout of ComfyUI's scaled fp8 T5-XXL encoder (`t5xxl_fp8_e4m3fn_scaled`).

Captured on 2026-10-11 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`comfyanonymous/flux_text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors`.

Subsetting rule: `encoder.block` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

Linear weights in float8, each with a scalar float32 `scale_weight` (per tensor, the older spelling
of `weight_scale`), beside a zero-element `scaled_fp8` marker that names no layer. Norms, the position
bias and the embeddings stay fp16. The same file is redistributed in Comfy-Org's SD 3.5 repositories.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "encoder.block.0.layer.0.SelfAttention.k.scale_weight": ([], "F32"),
    "encoder.block.0.layer.0.SelfAttention.k.weight": ([4096, 4096], "F8_E4M3"),
    "encoder.block.0.layer.0.SelfAttention.o.scale_weight": ([], "F32"),
    "encoder.block.0.layer.0.SelfAttention.o.weight": ([4096, 4096], "F8_E4M3"),
    "encoder.block.0.layer.0.SelfAttention.q.scale_weight": ([], "F32"),
    "encoder.block.0.layer.0.SelfAttention.q.weight": ([4096, 4096], "F8_E4M3"),
    "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight": ([32, 64], "F16"),
    "encoder.block.0.layer.0.SelfAttention.v.scale_weight": ([], "F32"),
    "encoder.block.0.layer.0.SelfAttention.v.weight": ([4096, 4096], "F8_E4M3"),
    "encoder.block.0.layer.0.layer_norm.weight": ([4096], "F16"),
    "encoder.block.0.layer.1.DenseReluDense.wi_0.scale_weight": ([], "F32"),
    "encoder.block.0.layer.1.DenseReluDense.wi_0.weight": ([10240, 4096], "F8_E4M3"),
    "encoder.block.0.layer.1.DenseReluDense.wi_1.scale_weight": ([], "F32"),
    "encoder.block.0.layer.1.DenseReluDense.wi_1.weight": ([10240, 4096], "F8_E4M3"),
    "encoder.block.0.layer.1.DenseReluDense.wo.scale_weight": ([], "F32"),
    "encoder.block.0.layer.1.DenseReluDense.wo.weight": ([4096, 10240], "F8_E4M3"),
    "encoder.block.0.layer.1.layer_norm.weight": ([4096], "F16"),
    "encoder.embed_tokens.weight": ([32128, 4096], "F16"),
    "encoder.final_layer_norm.weight": ([4096], "F16"),
    "scaled_fp8": ([0], "F8_E4M3"),
    "shared.weight": ([32128, 4096], "F16"),
}
