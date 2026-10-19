"""Representative key layout of Comfy-Org's `int8_tensorwise` Krea-2 Turbo build.

Captured on 2026-09-26 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`Comfy-Org/Krea-2/diffusion_models/krea2_turbo_int8_convrot.safetensors`.

Subsetting rule: `blocks` index 0, plus every key outside a stack. Values are `(shape, dtype)`.

These are the native Krea-2 names -- `blocks.0.attn.wq`, not the diffusers
`transformer_blocks.0.attn.to_q` the model is built with -- so this layout only reaches the module
tree through `_convert_krea2_native_to_diffusers`. That is what makes it worth capturing: the side
channel has to survive a rename, and two of these keys are exactly the ones a probe gets wrong.
`blocks.0.mod.lin` is a modulation table whose parameter is not called `weight`, and `txtmlp.0.scale`
is a norm that stores its parameter as `scale`; assuming `<module>.weight` for either costs the layer
its scale, with one INFO line to say so.

The build declares its scheme **only per tensor**: no `__metadata__` block at all, and a
`.comfy_quant` beside every quantized weight. `convrot` and its group size live in that marker and
can live nowhere else -- a header entry cannot carry them -- so a re-upload that dropped `convrot`
would load cleanly against a 256-wide Hadamard and generate noise. That is the one failure this
fixture exists to make loud.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "blocks.0.attn.gate.comfy_quant": ([72], "U8"),
    "blocks.0.attn.gate.weight": ([6144, 6144], "I8"),
    "blocks.0.attn.gate.weight_scale": ([6144, 1], "F32"),
    "blocks.0.attn.qknorm.knorm.scale": ([128], "F32"),
    "blocks.0.attn.qknorm.qnorm.scale": ([128], "F32"),
    "blocks.0.attn.wk.comfy_quant": ([72], "U8"),
    "blocks.0.attn.wk.weight": ([1536, 6144], "I8"),
    "blocks.0.attn.wk.weight_scale": ([1536, 1], "F32"),
    "blocks.0.attn.wo.comfy_quant": ([72], "U8"),
    "blocks.0.attn.wo.weight": ([6144, 6144], "I8"),
    "blocks.0.attn.wo.weight_scale": ([6144, 1], "F32"),
    "blocks.0.attn.wq.comfy_quant": ([72], "U8"),
    "blocks.0.attn.wq.weight": ([6144, 6144], "I8"),
    "blocks.0.attn.wq.weight_scale": ([6144, 1], "F32"),
    "blocks.0.attn.wv.comfy_quant": ([72], "U8"),
    "blocks.0.attn.wv.weight": ([1536, 6144], "I8"),
    "blocks.0.attn.wv.weight_scale": ([1536, 1], "F32"),
    "blocks.0.mlp.down.comfy_quant": ([72], "U8"),
    "blocks.0.mlp.down.weight": ([6144, 16384], "I8"),
    "blocks.0.mlp.down.weight_scale": ([6144, 1], "F32"),
    "blocks.0.mlp.gate.comfy_quant": ([72], "U8"),
    "blocks.0.mlp.gate.weight": ([16384, 6144], "I8"),
    "blocks.0.mlp.gate.weight_scale": ([16384, 1], "F32"),
    "blocks.0.mlp.up.comfy_quant": ([72], "U8"),
    "blocks.0.mlp.up.weight": ([16384, 6144], "I8"),
    "blocks.0.mlp.up.weight_scale": ([16384, 1], "F32"),
    "blocks.0.mod.lin": ([36864], "BF16"),
    "blocks.0.postnorm.scale": ([6144], "F32"),
    "blocks.0.prenorm.scale": ([6144], "F32"),
    "first.bias": ([6144], "BF16"),
    "first.weight": ([6144, 64], "BF16"),
    "last.linear.bias": ([64], "BF16"),
    "last.linear.weight": ([64, 6144], "BF16"),
    "last.modulation.lin": ([2, 6144], "BF16"),
    "last.norm.scale": ([6144], "F32"),
    "tmlp.0.bias": ([6144], "BF16"),
    "tmlp.0.weight": ([6144, 256], "BF16"),
    "tmlp.2.bias": ([6144], "BF16"),
    "tmlp.2.weight": ([6144, 6144], "BF16"),
    "tproj.1.bias": ([36864], "BF16"),
    "tproj.1.weight": ([36864, 6144], "BF16"),
    "txtfusion.layerwise_blocks.0.attn.gate.weight": ([2560, 2560], "BF16"),
    "txtfusion.layerwise_blocks.0.attn.qknorm.knorm.scale": ([128], "F32"),
    "txtfusion.layerwise_blocks.0.attn.qknorm.qnorm.scale": ([128], "F32"),
    "txtfusion.layerwise_blocks.0.attn.wk.weight": ([2560, 2560], "BF16"),
    "txtfusion.layerwise_blocks.0.attn.wo.weight": ([2560, 2560], "BF16"),
    "txtfusion.layerwise_blocks.0.attn.wq.weight": ([2560, 2560], "BF16"),
    "txtfusion.layerwise_blocks.0.attn.wv.weight": ([2560, 2560], "BF16"),
    "txtfusion.layerwise_blocks.0.mlp.down.weight": ([2560, 6912], "BF16"),
    "txtfusion.layerwise_blocks.0.mlp.gate.weight": ([6912, 2560], "BF16"),
    "txtfusion.layerwise_blocks.0.mlp.up.weight": ([6912, 2560], "BF16"),
    "txtfusion.layerwise_blocks.0.postnorm.scale": ([2560], "F32"),
    "txtfusion.layerwise_blocks.0.prenorm.scale": ([2560], "F32"),
    "txtfusion.layerwise_blocks.1.attn.gate.weight": ([2560, 2560], "BF16"),
    "txtfusion.layerwise_blocks.1.attn.qknorm.knorm.scale": ([128], "F32"),
    "txtfusion.layerwise_blocks.1.attn.qknorm.qnorm.scale": ([128], "F32"),
    "txtfusion.layerwise_blocks.1.attn.wk.weight": ([2560, 2560], "BF16"),
    "txtfusion.layerwise_blocks.1.attn.wo.weight": ([2560, 2560], "BF16"),
    "txtfusion.layerwise_blocks.1.attn.wq.weight": ([2560, 2560], "BF16"),
    "txtfusion.layerwise_blocks.1.attn.wv.weight": ([2560, 2560], "BF16"),
    "txtfusion.layerwise_blocks.1.mlp.down.weight": ([2560, 6912], "BF16"),
    "txtfusion.layerwise_blocks.1.mlp.gate.weight": ([6912, 2560], "BF16"),
    "txtfusion.layerwise_blocks.1.mlp.up.weight": ([6912, 2560], "BF16"),
    "txtfusion.layerwise_blocks.1.postnorm.scale": ([2560], "F32"),
    "txtfusion.layerwise_blocks.1.prenorm.scale": ([2560], "F32"),
    "txtfusion.projector.weight": ([1, 12], "BF16"),
    "txtfusion.refiner_blocks.0.attn.gate.weight": ([2560, 2560], "BF16"),
    "txtfusion.refiner_blocks.0.attn.qknorm.knorm.scale": ([128], "F32"),
    "txtfusion.refiner_blocks.0.attn.qknorm.qnorm.scale": ([128], "F32"),
    "txtfusion.refiner_blocks.0.attn.wk.weight": ([2560, 2560], "BF16"),
    "txtfusion.refiner_blocks.0.attn.wo.weight": ([2560, 2560], "BF16"),
    "txtfusion.refiner_blocks.0.attn.wq.weight": ([2560, 2560], "BF16"),
    "txtfusion.refiner_blocks.0.attn.wv.weight": ([2560, 2560], "BF16"),
    "txtfusion.refiner_blocks.0.mlp.down.weight": ([2560, 6912], "BF16"),
    "txtfusion.refiner_blocks.0.mlp.gate.weight": ([6912, 2560], "BF16"),
    "txtfusion.refiner_blocks.0.mlp.up.weight": ([6912, 2560], "BF16"),
    "txtfusion.refiner_blocks.0.postnorm.scale": ([2560], "F32"),
    "txtfusion.refiner_blocks.0.prenorm.scale": ([2560], "F32"),
    "txtfusion.refiner_blocks.1.attn.gate.weight": ([2560, 2560], "BF16"),
    "txtfusion.refiner_blocks.1.attn.qknorm.knorm.scale": ([128], "F32"),
    "txtfusion.refiner_blocks.1.attn.qknorm.qnorm.scale": ([128], "F32"),
    "txtfusion.refiner_blocks.1.attn.wk.weight": ([2560, 2560], "BF16"),
    "txtfusion.refiner_blocks.1.attn.wo.weight": ([2560, 2560], "BF16"),
    "txtfusion.refiner_blocks.1.attn.wq.weight": ([2560, 2560], "BF16"),
    "txtfusion.refiner_blocks.1.attn.wv.weight": ([2560, 2560], "BF16"),
    "txtfusion.refiner_blocks.1.mlp.down.weight": ([2560, 6912], "BF16"),
    "txtfusion.refiner_blocks.1.mlp.gate.weight": ([6912, 2560], "BF16"),
    "txtfusion.refiner_blocks.1.mlp.up.weight": ([6912, 2560], "BF16"),
    "txtfusion.refiner_blocks.1.postnorm.scale": ([2560], "F32"),
    "txtfusion.refiner_blocks.1.prenorm.scale": ([2560], "F32"),
    "txtmlp.0.scale": ([2560], "F32"),
    "txtmlp.1.bias": ([6144], "BF16"),
    "txtmlp.1.weight": ([6144, 2560], "BF16"),
    "txtmlp.3.bias": ([6144], "BF16"),
    "txtmlp.3.weight": ([6144, 6144], "BF16"),
}

# The per-layer `.comfy_quant` markers, decoded from the blobs themselves.
markers: dict[str, dict[str, object]] = {
    "blocks.0.attn.gate": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.attn.wk": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.attn.wo": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.attn.wq": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.attn.wv": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.mlp.down": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.mlp.gate": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
    "blocks.0.mlp.up": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256},
}
