"""A MOSS-Image LoRA from SimpleTuner, rank 32 in F32, with an F32 `alpha` per layer and a PEFT config in its header.

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`mossloras/moss-image-lora-archive/2026-11-04/MOSS-Image/2064032_DGN-LoRA/3450513/DGN_lora_moss_trigger_@fuhrriel.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Keys are `diffusion_model.` + the transformer's module path + `lora_A.weight`/`lora_B.weight` + `alpha`. Beside
the blocks' attention and feed-forward it trains the text refiner's `proj` and `qkv` in both of its blocks
(`text_embedder.blocks.N`, whose `qkv` is fused in the transformer too). The header's `lora_adapter_metadata`
is kept with its non-default fields only: r 32 and lora_alpha 32, the scale each `alpha` gives as well.
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "diffusion_model.layers.0.proj.ffn_down.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.ffn_down.lora_A.weight": ([32, 8192], "F32"),
    "diffusion_model.layers.0.proj.ffn_down.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.layers.0.proj.ffn_up_gate.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.layers.0.proj.ffn_up_gate.lora_B.weight": ([16384, 32], "F32"),
    "diffusion_model.layers.0.proj.k.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.k.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.layers.0.proj.k.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.layers.0.proj.o.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.o.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.layers.0.proj.o.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.layers.0.proj.q.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.q.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.layers.0.proj.q.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.layers.0.proj.v.alpha": ([], "F32"),
    "diffusion_model.layers.0.proj.v.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.layers.0.proj.v.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.layers.41.proj.ffn_down.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.ffn_down.lora_A.weight": ([32, 8192], "F32"),
    "diffusion_model.layers.41.proj.ffn_down.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.layers.41.proj.ffn_up_gate.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.layers.41.proj.ffn_up_gate.lora_B.weight": ([16384, 32], "F32"),
    "diffusion_model.layers.41.proj.k.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.k.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.layers.41.proj.k.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.layers.41.proj.o.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.o.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.layers.41.proj.o.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.layers.41.proj.q.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.q.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.layers.41.proj.q.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.layers.41.proj.v.alpha": ([], "F32"),
    "diffusion_model.layers.41.proj.v.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.layers.41.proj.v.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.text_embedder.blocks.0.proj.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.0.proj.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.text_embedder.blocks.0.proj.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.text_embedder.blocks.0.qkv.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.0.qkv.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.text_embedder.blocks.0.qkv.lora_B.weight": ([9216, 32], "F32"),
    "diffusion_model.text_embedder.blocks.1.proj.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.1.proj.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.text_embedder.blocks.1.proj.lora_B.weight": ([3072, 32], "F32"),
    "diffusion_model.text_embedder.blocks.1.qkv.alpha": ([], "F32"),
    "diffusion_model.text_embedder.blocks.1.qkv.lora_A.weight": ([32, 3072], "F32"),
    "diffusion_model.text_embedder.blocks.1.qkv.lora_B.weight": ([9216, 32], "F32"),
}

# The part of the header's `__metadata__` a loader reads or ignores on purpose, as the producer wrote it.
metadata: dict[str, str] = {
    "lora_adapter_metadata": '{\n  "transformer.bias": "none",\n  "transformer.init_lora_weights": true,\n  "transformer.lora_alpha": 32.0,\n  "transformer.lora_dropout": 0.1,\n  "transformer.megatron_core": "megatron.core",\n  "transformer.peft_type": "LORA",\n  "transformer.peft_version": "0.21.2",\n  "transformer.qalora_group_size": 16,\n  "transformer.r": 32,\n  "transformer.target_modules": [\n    "qkv",\n    "text_embedder.blocks.0.proj",\n    "proj.k",\n    "proj.v",\n    "proj.ffn_down",\n    "text_embedder.blocks.1.proj",\n    "proj.ffn_up_gate",\n    "proj.q",\n    "proj.o"\n  ]\n}',
}
