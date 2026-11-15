"""A MOSS-Image LoRA as DiffSynth saves it, akhaliq/MOSS-Image-IDM-2-Steps (rank 64, BF16).

Captured on 2026-11-14 with `scripts/capture_state_dict_fixture.py`,
which reads the header only, from
`akhaliq/MOSS-Image-IDM-2-Steps/opd_2step_v3/step-1500.safetensors`.

Subsetting rule: blocks [0, 41] (first, last, and one next to a missing block where the file has a gap) of the 42 it names, plus every key outside a block. Values are `(shape, dtype)`.

Keys are the transformer's module paths without a prefix, with PEFT's adapter name in each
(`lora_A.default.weight`), which `normalize_peft_adapter_names` strips before anything reads them, and no
`alpha`. It trains the blocks' attention and feed-forward and the text refiner's two blocks (`text_embedder.blocks`).
"""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "layers.0.proj.ffn_down.lora_A.default.weight": ([64, 8192], "BF16"),
    "layers.0.proj.ffn_down.lora_B.default.weight": ([3072, 64], "BF16"),
    "layers.0.proj.ffn_up_gate.lora_A.default.weight": ([64, 3072], "BF16"),
    "layers.0.proj.ffn_up_gate.lora_B.default.weight": ([16384, 64], "BF16"),
    "layers.0.proj.k.lora_A.default.weight": ([64, 3072], "BF16"),
    "layers.0.proj.k.lora_B.default.weight": ([3072, 64], "BF16"),
    "layers.0.proj.o.lora_A.default.weight": ([64, 3072], "BF16"),
    "layers.0.proj.o.lora_B.default.weight": ([3072, 64], "BF16"),
    "layers.0.proj.q.lora_A.default.weight": ([64, 3072], "BF16"),
    "layers.0.proj.q.lora_B.default.weight": ([3072, 64], "BF16"),
    "layers.0.proj.v.lora_A.default.weight": ([64, 3072], "BF16"),
    "layers.0.proj.v.lora_B.default.weight": ([3072, 64], "BF16"),
    "layers.41.proj.ffn_down.lora_A.default.weight": ([64, 8192], "BF16"),
    "layers.41.proj.ffn_down.lora_B.default.weight": ([3072, 64], "BF16"),
    "layers.41.proj.ffn_up_gate.lora_A.default.weight": ([64, 3072], "BF16"),
    "layers.41.proj.ffn_up_gate.lora_B.default.weight": ([16384, 64], "BF16"),
    "layers.41.proj.k.lora_A.default.weight": ([64, 3072], "BF16"),
    "layers.41.proj.k.lora_B.default.weight": ([3072, 64], "BF16"),
    "layers.41.proj.o.lora_A.default.weight": ([64, 3072], "BF16"),
    "layers.41.proj.o.lora_B.default.weight": ([3072, 64], "BF16"),
    "layers.41.proj.q.lora_A.default.weight": ([64, 3072], "BF16"),
    "layers.41.proj.q.lora_B.default.weight": ([3072, 64], "BF16"),
    "layers.41.proj.v.lora_A.default.weight": ([64, 3072], "BF16"),
    "layers.41.proj.v.lora_B.default.weight": ([3072, 64], "BF16"),
    "text_embedder.blocks.0.ffn_down.lora_A.default.weight": ([64, 8192], "BF16"),
    "text_embedder.blocks.0.ffn_down.lora_B.default.weight": ([3072, 64], "BF16"),
    "text_embedder.blocks.0.ffn_gate.lora_A.default.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.0.ffn_gate.lora_B.default.weight": ([8192, 64], "BF16"),
    "text_embedder.blocks.0.ffn_up.lora_A.default.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.0.ffn_up.lora_B.default.weight": ([8192, 64], "BF16"),
    "text_embedder.blocks.0.proj.lora_A.default.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.0.proj.lora_B.default.weight": ([3072, 64], "BF16"),
    "text_embedder.blocks.0.qkv.lora_A.default.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.0.qkv.lora_B.default.weight": ([9216, 64], "BF16"),
    "text_embedder.blocks.1.ffn_down.lora_A.default.weight": ([64, 8192], "BF16"),
    "text_embedder.blocks.1.ffn_down.lora_B.default.weight": ([3072, 64], "BF16"),
    "text_embedder.blocks.1.ffn_gate.lora_A.default.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.1.ffn_gate.lora_B.default.weight": ([8192, 64], "BF16"),
    "text_embedder.blocks.1.ffn_up.lora_A.default.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.1.ffn_up.lora_B.default.weight": ([8192, 64], "BF16"),
    "text_embedder.blocks.1.proj.lora_A.default.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.1.proj.lora_B.default.weight": ([3072, 64], "BF16"),
    "text_embedder.blocks.1.qkv.lora_A.default.weight": ([64, 3072], "BF16"),
    "text_embedder.blocks.1.qkv.lora_B.default.weight": ([9216, 64], "BF16"),
}
