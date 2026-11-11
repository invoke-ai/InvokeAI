"""Key layout of SimpleTuner's LyCORIS LoKR over Z-Image
(`noodlecake/Retro_Anime_Painting_lokr-R8-LR2-HIGHNOISEBIAS`).

Captured on 2026-11-11 with a header-only read of `retanimpaint_lkr_8_v2_2500.safetensors`, the
issue #10174 Z-Image row (ai-toolkit 0.7.9's LyCORIS integration writes the same spelling).
Subsetting rule: layer 0 in full -- every module kind of the file. Values are `(shape, dtype)`.

Z-Image's transformer module names flattened behind `lycoris_`
(`lycoris_layers_0_attention_to_k`, `lycoris_layers_0_feed_forward_w1`,
`lycoris_layers_0_adaLN_modulation_0`). 720 keys in the file, 240 layers (30 of 36 layers).
The attention width 3840 (LoKR product) is what tells Z-Image from Ideogram 4 (4608), which uses
the same module names."""

state_dict_keys: dict[str, tuple[list[int], str]] = {
    "lycoris_layers_0_adaLN_modulation_0.alpha": ([], "BF16"),
    "lycoris_layers_0_adaLN_modulation_0.lokr_w1": ([8, 8], "BF16"),
    "lycoris_layers_0_adaLN_modulation_0.lokr_w2": ([1920, 32], "BF16"),
    "lycoris_layers_0_attention_to_k.alpha": ([], "BF16"),
    "lycoris_layers_0_attention_to_k.lokr_w1": ([8, 8], "BF16"),
    "lycoris_layers_0_attention_to_k.lokr_w2": ([480, 480], "BF16"),
    "lycoris_layers_0_attention_to_out_0.alpha": ([], "BF16"),
    "lycoris_layers_0_attention_to_out_0.lokr_w1": ([8, 8], "BF16"),
    "lycoris_layers_0_attention_to_out_0.lokr_w2": ([480, 480], "BF16"),
    "lycoris_layers_0_attention_to_q.alpha": ([], "BF16"),
    "lycoris_layers_0_attention_to_q.lokr_w1": ([8, 8], "BF16"),
    "lycoris_layers_0_attention_to_q.lokr_w2": ([480, 480], "BF16"),
    "lycoris_layers_0_attention_to_v.alpha": ([], "BF16"),
    "lycoris_layers_0_attention_to_v.lokr_w1": ([8, 8], "BF16"),
    "lycoris_layers_0_attention_to_v.lokr_w2": ([480, 480], "BF16"),
    "lycoris_layers_0_feed_forward_w1.alpha": ([], "BF16"),
    "lycoris_layers_0_feed_forward_w1.lokr_w1": ([8, 8], "BF16"),
    "lycoris_layers_0_feed_forward_w1.lokr_w2": ([1280, 480], "BF16"),
    "lycoris_layers_0_feed_forward_w2.alpha": ([], "BF16"),
    "lycoris_layers_0_feed_forward_w2.lokr_w1": ([8, 8], "BF16"),
    "lycoris_layers_0_feed_forward_w2.lokr_w2": ([480, 1280], "BF16"),
    "lycoris_layers_0_feed_forward_w3.alpha": ([], "BF16"),
    "lycoris_layers_0_feed_forward_w3.lokr_w1": ([8, 8], "BF16"),
    "lycoris_layers_0_feed_forward_w3.lokr_w2": ([1280, 480], "BF16"),
}
