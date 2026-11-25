"""Key layout of an SD 2.1 UNet's attentions under SimpleTuner's LyCORIS flat spelling.

SD 2.1 names its UNet modules exactly like SD 1.5 (`UNet2DConditionModel`, `block_out_channels`
320/640/1280/1280, `attention_head_dim` 8, `use_linear_projection`); only the text encoding's width
differs, 1024 against SD 1.5's 768. Both therefore flatten into SDXL's own attention templates at
transformer-block index 0, which is why no config may claim a file of these names.
`state_dict_keys` holds the attention projections (128 layers),
`attention_and_ff_state_dict_keys` adds the feed-forward pair (160 layers).
"""

from tests.backend.patches.lora_conversions.lora_state_dicts import sd15_lycoris_flat_lokr_format

state_dict_keys = sd15_lycoris_flat_lokr_format.build_state_dict_keys(1024, include_feed_forward=False)
attention_and_ff_state_dict_keys = sd15_lycoris_flat_lokr_format.build_state_dict_keys(1024, include_feed_forward=True)
