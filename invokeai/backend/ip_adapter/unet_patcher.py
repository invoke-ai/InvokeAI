from contextlib import contextmanager

from diffusers.models import UNet2DConditionModel

from invokeai.backend.ip_adapter.attention_processor import AttnProcessor2_0, IPAttnProcessor2_0
from invokeai.backend.ip_adapter.ip_adapter import IPAdapter


def _prepare_attention_processors(unet: UNet2DConditionModel, ip_adapters: list[IPAdapter]):
    """Prepare a dict of attention processors that can be injected into a unet, and load the IP-Adapter attention
    weights into them.

    Note that the `unet` param is only used to determine attention block dimensions and naming.
    """
    # TODO(ryand): This logic can be simplified.

    # Construct a dict of attention processors based on the UNet's architecture.
    attn_procs = {}
    for idx, name in enumerate(unet.attn_processors.keys()):
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor2_0()
        else:
            # Collect the weights from each IP Adapter for the idx'th attention processor.
            attn_procs[name] = IPAttnProcessor2_0(
                [ip_adapter.attn_weights.get_attention_processor_weights(idx) for ip_adapter in ip_adapters]
            )
    return attn_procs


@contextmanager
def apply_ip_adapter_attention(unet: UNet2DConditionModel, ip_adapters: list[IPAdapter]):
    """A context manager that patches `unet` with IP-Adapter attention processors."""
    attn_procs = _prepare_attention_processors(unet, ip_adapters)

    orig_attn_processors = unet.attn_processors

    try:
        # Note to future devs: set_attn_processor(...) does something slightly unexpected - it pops elements from the
        # passed dict. So, if you wanted to keep the dict for future use, you'd have to make a moderately-shallow copy
        # of it. E.g. `attn_procs_copy = {k: v for k, v in attn_procs.items()}`.
        unet.set_attn_processor(attn_procs)
        yield None
    finally:
        unet.set_attn_processor(orig_attn_processors)
