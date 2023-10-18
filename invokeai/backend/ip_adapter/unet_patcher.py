from contextlib import contextmanager

from diffusers.models import UNet2DConditionModel

from invokeai.backend.ip_adapter.attention_processor import AttnProcessor2_0, IPAttnProcessor2_0
from invokeai.backend.ip_adapter.ip_adapter import IPAdapter


class UNetPatcher:
    """A class that contains multiple IP-Adapters and can apply them to a UNet."""

    def __init__(self, ip_adapters: list[IPAdapter]):
        self._ip_adapters = ip_adapters
        self._scales = [1.0] * len(self._ip_adapters)

    def set_scale(self, idx: int, value: float):
        self._scales[idx] = value

    def _prepare_attention_processors(self, unet: UNet2DConditionModel):
        """Prepare a dict of attention processors that can be injected into a unet, and load the IP-Adapter attention
        weights into them.

        Note that the `unet` param is only used to determine attention block dimensions and naming.
        """
        # Construct a dict of attention processors based on the UNet's architecture.
        attn_procs = {}
        for idx, name in enumerate(unet.attn_processors.keys()):
            if name.endswith("attn1.processor"):
                attn_procs[name] = AttnProcessor2_0()
            else:
                # Collect the weights from each IP Adapter for the idx'th attention processor.
                attn_procs[name] = IPAttnProcessor2_0(
                    [ip_adapter.attn_weights.get_attention_processor_weights(idx) for ip_adapter in self._ip_adapters],
                    self._scales,
                )
        return attn_procs

    @contextmanager
    def apply_ip_adapter_attention(self, unet: UNet2DConditionModel):
        """A context manager that patches `unet` with IP-Adapter attention processors."""

        attn_procs = self._prepare_attention_processors(unet)

        orig_attn_processors = unet.attn_processors

        try:
            # Note to future devs: set_attn_processor(...) does something slightly unexpected - it pops elements from the
            # passed dict. So, if you wanted to keep the dict for future use, you'd have to make a moderately-shallow copy
            # of it. E.g. `attn_procs_copy = {k: v for k, v in attn_procs.items()}`.
            unet.set_attn_processor(attn_procs)
            yield None
        finally:
            unet.set_attn_processor(orig_attn_processors)
