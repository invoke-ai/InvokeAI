from contextlib import contextmanager
from typing import List, Optional, TypedDict

from diffusers.models import UNet2DConditionModel

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import (
    CustomAttnProcessor2_0,
    IPAdapterAttentionWeights,
)


class UNetIPAdapterData(TypedDict):
    ip_adapter: IPAdapter
    target_blocks: List[str]


class UNetAttentionPatcher:
    """A class for patching a UNet with CustomAttnProcessor2_0 attention layers."""

    def __init__(self, ip_adapter_data: Optional[List[UNetIPAdapterData]]):
        self._ip_adapters = ip_adapter_data

    def _prepare_attention_processors(self, unet: UNet2DConditionModel):
        """Prepare a dict of attention processors that can be injected into a unet, and load the IP-Adapter attention
        weights into them (if IP-Adapters are being applied).
        Note that the `unet` param is only used to determine attention block dimensions and naming.
        """
        # Construct a dict of attention processors based on the UNet's architecture.
        attn_procs = {}
        for idx, name in enumerate(unet.attn_processors.keys()):
            if name.endswith("attn1.processor") or self._ip_adapters is None:
                # "attn1" processors do not use IP-Adapters.
                attn_procs[name] = CustomAttnProcessor2_0()
            else:
                # Collect the weights from each IP Adapter for the idx'th attention processor.
                ip_adapter_attention_weights_collection: list[IPAdapterAttentionWeights] = []

                for ip_adapter in self._ip_adapters:
                    ip_adapter_weights = ip_adapter["ip_adapter"].attn_weights.get_attention_processor_weights(idx)
                    skip = True
                    for block in ip_adapter["target_blocks"]:
                        if block in name:
                            skip = False
                            break
                    ip_adapter_attention_weights: IPAdapterAttentionWeights = IPAdapterAttentionWeights(
                        ip_adapter_weights=ip_adapter_weights, skip=skip
                    )
                    ip_adapter_attention_weights_collection.append(ip_adapter_attention_weights)

                attn_procs[name] = CustomAttnProcessor2_0(ip_adapter_attention_weights_collection)

        return attn_procs

    @contextmanager
    def apply_ip_adapter_attention(self, unet: UNet2DConditionModel):
        """A context manager that patches `unet` with CustomAttnProcessor2_0 attention layers."""
        attn_procs = self._prepare_attention_processors(unet)
        orig_attn_processors = unet.attn_processors

        try:
            # Note to future devs: set_attn_processor(...) does something slightly unexpected - it pops elements from
            # the passed dict. So, if you wanted to keep the dict for future use, you'd have to make a
            # moderately-shallow copy of it. E.g. `attn_procs_copy = {k: v for k, v in attn_procs.items()}`.
            unet.set_attn_processor(attn_procs)
            yield None
        finally:
            unet.set_attn_processor(orig_attn_processors)
