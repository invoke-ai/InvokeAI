from types import SimpleNamespace

import torch
from torch import nn

from invokeai.backend.stable_diffusion import diffusion_backend as diffusion_backend_module
from invokeai.backend.stable_diffusion.diffusion_backend import StableDiffusionBackend


class HookAwareUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.info = {}
        self.received_kwargs = None

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        self.received_kwargs = kwargs
        return SimpleNamespace(sample=sample)


def test_unet_forward_passes_required_inputs_positionally(monkeypatch):
    monkeypatch.setattr(
        diffusion_backend_module,
        "get_config",
        lambda: SimpleNamespace(sequential_guidance=False),
    )
    unet = HookAwareUNet()
    hook_args = []

    def hidiffusion_style_pre_hook(module, args):
        hook_args.append(args)
        module.info["size"] = (args[0].shape[2], args[0].shape[3])

    unet.register_forward_pre_hook(hidiffusion_style_pre_hook)
    backend = StableDiffusionBackend(unet=unet, scheduler=object())

    sample = torch.ones((1, 4, 8, 8))
    timestep = torch.tensor(1)
    encoder_hidden_states = torch.ones((1, 2, 3))
    cross_attention_kwargs = {"scale": 0.5}

    result = backend._unet_forward(
        sample=sample,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        cross_attention_kwargs=cross_attention_kwargs,
    )

    assert unet.info["size"] == (8, 8)
    assert len(hook_args[0]) == 3
    assert hook_args[0][0] is sample
    assert hook_args[0][1] is timestep
    assert hook_args[0][2] is encoder_hidden_states
    assert unet.received_kwargs == {"cross_attention_kwargs": cross_attention_kwargs}
    assert result is sample
