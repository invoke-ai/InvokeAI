"""CPU VAE conditioning reaches the session's real denoise concatenation boundary.

Meta stands in for a different session device, not for GPU execution. The VAE encodes real
pixels in fp32 on CPU; only the expensive transformer's forward is stopped at its input.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from invokeai.app.invocations.fields import FluxConditioningField
from invokeai.app.invocations.flux.flux_denoise import FluxDenoiseInvocation
from invokeai.backend.flux.model import Flux, FluxParams
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_MAP
from invokeai.backend.model_manager.taxonomy import BaseModelType, FluxVariantType, ModelFormat, ModelType
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningFieldData, FLUXConditioningInfo
from invokeai.backend.util.devices import TorchDevice


class _ReachedTransformer(Exception):
    pass


class _CpuTimestepsEuler(FlowMatchEulerDiscreteScheduler):
    def set_timesteps(self, *args, device=None, **kwargs):
        # Meta has no scalar values for denoise's timestep.item(). Keep the real schedule on
        # CPU while all session tensors still travel through the real meta concatenation.
        return super().set_timesteps(*args, device="cpu", **kwargs)


@pytest.mark.parametrize("route", ["fill", "control-lora"])
@pytest.mark.parametrize("session_device", ["cpu", "meta"])
def test_cpu_vae_conditioning_reaches_the_session_transformer(route, session_device, flux_shaped_vae, monkeypatch):
    device = torch.device(session_device)
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: device)
    if session_device == "meta":
        monkeypatch.setitem(FLUX_SCHEDULER_MAP, "euler", _CpuTimestepsEuler)
    vae = flux_shaped_vae().eval()
    encode = Mock(wraps=vae.encode)
    monkeypatch.setattr(vae, "encode", encode)
    vae_info = SimpleNamespace(
        model=vae,
        compute_device=torch.device("cpu"),
        model_on_device=lambda **_: nullcontext((None, vae)),
    )
    # The real invocation and backend denoise run until the transformer forward. No CPU bf16
    # matmul is needed: the forward is outside this VAE-to-session boundary regression.
    transformer = Flux(
        FluxParams(
            in_channels=384 if route == "fill" else 128,
            out_channels=64,
            vec_in_dim=8,
            context_in_dim=8,
            hidden_size=12,
            mlp_ratio=1.0,
            num_heads=2,
            depth=0,
            depth_single_blocks=0,
            axes_dim=[2, 2, 2],
            theta=10000,
            qkv_bias=True,
            guidance_embed=True,
        )
    )

    def transformer_input(**kwargs):
        pixels = encode.call_args.args[0]
        assert pixels.device == torch.device("cpu")
        assert pixels.dtype == torch.float32
        assert torch.isfinite(pixels).all()
        img = kwargs["img"]
        assert img.device == device
        assert img.dtype == torch.bfloat16
        assert img.shape == (1, 4, transformer.in_channels)
        if route == "fill" and session_device == "cpu":
            # Mask channels follow the 64 noise and 64 VAE channels, and survive packing.
            assert (img[..., 128:] == 0).any()
            assert (img[..., 128:] == 1).any()
        raise _ReachedTransformer

    monkeypatch.setattr(transformer, "forward", transformer_input)
    transformer_info = SimpleNamespace(
        model=transformer,
        model_on_device=lambda **_: nullcontext((None, transformer)),
    )
    lora_info = SimpleNamespace(model=ModelPatchRaw(layers={}), model_in_ram=lambda: nullcontext())
    models = {"vae": vae_info, "transformer": transformer_info, "lora": lora_info}
    mask = torch.zeros(1, 32, 32, dtype=torch.bool)
    mask[:, :, 16:] = True
    context = SimpleNamespace(
        models=SimpleNamespace(
            load=models.__getitem__,
            get_config=lambda _: SimpleNamespace(
                base=BaseModelType.Flux,
                type=ModelType.Main,
                format=ModelFormat.Checkpoint,
                variant=FluxVariantType.DevFill if route == "fill" else FluxVariantType.Dev,
            ),
        ),
        images=SimpleNamespace(get_pil=lambda *_, **__: Image.new("RGB", (32, 32), (40, 80, 120))),
        tensors=SimpleNamespace(load=lambda _: mask.clone()),
        conditioning=SimpleNamespace(
            load=lambda _: ConditioningFieldData(
                conditionings=[FLUXConditioningInfo(clip_embeds=torch.zeros(1, 8), t5_embeds=torch.zeros(1, 2, 8))]
            )
        ),
        logger=Mock(),
    )
    invocation = FluxDenoiseInvocation.model_construct(
        transformer=SimpleNamespace(transformer="transformer", loras=[]),
        controlnet_vae=SimpleNamespace(vae="vae"),
        positive_text_conditioning=FluxConditioningField(conditioning_name="positive", mask=None),
        control_lora=SimpleNamespace(img=SimpleNamespace(image_name="control"), lora="lora", weight=1.0)
        if route == "control-lora"
        else None,
        fill_conditioning=SimpleNamespace(
            image=SimpleNamespace(image_name="fill"), mask=SimpleNamespace(tensor_name="mask")
        )
        if route == "fill"
        else None,
        width=32,
        height=32,
        num_steps=1,
        cfg_scale=1.0,
        guidance=30.0,
        seed=0,
    )

    with pytest.raises(_ReachedTransformer):
        invocation._run_diffusion(context)

    encode.assert_called_once()
