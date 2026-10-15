"""FLUX.1 denoise wiring: the dequant transient and the sidecar decision.

`peak_dequant_transient_bytes` and `requires_sidecar_patching` are unit-tested next to the module
they belong to. What is pinned here is the wire -- deleting the `working_mem_bytes=` argument from
the node, or deciding the sidecar flag from the format alone, leaves every other test green.

Z-Image's equivalent lives in `test_z_image_denoise_working_memory.py`, Krea-2's in
`test_krea2_denoise.py` and FLUX.2's in `test_flux2_working_memory.py`.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.invocations.fields import FluxConditioningField
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.quantization.int8_convrot import CONVROT_GROUP_SIZE, Int8ConvrotLinear


class _StopBeforeLoad(Exception):
    """Raised out of the mocked `model_on_device()` so the node stops at the reservation."""


def _int8_model(out_features: int, in_features: int) -> torch.nn.Module:
    model = torch.nn.Module()
    model.proj = Int8ConvrotLinear(
        torch.zeros(out_features, in_features, dtype=torch.int8),
        torch.ones(out_features, 1),
        convrot=False,
    )
    return model


class TestFlux1:
    """FLUX.1's node passed no `working_mem_bytes` at all before int8 reached this loader.

    Its largest quantized layer is `single_blocks.N.linear1` at 21504x3072, i.e. a 252 MiB bf16
    transient on the released repack -- against a card that is already holding 11.5 GiB of weights
    plus T5-XXL.
    """

    def _reserved_by(self, transformer: torch.nn.Module, on_device=None) -> int:
        """`on_device` lets a caller get past the load instead of stopping at it."""
        from invokeai.app.invocations.flux.flux_denoise import FluxDenoiseInvocation

        transformer_info = MagicMock()
        transformer_info.model = transformer
        transformer_info.model_on_device = MagicMock(side_effect=on_device or _StopBeforeLoad)

        context = MagicMock()
        context.models.load.return_value = transformer_info
        from invokeai.backend.model_manager.taxonomy import FluxVariantType

        context.models.get_config.return_value = SimpleNamespace(
            base=BaseModelType.Flux,
            type=ModelType.Main,
            format=ModelFormat.Checkpoint,
            variant=FluxVariantType.Dev,
        )
        from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
            ConditioningFieldData,
            FLUXConditioningInfo,
        )

        context.conditioning.load.return_value = ConditioningFieldData(
            conditionings=[FLUXConditioningInfo(clip_embeds=torch.zeros(1, 768), t5_embeds=torch.zeros(1, 8, 4096))]
        )

        invocation = FluxDenoiseInvocation.model_construct(
            latents=None,
            noise=None,
            denoise_mask=None,
            denoising_start=0.0,
            denoising_end=1.0,
            add_noise=True,
            transformer=MagicMock(transformer=MagicMock(), loras=[]),
            positive_text_conditioning=FluxConditioningField(conditioning_name="pos", mask=None),
            negative_text_conditioning=None,
            control_lora=None,
            controlnet_vae=None,
            control=None,
            ip_adapter=None,
            kontext_conditioning=None,
            redux_conditioning=None,
            cfg_scale=1.0,
            cfg_scale_start_step=0,
            cfg_scale_end_step=-1,
            guidance=4.0,
            width=256,
            height=256,
            num_steps=2,
            seed=0,
        )

        with (
            patch(
                "invokeai.app.invocations.flux.flux_denoise.TorchDevice.choose_torch_device",
                return_value=torch.device("cpu"),
            ),
            pytest.raises(_StopBeforeLoad),
        ):
            invocation._run_diffusion(context)

        transformer_info.model_on_device.assert_called_once()
        return transformer_info.model_on_device.call_args.kwargs["working_mem_bytes"]

    def test_the_transient_reaches_the_model_cache(self) -> None:
        reserved = self._reserved_by(_int8_model(1024, CONVROT_GROUP_SIZE))

        assert reserved == 2 * 1024 * CONVROT_GROUP_SIZE * torch.bfloat16.itemsize

    def test_an_unquantized_transformer_asks_for_nothing_extra(self) -> None:
        dense = torch.nn.Module()
        dense.proj = torch.nn.Linear(CONVROT_GROUP_SIZE, 8)

        assert self._reserved_by(dense) == 0

    @staticmethod
    def _flux(quantized: bool) -> torch.nn.Module:
        """A real (tiny) `Flux`, because the node asserts the type before it patches."""
        from invokeai.backend.flux.model import Flux, FluxParams

        params = FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=CONVROT_GROUP_SIZE,
            mlp_ratio=4.0,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        )
        model = Flux(params)
        if quantized:
            model.single_blocks[0].linear1 = Int8ConvrotLinear(
                torch.zeros(1024, CONVROT_GROUP_SIZE, dtype=torch.int8),
                torch.ones(1024, 1),
                convrot=False,
            )
        return model

    def _sidecar_flag_at_the_node(self, transformer: torch.nn.Module) -> bool:
        """What the node actually hands `LayerPatcher`.

        `Checkpoint` is the format an int8 build carries, so the old format-only decision answered
        "not quantized" and `LayerPatcher` picked direct patching -- which asks an
        `Int8ConvrotLinear` for a `weight` *parameter* it does not have. Every LoRA on an int8
        FLUX.1 would die there, with an error naming nothing.
        """
        from contextlib import contextmanager

        from invokeai.backend.patches.layer_patcher import LayerPatcher

        captured: dict = {}

        @contextmanager
        def on_device(**_kwargs):
            yield (None, transformer)

        def apply_patches(**kwargs):
            captured["force_sidecar_patching"] = kwargs["force_sidecar_patching"]
            raise _StopBeforeLoad

        with patch.object(LayerPatcher, "apply_smart_model_patches", staticmethod(apply_patches)):
            self._reserved_by(transformer, on_device=on_device)
        return captured["force_sidecar_patching"]

    def test_the_node_asks_the_model_and_not_only_the_format(self) -> None:
        assert self._sidecar_flag_at_the_node(self._flux(quantized=True)) is True
        assert self._sidecar_flag_at_the_node(self._flux(quantized=False)) is False
