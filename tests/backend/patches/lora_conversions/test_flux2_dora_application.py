from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from invokeai.app.invocations.flux2.flux2_denoise import Flux2DenoiseInvocation
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders.lora import LoRALoader
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.lora_conversions.flux_lora_constants import FLUX_LORA_TRANSFORMER_PREFIX


class SplitAttention(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.names = ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")
        attention = torch.nn.ModuleDict({name: torch.nn.Linear(7, 4) for name in self.names})
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.ModuleDict({"attn": attention})])
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
        apply_custom_layers_to_model(self)

    def forward(self, x):
        return torch.cat([self.transformer_blocks[0]["attn"][name](x) for name in self.names], dim=-1)


def load_adapter(tmp_path, axis, layout, storage):
    generator = torch.Generator().manual_seed(581026)
    state, factors = {}, []
    for stream in ("img", "txt"):
        up = (torch.randn(12, 2, generator=generator) * 0.2).to(storage)
        down = (torch.randn(2, 7, generator=generator) * 0.3).to(storage)
        magnitude = torch.linspace(0.7, 1.7, 7 if axis == "input" else 12).to(storage)
        magnitude = magnitude.reshape(1, 7) if axis == "input" else magnitude.reshape(12, 1)
        factors.append((up.float(), down.float(), magnitude.float()))
        if layout == "kohya":
            key = f"lora_unet_double_blocks_0_{stream}_attn_qkv"
            suffixes = ("lora_up.weight", "lora_down.weight")
        else:
            key = f"{layout}double_blocks.0.{stream}_attn.qkv"
            suffixes = ("lora_B.weight", "lora_A.weight")
        state.update(
            {
                f"{key}.{suffixes[0]}": up,
                f"{key}.{suffixes[1]}": down,
                f"{key}.dora_scale": magnitude,
                f"{key}.alpha": torch.tensor(3.0).to(storage),
            }
        )
    return load_state(tmp_path, state, storage), factors


def load_state(tmp_path, state, storage=torch.float32):
    path = tmp_path / "adapter.safetensors"
    save_file(state, path)
    config = SimpleNamespace(path=str(path), base=BaseModelType.Flux2, format=ModelFormat.LyCORIS, name="fixture")
    loader = LoRALoader(MagicMock(models_path=tmp_path), MagicMock(), MagicMock())
    loader._get_model_path(config)
    loader._torch_dtype = storage
    return loader._load_model(config)


def fused_expected(model, factors, x, strength, axis):
    outputs = []
    attention = model.transformer_blocks[0]["attn"]
    for names, (up, down, magnitude) in zip((model.names[:3], model.names[3:]), factors, strict=True):
        base = torch.cat([attention[name].weight.float() for name in names])
        bias = torch.cat([attention[name].bias.float() for name in names])
        direction = base + (3.0 / 2) * (up @ down)
        normalized = magnitude * direction / direction.norm(dim=0 if axis == "input" else 1, keepdim=True)
        outputs.append(torch.nn.functional.linear(x, base + strength * (normalized - base), bias))
    return torch.cat(outputs, dim=-1)


@pytest.mark.parametrize(
    "axis,layout,storage,sidecar",
    [
        ("input", "kohya", torch.float32, False),
        ("input", "diffusion_model.", torch.bfloat16, True),
        ("input", "base_model.model.", torch.bfloat16, False),
        ("output", "diffusion_model.", torch.float32, False),
        ("output", "kohya", torch.bfloat16, True),
    ],
)
@torch.no_grad()
def test_loader_iterator_and_patcher_match_fused_formula_across_bases(tmp_path: Path, axis, layout, storage, sidecar):
    patch, factors = load_adapter(tmp_path, axis, layout, storage)
    placements = [
        (leaf, leaf.up.clone(), leaf.down.clone(), leaf.dora_scale.clone())
        for layer in patch.layers.values()
        for leaf, _, _ in LayerPatcher._leaf_placements(layer)
    ]
    pinned = []

    @contextmanager
    def model_in_ram():
        pinned.append(True)
        try:
            yield patch
        finally:
            pinned.pop()

    context = SimpleNamespace(
        models=SimpleNamespace(load=lambda _: SimpleNamespace(model=patch, model_in_ram=model_in_ram))
    )
    lora = SimpleNamespace(lora="adapter", weight=0.0)
    invocation = SimpleNamespace(transformer=SimpleNamespace(loras=[lora]))
    x = torch.randn(2, 3, 7, generator=torch.Generator().manual_seed(23))
    for seed in (71, 92):
        model = SplitAttention(seed)
        original = {name: value.clone() for name, value in model.state_dict().items()}
        baseline = model(x)
        for strength, fail in ((0.0, False), (-0.4, False), (0.65, True)):
            lora.weight = strength
            expected = fused_expected(model, factors, x, strength, axis)
            try:
                with LayerPatcher.apply_smart_model_patches(
                    model=model,
                    patches=Flux2DenoiseInvocation._lora_iterator(invocation, context),
                    prefix=FLUX_LORA_TRANSFORMER_PREFIX,
                    dtype=torch.float32,
                    force_direct_patching=not sidecar,
                    force_sidecar_patching=sidecar,
                ):
                    assert pinned
                    torch.testing.assert_close(model(x), expected, atol=4e-6, rtol=4e-6)
                    if fail:
                        raise RuntimeError("cancelled")
            except RuntimeError as error:
                assert fail and str(error) == "cancelled"
            assert not pinned
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, original[name], atol=0, rtol=0)
            assert all(module.get_num_patches() == 0 for module in model.transformer_blocks[0]["attn"].values())
            torch.testing.assert_close(model(x), baseline, atol=0, rtol=0)
            for leaf, up, down, magnitude in placements:
                assert leaf.up.dtype == storage and leaf.down.dtype == storage
                torch.testing.assert_close(leaf.up, up, atol=0, rtol=0)
                torch.testing.assert_close(leaf.down, down, atol=0, rtol=0)
                torch.testing.assert_close(leaf.dora_scale, magnitude, atol=0, rtol=0)
