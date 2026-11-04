"""Run the denoiser with lightweight real attention to check regional CFG routing."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.anima.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.fields import AnimaConditioningField, TensorField
from invokeai.backend.anima.anima_transformer import CosmosAttention
from invokeai.backend.model_manager.taxonomy import ModelFormat
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import AnimaConditioningInfo, ConditioningFieldData


class _Transformer(torch.nn.Module):
    def __init__(self, connector: bool = False, fail: bool = False):
        super().__init__()
        self.has_semantic_connector = connector
        self.fail = fail
        self.patch_spatial = 2
        self.blocks = torch.nn.ModuleList([torch.nn.Module()])
        attention = CosmosAttention(query_dim=2, context_dim=2, n_heads=1, head_dim=2)
        with torch.no_grad():
            attention.q_proj.weight.zero_()
            attention.k_proj.weight.zero_()
            attention.v_proj.weight.copy_(torch.eye(2))
            attention.output_proj.weight.copy_(torch.eye(2))
        self.blocks[0].cross_attn = attention
        self.original_forward = attention.forward
        self.outputs: list[torch.Tensor] = []
        self.patched: list[bool] = []
        self.adapter_timesteps: list[torch.Tensor | None] = []

    def preprocess_text_embeds(self, text_embeds, text_ids, **kwargs):
        self.adapter_timesteps.append(kwargs["timesteps"])
        return text_embeds

    def forward(self, x, context, **kwargs):
        self.patched.append(self.blocks[0].cross_attn.forward != self.original_forward)
        output = self.blocks[0].cross_attn(torch.zeros(1, 2, 2), context)
        self.outputs.append(output.detach())
        if self.fail:
            raise RuntimeError("transformer failed")
        return torch.zeros_like(x)


class _Loaded:
    def __init__(self, model):
        self.model = model
        self.config = SimpleNamespace(format=ModelFormat.Checkpoint)

    @contextmanager
    def model_on_device(self, **kwargs):
        yield None, self.model


def _run(
    monkeypatch,
    *,
    positive_region: bool = True,
    negative_region: bool = True,
    negative_tokens: int = 1,
    cfg: float = 2.0,
    scheduler: str = "euler",
    transformer: _Transformer | None = None,
):
    module = "invokeai.app.invocations.anima.anima_denoise"
    monkeypatch.setattr(f"{module}.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{module}.TorchDevice.choose_anima_inference_dtype", lambda _d: torch.float32)
    monkeypatch.setattr(f"{module}.LayerPatcher.apply_smart_model_patches", lambda **_kw: nullcontext())
    if transformer is None:
        transformer = _Transformer()
    context = MagicMock()
    context.models.load.return_value = _Loaded(transformer)
    values = {"pos": [1.0, 0.0], "pos-region": [0.0, 2.0], "neg": [-1.0, 0.0], "neg-region": [0.0, -2.0]}

    def load_conditioning(name):
        if cfg == 1.0:
            assert not name.startswith("neg"), "CFG-off must not load negative conditioning"
        tokens = negative_tokens if name == "neg-region" else 1
        return ConditioningFieldData(
            conditionings=[
                AnimaConditioningInfo(
                    qwen3_embeds=torch.tensor([values[name]] * tokens),
                    t5xxl_ids=torch.zeros(tokens, dtype=torch.long),
                    qwen35_states=torch.zeros(4, 1, 2) if transformer.has_semantic_connector else None,
                    qwen35_mask=torch.ones(1, dtype=torch.bool) if transformer.has_semantic_connector else None,
                )
            ]
        )

    context.conditioning.load.side_effect = load_conditioning
    context.tensors.load.side_effect = lambda name: {
        "pos-mask": torch.tensor([[1.0, 0.0]]),
        "neg-mask": torch.tensor([[0.0, 1.0]]),
    }[name]
    positive = [AnimaConditioningField(conditioning_name="pos")]
    negative = [AnimaConditioningField(conditioning_name="neg")]
    if positive_region:
        positive.append(
            AnimaConditioningField(conditioning_name="pos-region", mask=TensorField(tensor_name="pos-mask"))
        )
    if negative_region:
        negative.append(
            AnimaConditioningField(conditioning_name="neg-region", mask=TensorField(tensor_name="neg-mask"))
        )
    invocation = AnimaDenoiseInvocation.model_construct(
        transformer=SimpleNamespace(transformer=object(), loras=[]),
        positive_conditioning=positive, negative_conditioning=negative, guidance_scale=cfg,
        width=32, height=16, steps=2, seed=0, scheduler=scheduler,
    )  # fmt: skip
    invocation._run_diffusion(context)
    assert transformer.blocks[0].cross_attn.forward == transformer.original_forward
    return transformer


def _expected_negative(negative_tokens: int = 1) -> torch.Tensor:
    """The global negative stays image-wide: the region token averages it with its `negative_tokens` copies."""
    region_token = [-1.0 / (1 + negative_tokens), -2.0 * negative_tokens / (1 + negative_tokens)]
    return torch.tensor([[[-1.0, 0.0], region_token]])


@pytest.mark.parametrize("scheduler", ["euler", "heun"])
@pytest.mark.parametrize("negative_tokens", [1, 3], ids=["equal-context-length", "different-context-length"])
def test_each_cfg_pass_uses_its_own_regions(monkeypatch, scheduler, negative_tokens) -> None:
    transformer = _run(monkeypatch, scheduler=scheduler, negative_tokens=negative_tokens)
    # The positive pass's global prompt yields the region to the regional prompt.
    expected_pos = torch.tensor([[[0.0, 2.0], [1.0, 0.0]]])
    expected_neg = _expected_negative(negative_tokens)
    for positive, negative in zip(transformer.outputs[::2], transformer.outputs[1::2], strict=True):
        torch.testing.assert_close(positive, expected_pos)
        torch.testing.assert_close(negative, expected_neg)


@pytest.mark.parametrize("positive_region,negative_region", [(False, True), (True, False), (False, False)])
def test_a_global_branch_is_unmasked_even_when_the_other_branch_has_regions(
    monkeypatch, positive_region, negative_region
) -> None:
    transformer = _run(monkeypatch, positive_region=positive_region, negative_region=negative_region)
    expected_pos = [[[0.0, 2.0], [1.0, 0.0]]] if positive_region else [[[1.0, 0.0], [1.0, 0.0]]]
    expected_neg = _expected_negative() if negative_region else torch.tensor([[[-1.0, 0.0], [-1.0, 0.0]]])
    for positive, negative in zip(transformer.outputs[::2], transformer.outputs[1::2], strict=True):
        torch.testing.assert_close(positive, torch.tensor(expected_pos))
        torch.testing.assert_close(negative, expected_neg)
    # Without any regional mask the cross-attention stays the original, unpatched forward.
    assert set(transformer.patched) == {positive_region or negative_region}


def test_cfg_off_ignores_negative_regions(monkeypatch) -> None:
    transformer = _run(monkeypatch, cfg=1.0)
    assert len(transformer.outputs) == 2
    for output in transformer.outputs:
        torch.testing.assert_close(output, torch.tensor([[[0.0, 2.0], [1.0, 0.0]]]))


def test_connector_rebuilds_every_region_on_both_sides_at_each_sigma(monkeypatch) -> None:
    transformer = _run(monkeypatch, transformer=_Transformer(connector=True))
    sigmas = AnimaDenoiseInvocation.model_construct()._get_sigmas(2)
    assert len(transformer.adapter_timesteps) == 8
    for step in range(2):
        for timestep in transformer.adapter_timesteps[step * 4 : (step + 1) * 4]:
            assert timestep is not None and timestep.dtype == torch.float32
            assert timestep.item() == pytest.approx(sigmas[step])
    for output in transformer.outputs[1::2]:
        torch.testing.assert_close(output, _expected_negative())


def test_transformer_failure_restores_regional_patches(monkeypatch) -> None:
    transformer = _Transformer(fail=True)
    with pytest.raises(RuntimeError, match="transformer failed"):
        _run(monkeypatch, transformer=transformer)
    assert transformer.blocks[0].cross_attn.forward == transformer.original_forward
