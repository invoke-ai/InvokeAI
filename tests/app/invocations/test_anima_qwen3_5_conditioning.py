"""Anima-3.8B's Qwen3.5 conditioning through the prompt and denoise nodes.

The semantic connector depends on the timestep, so the denoise node must recompute its context at every
step, with that step's sigma -- in float32, because the connector scales sigma by 1000 before embedding
it -- while every other Anima model keeps computing its context once.
"""

import subprocess
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.anima.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.fields import AnimaConditioningField
from invokeai.app.invocations.text_encoder.anima_text_encoder import AnimaTextEncoderInvocation
from invokeai.backend.anima.semantic_connector import QWEN35_LAYER_INDICES
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import AnimaConditioningInfo, ConditioningFieldData


class _Loaded:
    def __init__(self, model, compute_device=torch.device("cpu")):
        self.model = model
        self.compute_device = compute_device

    @contextmanager
    def model_on_device(self, working_mem_bytes=None):
        yield (None, self.model)


# --- prompt node --------------------------------------------------------------------------------------


class _FakeQwen35Tokenizer:
    pad_token_id = 248044

    def encode(self, prompt: str, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [] if not prompt else [5, 6, 7]


class _FakeQwen35Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[list[int], tuple[int, ...], bool]] = []

    def forward(self, input_ids, layer_indices, last_layer_attention_only=False):
        self.calls.append((input_ids[0].tolist(), layer_indices, last_layer_attention_only))
        return [torch.full((1, input_ids.shape[1], 8), float(i)) for i in layer_indices]


def _encode_qwen3_5(monkeypatch, prompt: str) -> tuple[torch.Tensor, torch.Tensor, _FakeQwen35Encoder]:
    module = "invokeai.app.invocations.text_encoder.anima_text_encoder"
    monkeypatch.setattr(f"{module}.PreTrainedTokenizerBase", _FakeQwen35Tokenizer)
    # The node imports Qwen35Encoder when it runs, from its own module.
    monkeypatch.setattr("invokeai.backend.qwen3_5.qwen3_5_encoder.Qwen35Encoder", _FakeQwen35Encoder)
    encoder = _FakeQwen35Encoder()
    context = MagicMock()
    context.models.load.side_effect = [_Loaded(_FakeQwen35Tokenizer()), _Loaded(encoder)]
    invocation = AnimaTextEncoderInvocation.model_construct(
        prompt=prompt, qwen3_5_encoder=SimpleNamespace(tokenizer=object(), text_encoder=object())
    )
    states, mask = invocation._encode_qwen3_5(context)
    return states, mask, encoder


def test_qwen3_5_is_read_at_the_trained_layers_with_an_attention_only_last_layer(monkeypatch) -> None:
    states, mask, encoder = _encode_qwen3_5(monkeypatch, "1girl, Miku")

    assert encoder.calls == [([5, 6, 7], QWEN35_LAYER_INDICES, True)]
    assert states.shape == (len(QWEN35_LAYER_INDICES), 3, 8)
    assert mask.tolist() == [True, True, True]


def test_the_prompt_node_does_not_import_transformers_qwen3_5_at_startup() -> None:
    """Every app start imports every node module; transformers' qwen3_5 modeling costs ~0.25 s of it."""
    program = (
        "import sys\n"
        "import invokeai.app.invocations.text_encoder.anima_text_encoder\n"
        "loaded = [m for m in sys.modules if m.startswith('transformers.models.qwen3_5')]\n"
        "assert not loaded, loaded\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"stderr:\n{result.stderr[-3000:]}"
    assert "OK" in result.stdout


def test_an_empty_prompt_is_one_masked_padding_token(monkeypatch) -> None:
    states, mask, encoder = _encode_qwen3_5(monkeypatch, "")

    assert encoder.calls[0][0] == [_FakeQwen35Tokenizer.pad_token_id]
    assert states.shape[1] == 1
    assert mask.tolist() == [False]


# --- denoise node -------------------------------------------------------------------------------------


class _FakeTransformer(torch.nn.Module):
    """Records what the adapter is asked for; predicts zero velocity."""

    def __init__(self, connector: bool) -> None:
        super().__init__()
        self.has_semantic_connector = connector
        self.adapter_calls: list[tuple[torch.Tensor | None, bool]] = []
        self.patch_spatial = 2
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(28)])

    def preprocess_text_embeds(self, text_embeds, text_ids, t5xxl_weights=None, *, semantic_states=None,
                               semantic_mask=None, timesteps=None):  # fmt: skip
        self.adapter_calls.append((timesteps, semantic_states is not None))
        return torch.zeros(1, 512, 1024)

    def forward(self, x, timesteps, context, **kwargs):
        return torch.zeros_like(x)


def _conditioning(with_qwen3_5: bool) -> ConditioningFieldData:
    return ConditioningFieldData(
        conditionings=[
            AnimaConditioningInfo(
                qwen3_embeds=torch.zeros(3, 1024),
                t5xxl_ids=torch.zeros(3, dtype=torch.long),
                qwen35_states=torch.zeros(4, 2, 2560) if with_qwen3_5 else None,
                qwen35_mask=torch.ones(2, dtype=torch.bool) if with_qwen3_5 else None,
            )
        ]
    )


def _denoise(monkeypatch, *, connector: bool, with_qwen3_5: bool, steps: int = 4) -> _FakeTransformer:
    module = "invokeai.app.invocations.anima.anima_denoise"
    monkeypatch.setattr(f"{module}.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{module}.TorchDevice.choose_anima_inference_dtype", lambda _d: torch.bfloat16)
    monkeypatch.setattr(f"{module}.LayerPatcher.apply_smart_model_patches", lambda **_kw: nullcontext())
    monkeypatch.setattr(f"{module}.patch_anima_for_regional_prompting", lambda *_a: nullcontext())
    transformer = _FakeTransformer(connector)
    context = MagicMock()
    context.models.load.return_value = _Loaded(transformer)
    context.conditioning.load.side_effect = lambda _name: _conditioning(with_qwen3_5)

    invocation = AnimaDenoiseInvocation.model_construct(
        latents=None, noise=None, denoise_mask=None, denoising_start=0.0, denoising_end=1.0, add_noise=True,
        transformer=SimpleNamespace(transformer=object(), loras=[]),
        positive_conditioning=AnimaConditioningField(conditioning_name="pos"),
        negative_conditioning=AnimaConditioningField(conditioning_name="neg"),
        guidance_scale=6.0, width=64, height=64, steps=steps, seed=0, control_lllite=None, scheduler="euler",
    )  # fmt: skip
    invocation._run_diffusion(context)
    return transformer


def test_the_connector_context_is_recomputed_at_every_step_in_float32(monkeypatch) -> None:
    transformer = _denoise(monkeypatch, connector=True, with_qwen3_5=True, steps=4)
    sigmas = AnimaDenoiseInvocation.model_construct()._get_sigmas(4)

    # Positive and negative, once per step; the first step reuses what was built before the loop.
    timesteps = [t for t, _ in transformer.adapter_calls]
    assert len(timesteps) == 2 * 4
    assert all(t is not None and t.dtype == torch.float32 for t in timesteps)
    assert [round(t.item(), 6) for t in timesteps[::2]] == [round(s, 6) for s in sigmas[:4]]
    assert all(has_states for _, has_states in transformer.adapter_calls)


def test_a_model_without_the_connector_computes_its_context_once(monkeypatch) -> None:
    transformer = _denoise(monkeypatch, connector=False, with_qwen3_5=False)

    assert transformer.adapter_calls == [(None, False), (None, False)]


def test_the_connector_without_qwen3_5_conditioning_is_refused(monkeypatch) -> None:
    with pytest.raises(ValueError, match="encoded without Qwen3.5"):
        _denoise(monkeypatch, connector=True, with_qwen3_5=False)
