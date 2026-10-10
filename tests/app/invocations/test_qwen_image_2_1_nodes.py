"""The Qwen-Image-2.1 denoise and model-loader nodes, run against stand-ins for the model and the context.

What the GPU parity run cannot see: the order the prefix cache is filled in when guidance changes per step, where
an image-to-image run starts on each variant's schedule, and which components the loader refuses.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.fields import LatentsField, QwenImage21ConditioningField
from invokeai.app.invocations.model import ModelIdentifierField, TransformerField
from invokeai.app.invocations.qwen_image_2_1.qwen_image_2_1_denoise import (
    LATENT_CHANNELS,
    QwenImage21DenoiseInvocation,
    pack_latents,
    unpack_latents,
)
from invokeai.app.invocations.qwen_image_2_1.qwen_image_2_1_model_loader import QwenImage21ModelLoaderInvocation
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen3VLVariantType,
    QwenImage21VariantType,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    QwenImage21ConditioningInfo,
)

DENOISE = "invokeai.app.invocations.qwen_image_2_1.qwen_image_2_1_denoise"


def test_latents_pack_one_token_per_pixel_in_raster_order() -> None:
    latents = torch.randn(1, LATENT_CHANNELS, 2, 3)
    packed = pack_latents(latents)
    assert packed.shape == (1, 6, LATENT_CHANNELS)
    assert torch.equal(packed[0, 1 * 3 + 2], latents[0, :, 1, 2])
    assert torch.equal(unpack_latents(packed, 2, 3), latents)


class _Transformer:
    """Records each forward: which prompt it saw (by its fill value), the cache it got and the cache mode."""

    config = SimpleNamespace(causal_condition=True, num_attention_heads=2, attention_head_dim=4)
    transformer_blocks = [object(), object()]

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None, float]] = []

    def modules(self):
        return iter(())

    def __call__(self, *, hidden_states, encoder_hidden_states, timestep, kv_cache, kv_cache_mode, **_kwargs):
        prompt = "pos" if float(encoder_hidden_states.mean()) > 0 else "neg"
        self.calls.append((prompt, id(kv_cache), kv_cache_mode, round(float(timestep[0]), 3)))
        # The transformer returns the joint sequence; the node keeps the image tokens at its end.
        return (torch.zeros(1, encoder_hidden_states.shape[1] + hidden_states.shape[1], hidden_states.shape[2]),)


class _TransformerInfo:
    def __init__(self, transformer: _Transformer) -> None:
        self.model = transformer

    @contextmanager
    def model_on_device(self, **_kwargs):
        yield ({}, self.model)


def _context(transformer: _Transformer, variant: QwenImage21VariantType, init_latents: torch.Tensor | None = None):
    conditionings = {
        "pos": ConditioningFieldData(conditionings=[QwenImage21ConditioningInfo(prompt_embeds=torch.ones(1, 3, 8))]),
        "neg": ConditioningFieldData(conditionings=[QwenImage21ConditioningInfo(prompt_embeds=-torch.ones(1, 2, 8))]),
    }
    return SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: _TransformerInfo(transformer),
            get_config=lambda _identifier: SimpleNamespace(variant=variant),
        ),
        conditioning=SimpleNamespace(load=lambda name: conditionings[name]),
        tensors=SimpleNamespace(load=lambda _name: init_latents),
        util=SimpleNamespace(sd_step_callback=lambda *_args: None),
    )


def _denoise(**fields) -> QwenImage21DenoiseInvocation:
    model = ModelIdentifierField(key="m", hash="h", name="m", base=BaseModelType.QwenImage21, type=ModelType.Main)
    defaults = {
        "transformer": TransformerField(transformer=model, loras=[]),
        "positive_conditioning": QwenImage21ConditioningField(conditioning_name="pos"),
        "negative_conditioning": QwenImage21ConditioningField(conditioning_name="neg"),
        "latents": None,
        "denoise_mask": None,
        "denoising_start": 0.0,
        "denoising_end": 1.0,
        "cfg_scale": 1.0,
        "width": 64,
        "height": 64,
        "steps": 3,
        "seed": 0,
    }
    return QwenImage21DenoiseInvocation.model_construct(**(defaults | fields))


@pytest.fixture(autouse=True)
def _cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{DENOISE}.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{DENOISE}.TorchDevice.choose_bfloat16_safe_dtype", lambda _device: torch.float32)


def test_each_guidance_branch_prefills_its_own_cache_on_the_first_step() -> None:
    transformer = _Transformer()
    # Guidance off at step 0: the negative branch must still prefill there, or step 1 decodes from an empty cache.
    _denoise(cfg_scale=[1.0, 4.0, 4.0])._run_diffusion(_context(transformer, QwenImage21VariantType.Base))

    branches = [(prompt, mode) for prompt, _cache, mode, _t in transformer.calls]
    assert branches == [
        ("pos", "extract"),
        ("neg", "extract"),
        ("pos", "cached"),
        ("neg", "cached"),
        ("pos", "cached"),
        ("neg", "cached"),
    ]
    caches = {prompt: {cache for p, cache, _m, _t in transformer.calls if p == prompt} for prompt in ("pos", "neg")}
    assert len(caches["pos"]) == len(caches["neg"]) == 1 and caches["pos"] != caches["neg"]


def test_without_guidance_the_negative_prompt_is_never_run() -> None:
    transformer = _Transformer()
    _denoise(cfg_scale=1.0)._run_diffusion(_context(transformer, QwenImage21VariantType.Base))
    assert {prompt for prompt, *_ in transformer.calls} == {"pos"}


def test_image_to_image_on_turbo_starts_at_the_noise_level_of_its_strength() -> None:
    # Turbo's table is front-loaded (six of eight steps above 0.84). A strength of 0.5 must start at 0.5, not at
    # the table's fifth entry (0.895), which would all but discard the input image.
    transformer = _Transformer()
    init = torch.zeros(1, LATENT_CHANNELS, 4, 4)
    _denoise(
        steps=8,
        denoising_start=0.5,
        latents=LatentsField(latents_name="init"),
        # One value per configured step; only the last two are reached from 0.5.
        cfg_scale=[1.0] * 6 + [4.0, 4.0],
    )._run_diffusion(_context(transformer, QwenImage21VariantType.Turbo, init_latents=init))

    positive = [(mode, t) for prompt, _cache, mode, t in transformer.calls if prompt == "pos"]
    assert positive == [("extract", 0.5), ("cached", 0.415)]
    assert sum(prompt == "neg" for prompt, *_ in transformer.calls) == 2


def _loader(**fields) -> QwenImage21ModelLoaderInvocation:
    def ident(key: str, base: BaseModelType, type: ModelType) -> ModelIdentifierField:
        return ModelIdentifierField(key=key, hash=key, name=key, base=base, type=type)

    defaults = {
        "model": ident("main", BaseModelType.QwenImage21, ModelType.Main),
        "vae_model": None,
        "qwen3_vl_encoder_model": None,
    }
    return QwenImage21ModelLoaderInvocation.model_construct(
        **(defaults | {k: ident(*v) if isinstance(v, tuple) else v for k, v in fields.items()})
    )


def _loader_context(main_format: ModelFormat, encoder_variant: Qwen3VLVariantType = Qwen3VLVariantType.Qwen3VL_8B):
    configs = {
        "main": SimpleNamespace(name="main", base=BaseModelType.QwenImage21, type=ModelType.Main, format=main_format),
        "vae": SimpleNamespace(name="vae", base=BaseModelType.QwenImage21, type=ModelType.VAE),
        "wan_vae": SimpleNamespace(name="wan_vae", base=BaseModelType.Wan, type=ModelType.VAE),
        "encoder": SimpleNamespace(name="encoder", type=ModelType.Qwen3VLEncoder, variant=encoder_variant),
    }
    return SimpleNamespace(models=SimpleNamespace(get_config=lambda identifier: configs[identifier.key]))


def test_a_gguf_transformer_without_components_is_refused() -> None:
    with pytest.raises(ValueError, match="no VAE of its own"):
        _loader().invoke(_loader_context(ModelFormat.GGUFQuantized))


def test_a_gguf_transformer_takes_the_standalone_components() -> None:
    output = _loader(
        vae_model=("vae", BaseModelType.QwenImage21, ModelType.VAE),
        qwen3_vl_encoder_model=("encoder", BaseModelType.Any, ModelType.Qwen3VLEncoder),
    ).invoke(_loader_context(ModelFormat.GGUFQuantized))
    assert output.vae.vae.key == "vae"
    assert output.qwen3_vl_encoder.text_encoder.key == "encoder"


def test_another_familys_vae_is_refused() -> None:
    loader = _loader(vae_model=("wan_vae", BaseModelType.Wan, ModelType.VAE))
    with pytest.raises(ValueError, match="not a Qwen-Image-2.1 VAE"):
        loader.invoke(_loader_context(ModelFormat.Diffusers))


def test_the_4b_encoder_is_refused() -> None:
    loader = _loader(qwen3_vl_encoder_model=("encoder", BaseModelType.Any, ModelType.Qwen3VLEncoder))
    with pytest.raises(ValueError, match="8B"):
        loader.invoke(_loader_context(ModelFormat.Diffusers, encoder_variant=Qwen3VLVariantType.Qwen3VL_4B))
