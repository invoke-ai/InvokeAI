"""The Qwen-Image-2.1 denoise and model-loader nodes, run against stand-ins for the model and the context.

What the GPU parity run cannot see: the order the prefix cache is filled in when guidance changes per step, where
an image-to-image run starts on each variant's schedule, and which components the loader refuses.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from invokeai.app.invocations.fields import LatentsField, QwenImage21ConditioningField
from invokeai.app.invocations.model import ModelIdentifierField, TransformerField
from invokeai.app.invocations.qwen_image_2_1.qwen_image_2_1_denoise import (
    LATENT_CHANNELS,
    QwenImage21DenoiseInvocation,
    clip_sigmas,
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


@pytest.mark.parametrize(
    ("cfg_scale", "negative_calls"),
    # One value per configured step; only the last two are reached from 0.5. Guidance on just those two runs the
    # negative prompt twice; guidance on just the skipped ones, never -- a slice off by one would run it once.
    [([1.0] * 6 + [4.0, 4.0], 2), ([4.0] * 6 + [1.0, 1.0], 0)],
    ids=["guided-tail", "guided-head"],
)
def test_image_to_image_on_turbo_starts_at_the_noise_level_of_its_strength(
    cfg_scale: list[float], negative_calls: int
) -> None:
    # Turbo's table is front-loaded (six of eight steps above 0.84). A strength of 0.5 must start at 0.5, not at
    # the table's fifth entry (0.895), which would all but discard the input image.
    transformer = _Transformer()
    init = torch.zeros(1, LATENT_CHANNELS, 4, 4)
    _denoise(
        steps=8, denoising_start=0.5, latents=LatentsField(latents_name="init"), cfg_scale=cfg_scale
    )._run_diffusion(_context(transformer, QwenImage21VariantType.Turbo, init_latents=init))

    positive = [(mode, t) for prompt, _cache, mode, t in transformer.calls if prompt == "pos"]
    assert positive == [("extract", 0.5), ("cached", 0.415)]
    assert sum(prompt == "neg" for prompt, *_ in transformer.calls) == negative_calls


def test_image_to_image_on_base_starts_at_the_step_its_strength_names() -> None:
    # Base picks by step index, as Qwen-Image and diffusers' image-to-image pipelines do: a strength of 0.5 skips
    # the first half of the steps. Its shifted schedule is still well above 0.5 there.
    from invokeai.backend.qwen_image_2_1.sampling import build_sigmas

    schedule = build_sigmas(QwenImage21VariantType.Base, 8, (64 // 16) * (64 // 16)).tolist()
    transformer = _Transformer()
    _denoise(steps=8, denoising_start=0.5, latents=LatentsField(latents_name="init"))._run_diffusion(
        _context(transformer, QwenImage21VariantType.Base, init_latents=torch.zeros(1, LATENT_CHANNELS, 4, 4))
    )

    starts = [t for prompt, _cache, _mode, t in transformer.calls if prompt == "pos"]
    assert starts == pytest.approx(schedule[4:8], abs=1e-3)
    assert starts[0] > 0.5


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0.0, 1.0, (0, 8)),
        (0.5, 1.0, (4, 4)),
        # Rounded to the nearest step, as Qwen-Image's own denoise rounds: 5.6 -> 6, 4.8 -> 5.
        (0.7, 1.0, (6, 2)),
        (0.0, 0.6, (0, 5)),
        (0.99, 1.0, (7, 1)),
        (0.0, 0.01, (0, 1)),
        (0.25, 0.75, (2, 4)),
    ],
)
def test_a_strength_on_base_names_whole_steps_and_always_runs_one(
    start: float, end: float, expected: tuple[int, int]
) -> None:
    sigmas = [1.0 - i / 8 for i in range(9)]
    clipped, first_step = clip_sigmas(sigmas, start, end, by_noise_level=False)
    assert (first_step, len(clipped) - 1) == expected
    assert clipped == sigmas[first_step : first_step + len(clipped)]


def test_a_float32_fallback_reserves_twice_the_bfloat16_activations() -> None:
    estimate = QwenImage21DenoiseInvocation._estimate_working_memory
    base = 1024**3
    bf16 = estimate(4096, 300, 300, True, 2) - base
    assert estimate(4096, 300, 300, True, 4) - base == 2 * bf16


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_estimate_is_priced_at_the_compute_dtypes_width(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A hard-coded width at the call site would halve the float32 estimate, or double the bf16 one CUDA runs.
    monkeypatch.setattr(f"{DENOISE}.TorchDevice.choose_bfloat16_safe_dtype", lambda _device: dtype)
    widths: list[int] = []
    estimate = QwenImage21DenoiseInvocation._estimate_working_memory

    def spy(*args):
        widths.append(args[-1])
        return estimate(*args)

    monkeypatch.setattr(QwenImage21DenoiseInvocation, "_estimate_working_memory", staticmethod(spy))
    _denoise()._run_diffusion(_context(_Transformer(), QwenImage21VariantType.Base))
    assert widths == [dtype.itemsize]


class _DecodingVAE:
    """A bf16 VAE that records what it decodes and answers with a fixed opaque bf16 decode."""

    dtype = torch.bfloat16
    use_tiling = False
    tile_sample_min_height = tile_sample_min_width = 256
    tile_sample_stride_height = tile_sample_stride_width = 192

    def __init__(self, mean: list[float], std: list[float]) -> None:
        self.config = SimpleNamespace(latents_mean=mean, latents_std=std)
        self.decoded_from: torch.Tensor | None = None
        generator = torch.Generator().manual_seed(1)
        self.decode_output = (torch.rand(1, 4, 1, 64, 64, generator=generator) * 2.4 - 1.2).to(torch.bfloat16)
        self.decode_output[:, 3] = 1.0

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.bfloat16)])

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def decode(self, z: torch.Tensor, return_dict: bool = False):
        self.decoded_from = z
        return (self.decode_output,)


def test_a_decode_reads_and_writes_exactly_what_the_pipeline_does() -> None:
    from diffusers.image_processor import VaeImageProcessor

    from invokeai.app.invocations.vae.qwen_image_2_1_latents_to_image import QwenImage21LatentsToImageInvocation

    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(1, LATENT_CHANNELS, 4, 4, generator=generator)
    mean = torch.randn(LATENT_CHANNELS, generator=generator).tolist()
    std = (torch.rand(LATENT_CHANNELS, generator=generator) * 3).tolist()
    vae = _DecodingVAE(mean, std)
    saved: list = []

    @contextmanager
    def on_device(**_kwargs):
        yield None, vae

    def save(image):
        saved.append(image)
        return SimpleNamespace(image_name="out.png", width=image.width, height=image.height)

    context = SimpleNamespace(
        tensors=SimpleNamespace(load=lambda _name: latents),
        images=SimpleNamespace(save=save),
        models=SimpleNamespace(
            load=lambda _identifier: SimpleNamespace(
                compute_device=torch.device("cpu"), model=vae, model_on_device=on_device
            )
        ),
        config=SimpleNamespace(get=lambda: SimpleNamespace(force_tiled_decode=False, auto_tiled_decode=False)),
        util=SimpleNamespace(signal_progress=lambda _message: None),
    )
    node = QwenImage21LatentsToImageInvocation.model_construct(
        latents=LatentsField(latents_name="latents"), vae=SimpleNamespace(vae=None), tiled=False, tile_size=0
    )

    node.invoke(context)

    # In: the pipeline's order -- latents to bf16, the float32 statistics to bf16, the multiply-add in bf16. In
    # float32 first, a bf16 decode reads a different input for most elements.
    bf16_mean = torch.tensor(mean).view(1, -1, 1, 1, 1).to(torch.bfloat16)
    bf16_std = torch.tensor(std).view(1, -1, 1, 1, 1).to(torch.bfloat16)
    expected = latents.to(torch.bfloat16).unsqueeze(2) * bf16_std + bf16_mean
    float32_first = (
        latents.unsqueeze(2) * torch.tensor(std).view(1, -1, 1, 1, 1) + torch.tensor(mean).view(1, -1, 1, 1, 1)
    ).to(torch.bfloat16)
    assert not torch.equal(expected, float32_first)
    assert torch.equal(vae.decoded_from, expected)
    # Out: the pixels diffusers' own post-processing makes of the same bf16 decode.
    pipeline = VaeImageProcessor(vae_scale_factor=16).postprocess(vae.decode_output[:, :, 0], output_type="pil")[0]
    assert np.array_equal(np.asarray(saved[0]), np.asarray(pipeline.convert("RGB")))


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
