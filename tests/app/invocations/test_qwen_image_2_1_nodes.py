"""The Qwen-Image-2.1 denoise and model-loader nodes, run against stand-ins for the model and the context.

What the GPU parity run cannot see: the order the prefix cache is filled in when guidance changes per step, where
an image-to-image run starts on each variant's schedule, and which components the loader refuses.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.app.invocations.fields import ImageField, LatentsField, MetadataField, QwenImage21ConditioningField
from invokeai.app.invocations.model import ModelIdentifierField, Qwen3VLEncoderField, TransformerField, VAEField
from invokeai.app.invocations.qwen_image_2_1.qwen_image_2_1_denoise import (
    LATENT_CHANNELS,
    PREFIX_CACHE_METADATA_KEY,
    QwenImage21DenoiseInvocation,
    clip_sigmas,
    kv_bytes_per_prefix_token,
    pack_latents,
    prefix_cache_fits,
    prefix_length,
    unpack_latents,
)
from invokeai.app.invocations.qwen_image_2_1.qwen_image_2_1_model_loader import QwenImage21ModelLoaderInvocation
from invokeai.app.invocations.text_encoder.qwen_image_2_1_text_encoder import QwenImage21TextEncoderInvocation
from invokeai.app.invocations.vae.qwen_image_2_1_image_to_latents import QwenImage21ImageToLatentsInvocation
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
    Qwen3VLVariantType,
    QwenImage21VariantType,
    SubModelType,
)
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    QwenImage21ConditioningInfo,
)

DENOISE = "invokeai.app.invocations.qwen_image_2_1.qwen_image_2_1_denoise"
# K and V of one prefix token in the released model: 32 blocks of 32 heads x 128, two bytes each in bf16.
RELEASED_KV_BYTES = 2 * 32 * 4096 * 2


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
        self.inputs: list[SimpleNamespace] = []

    def modules(self):
        return iter(())

    def __call__(
        self,
        *,
        hidden_states,
        encoder_hidden_states,
        timestep,
        kv_cache,
        kv_cache_mode,
        img_shapes,
        img_mask,
        **_kwargs,
    ):
        prompt = "pos" if float(encoder_hidden_states.mean()) > 0 else "neg"
        self.calls.append((prompt, id(kv_cache), kv_cache_mode, round(float(timestep[0]), 3)))
        self.inputs.append(
            SimpleNamespace(hidden_states=hidden_states.clone(), img_shapes=img_shapes, img_mask=img_mask)
        )
        # The transformer returns the joint sequence; the node keeps the image tokens at its end.
        return (torch.zeros(1, encoder_hidden_states.shape[1] + hidden_states.shape[1], hidden_states.shape[2]),)


class _TransformerInfo:
    def __init__(self, transformer: _Transformer, events: list[str]) -> None:
        self.model = transformer
        self.events = events

    @contextmanager
    def model_on_device(self, **_kwargs):
        self.events.append("lock")
        yield ({}, self.model)
        self.events.append("unlock")


def _context(
    transformer: _Transformer,
    variant: QwenImage21VariantType,
    init_latents: torch.Tensor | None = None,
    events: list[str] | None = None,
    *,
    infos: dict[str, QwenImage21ConditioningInfo] | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
):
    events = [] if events is None else events
    infos = infos or {
        "pos": QwenImage21ConditioningInfo(prompt_embeds=torch.ones(1, 3, 8)),
        "neg": QwenImage21ConditioningInfo(prompt_embeds=-torch.ones(1, 2, 8)),
    }
    conditionings = {name: ConditioningFieldData(conditionings=[info]) for name, info in infos.items()}
    tensors = tensors or {}
    return SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: _TransformerInfo(transformer, events),
            get_config=lambda _identifier: SimpleNamespace(variant=variant),
        ),
        conditioning=SimpleNamespace(load=lambda name: conditionings[name]),
        tensors=SimpleNamespace(load=lambda name: tensors.get(name, init_latents)),
        util=SimpleNamespace(sd_step_callback=lambda *_args: events.append("preview")),
    )


def _denoise(**fields) -> QwenImage21DenoiseInvocation:
    model = ModelIdentifierField(key="m", hash="h", name="m", base=BaseModelType.QwenImage21, type=ModelType.Main)
    defaults = {
        "transformer": TransformerField(transformer=model, loras=[]),
        "positive_conditioning": QwenImage21ConditioningField(conditioning_name="pos"),
        "negative_conditioning": QwenImage21ConditioningField(conditioning_name="neg"),
        "latents": None,
        "denoise_mask": None,
        "reference_latents": None,
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


def test_every_preview_is_sent_while_the_transformer_is_locked() -> None:
    """A Tiny AutoEncoder preview places its TAE next to the locked model; before the lock there is none to fit."""
    events: list[str] = []

    _denoise(steps=3)._run_diffusion(_context(_Transformer(), QwenImage21VariantType.Base, events=events))

    assert events == ["lock", "preview", "preview", "preview", "preview", "unlock"]


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


def _released_transformer(num_blocks: int = 32) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=32, attention_head_dim=128), transformer_blocks=[None] * num_blocks
    )


def test_the_prefix_cache_is_priced_from_the_loaded_transformers_shape() -> None:
    assert kv_bytes_per_prefix_token(_released_transformer(), torch.bfloat16) == RELEASED_KV_BYTES == 2**19
    # A shallower model keeps fewer blocks of K and V; a float32 fallback keeps each at twice the width.
    assert kv_bytes_per_prefix_token(_released_transformer(16), torch.bfloat16) == RELEASED_KV_BYTES // 2
    assert kv_bytes_per_prefix_token(_released_transformer(), torch.float32) == 2 * RELEASED_KV_BYTES


def test_a_float32_fallback_reserves_twice_the_bfloat16_activations() -> None:
    estimate = QwenImage21DenoiseInvocation._estimate_working_memory
    base = 1024**3
    bf16_kv = kv_bytes_per_prefix_token(_released_transformer(), torch.bfloat16)
    fp32_kv = kv_bytes_per_prefix_token(_released_transformer(), torch.float32)
    bf16 = estimate(4096, 300, 300, True, True, 2, bf16_kv) - base
    assert estimate(4096, 300, 300, True, True, 4, fp32_kv) - base == 2 * bf16


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_estimate_is_priced_at_the_compute_dtypes_width(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A hard-coded width at the call site would halve the float32 estimate, or double the bf16 one CUDA runs.
    monkeypatch.setattr(f"{DENOISE}.TorchDevice.choose_bfloat16_safe_dtype", lambda _device: dtype)
    widths: list[int] = []
    estimate = QwenImage21DenoiseInvocation._estimate_working_memory

    def spy(*args):
        widths.append(args[5])
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


def _edit_prompt(sign: float, grids: list[tuple[int, int]]) -> QwenImage21ConditioningInfo:
    """A prompt encoded with references: a token, then one run of image slots per reference, then a token."""
    mask = [False]
    for rows, columns in grids:
        mask += [True] * (rows * columns) + [False]
    return QwenImage21ConditioningInfo(
        prompt_embeds=sign * torch.ones(1, len(mask), 8),
        image_pad_mask=torch.tensor([mask]),
        reference_grids=tuple(grids),
    )


def _edit_context(transformer, prompt_grids, references: dict[str, torch.Tensor], negative=None):
    infos = {"pos": _edit_prompt(1.0, prompt_grids), "neg": negative or _edit_prompt(-1.0, prompt_grids)}
    return _context(transformer, QwenImage21VariantType.Base, infos=infos, tensors=references)


def test_references_lead_the_image_tokens_and_fill_the_prompts_slots() -> None:
    transformer = _Transformer()
    # A 4x8 reference is 32 latent tokens in 2x4 slots; the 64x64 target is 4x4.
    reference = torch.randn(1, LATENT_CHANNELS, 4, 8)
    _denoise(reference_latents=[LatentsField(latents_name="ref")], steps=1)._run_diffusion(
        _edit_context(transformer, [(2, 4)], {"ref": reference})
    )

    seen = transformer.inputs[0]
    assert seen.img_shapes == [[(1, 4, 8), (1, 4, 4)]]
    assert seen.hidden_states.shape[1] == 32 + 16
    assert torch.equal(seen.hidden_states[:, :32], pack_latents(reference))
    # The prompt's own mask, then one slot per 2x2 group of the target's 4x4 latents.
    assert seen.img_mask.tolist() == [[False] + [True] * 8 + [False] + [True] * 4]


def test_references_that_do_not_fill_the_slots_are_refused() -> None:
    context = _edit_context(_Transformer(), [(2, 2)], {"ref": torch.zeros(1, LATENT_CHANNELS, 4, 8)})
    with pytest.raises(ValueError, match=r"positive prompt was encoded with 1 reference image\(s\) \(64x64\)"):
        _denoise(reference_latents=LatentsField(latents_name="ref"))._run_diffusion(context)


def test_two_references_of_one_size_in_swapped_order_are_refused() -> None:
    # Landscape then portrait in the prompt; portrait then landscape as latents. The slot counts agree.
    context = _edit_context(
        _Transformer(),
        [(2, 4), (4, 2)],
        {"portrait": torch.zeros(1, LATENT_CHANNELS, 8, 4), "landscape": torch.zeros(1, LATENT_CHANNELS, 4, 8)},
    )
    latents = [LatentsField(latents_name="portrait"), LatentsField(latents_name="landscape")]
    with pytest.raises(ValueError, match="in reference mode, from the same images and in the same order"):
        _denoise(reference_latents=latents)._run_diffusion(context)


def test_latents_from_another_vae_are_refused() -> None:
    context = _edit_context(_Transformer(), [(2, 4)], {"ref": torch.zeros(1, 16, 4, 8)})
    with pytest.raises(ValueError, match="64 channels"):
        _denoise(reference_latents=LatentsField(latents_name="ref"))._run_diffusion(context)


def test_a_negative_prompt_encoded_without_the_references_is_refused() -> None:
    context = _edit_context(
        _Transformer(),
        [(2, 4)],
        {"ref": torch.zeros(1, LATENT_CHANNELS, 4, 8)},
        negative=QwenImage21ConditioningInfo(prompt_embeds=-torch.ones(1, 2, 8)),
    )
    with pytest.raises(ValueError, match="negative prompt was encoded with 0"):
        _denoise(reference_latents=LatentsField(latents_name="ref"), cfg_scale=4.0)._run_diffusion(context)


def test_the_prefix_counts_each_reference_slot_as_four_latent_tokens() -> None:
    # A 1 MP reference is 32x32 slots: 1024 slots in the prompt, 4096 tokens in the joint sequence.
    info = _edit_prompt(1.0, [(32, 32)])
    assert prefix_length(info) == info.prompt_embeds.shape[1] + 3 * 1024


def test_the_prefix_caches_are_reserved_only_when_they_are_kept() -> None:
    estimate = QwenImage21DenoiseInvocation._estimate_working_memory
    prefix = 4200
    cached = estimate(4096, prefix, prefix, True, True, 2, RELEASED_KV_BYTES)
    recomputed = estimate(4096, prefix, prefix, True, False, 2, RELEASED_KV_BYTES)
    # K and V of every prefix token in all 32 blocks, per guidance branch: ~2 GiB per 1 MP reference.
    assert cached - recomputed == 2 * prefix * RELEASED_KV_BYTES


def test_caches_larger_than_half_the_card_are_not_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=24 * 2**30))
    cuda = torch.device("cuda")
    # Two references at CFG fit a 24 GB card; four do not.
    assert prefix_cache_fits(2 * 2 * 4200 * RELEASED_KV_BYTES, cuda)
    assert not prefix_cache_fits(2 * 4 * 4200 * RELEASED_KV_BYTES, cuda)


@pytest.mark.parametrize(("integrated", "fits"), [(False, False), (True, True)], ids=["discrete", "integrated"])
def test_a_discrete_xpu_card_is_limited_like_any_dedicated_card(
    monkeypatch: pytest.MonkeyPatch, integrated: bool, fits: bool
) -> None:
    # Four references at CFG hold ~16 GiB: more than half a 16 GB Arc card. An integrated GPU shares system RAM,
    # which the cache does not budget as a card.
    monkeypatch.setattr("invokeai.backend.util.devices.xpu_device_is_integrated", lambda _device: integrated)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda _device: SimpleNamespace(total_memory=16 * 2**30))
    assert prefix_cache_fits(2 * 4 * 4200 * RELEASED_KV_BYTES, torch.device("xpu")) is fits
    assert prefix_cache_fits(2 * 4 * 4200 * RELEASED_KV_BYTES, torch.device("cpu"))


def test_without_room_for_the_caches_every_step_runs_the_whole_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{DENOISE}.prefix_cache_fits", lambda _bytes, _device: False)
    transformer = _Transformer()
    _denoise(reference_latents=LatentsField(latents_name="ref"), cfg_scale=4.0)._run_diffusion(
        _edit_context(transformer, [(2, 4)], {"ref": torch.zeros(1, LATENT_CHANNELS, 4, 8)})
    )
    assert {(cache, mode) for _prompt, cache, mode, _t in transformer.calls} == {(id(None), None)}


def test_a_later_step_decodes_from_the_cache_its_prefill_filled_with_the_references() -> None:
    transformer = _Transformer()
    reference = torch.randn(1, LATENT_CHANNELS, 4, 8)
    _denoise(reference_latents=[LatentsField(latents_name="ref")], steps=2)._run_diffusion(
        _edit_context(transformer, [(2, 4)], {"ref": reference})
    )

    assert [(prompt, mode) for prompt, _cache, mode, _t in transformer.calls] == [("pos", "extract"), ("pos", "cached")]
    # One cache for the run: the cached step reads what the prefill wrote.
    assert len({cache for _prompt, cache, _mode, _t in transformer.calls}) == 1
    # The transformer rebuilds the joint sequence on every step, so the cached step still needs the references in
    # front of the target, and the same layout.
    for seen in transformer.inputs:
        assert seen.img_shapes == [[(1, 4, 8), (1, 4, 4)]]
        assert torch.equal(seen.hidden_states[:, :32], pack_latents(reference))
        assert seen.img_mask.tolist() == [[False] + [True] * 8 + [False] + [True] * 4]


def test_large_references_before_a_small_target_price_the_last_references_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sides: list[int] = []

    def score_bytes(*, seq_len: int, **_kwargs) -> int:
        sides.append(seq_len)
        return 0

    monkeypatch.setattr(f"{DENOISE}.sdpa_score_matrix_bytes", score_bytes)
    # Four 1 MP references (32x32 slots, 64x64 latents each) before a 256x256 target (16x16 latents).
    grids = [(32, 32)] * 4
    references = {f"ref{i}": torch.zeros(1, LATENT_CHANNELS, 64, 64) for i in range(4)}
    _denoise(
        reference_latents=[LatentsField(latents_name=name) for name in references], width=256, height=256, steps=1
    )._run_diffusion(_edit_context(_Transformer(), grids, references))

    # The prompt's layout: one text token, then each reference's 4096 latent tokens and one text token. The last
    # reference's 4096 queries attend over every key up to its end: 4 * 4096 + 4 of them. The target's call is
    # 256 x (16389 + 256), a sixteenth of that.
    largest = 4096 * (4 * 4096 + 4)
    assert len(sides) == 1
    assert (sides[0] - 1) ** 2 < largest <= sides[0] ** 2


@pytest.mark.parametrize("cache_fits", [True, False], ids=["cached", "recomputed"])
def test_the_metadata_records_whether_the_prefix_was_cached(cache_fits: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{DENOISE}.prefix_cache_fits", lambda _bytes, _device: cache_fits)
    context = _edit_context(_Transformer(), [(2, 4)], {"ref": torch.zeros(1, LATENT_CHANNELS, 4, 8)})
    context.tensors.save = lambda tensor: "out"
    node = _denoise(
        reference_latents=LatentsField(latents_name="ref"),
        cfg_scale=4.0,
        metadata=MetadataField.model_validate({"seed": 7, "steps": 3}),
    )

    output = node.invoke(context)

    # What the graph recorded passes through, with the mode this run took beside it.
    assert output.metadata.root == {"seed": 7, "steps": 3, PREFIX_CACHE_METADATA_KEY: cache_fits}
    assert (output.latents.latents_name, output.width, output.height) == ("out", 64, 64)


def test_a_standalone_encoder_cannot_read_references() -> None:
    encoder = ModelIdentifierField(
        key="enc", hash="h", name="qwen3vl_8b", base=BaseModelType.Any, type=ModelType.Qwen3VLEncoder
    )
    node = QwenImage21TextEncoderInvocation.model_construct(
        prompt="swap them",
        qwen3_vl_encoder=Qwen3VLEncoderField(tokenizer=encoder, text_encoder=encoder, loras=[]),
        reference_images=[ImageField(image_name="ref.png")],
    )
    context = SimpleNamespace(
        images=SimpleNamespace(get_pil=lambda _name: Image.new("RGB", (64, 64))),
        models=SimpleNamespace(
            get_config=lambda _identifier: SimpleNamespace(
                base=BaseModelType.Any, type=ModelType.Qwen3VLEncoder, format=ModelFormat.Checkpoint
            )
        ),
    )
    with pytest.raises(ValueError, match="vision tower"):
        node.invoke(context)


def _pipeline_encoder_node(num_references: int) -> QwenImage21TextEncoderInvocation:
    main = ModelIdentifierField(
        key="qwen21", hash="h", name="Qwen-Image-2.1", base=BaseModelType.QwenImage21, type=ModelType.Main
    )
    return QwenImage21TextEncoderInvocation.model_construct(
        prompt="Combine the subjects of all reference images.",
        qwen3_vl_encoder=Qwen3VLEncoderField(
            tokenizer=main.model_copy(update={"submodel_type": SubModelType.Tokenizer}),
            text_encoder=main.model_copy(update={"submodel_type": SubModelType.TextEncoder}),
            loras=[],
        ),
        reference_images=[ImageField(image_name=f"ref{i}.png") for i in range(num_references)],
    )


def test_the_processor_comes_from_the_qwen_image_2_1_pipeline_the_tokenizer_is_from() -> None:
    context = SimpleNamespace(
        models=SimpleNamespace(
            get_config=lambda _identifier: SimpleNamespace(
                base=BaseModelType.QwenImage21, type=ModelType.Main, format=ModelFormat.Diffusers
            )
        )
    )
    tower, processor = _pipeline_encoder_node(1)._edit_identifiers(context)
    assert (tower.key, tower.submodel_type) == ("qwen21", SubModelType.VisionEncoder)
    assert (processor.key, processor.submodel_type) == ("qwen21", SubModelType.Processor)


def test_a_standalone_encoder_beside_the_pipelines_tokenizer_cannot_read_references() -> None:
    # The tokenizer passes; the encoder, whose pipeline would have to bring the vision tower, does not.
    def get_config(identifier):
        if identifier.submodel_type is SubModelType.Tokenizer:
            return SimpleNamespace(base=BaseModelType.QwenImage21, type=ModelType.Main, format=ModelFormat.Diffusers)
        return SimpleNamespace(base=BaseModelType.Any, type=ModelType.Qwen3VLEncoder, format=ModelFormat.Checkpoint)

    with pytest.raises(ValueError, match="vision tower"):
        _pipeline_encoder_node(1)._edit_identifiers(SimpleNamespace(models=SimpleNamespace(get_config=get_config)))


def test_more_references_than_the_architecture_reads_are_refused_before_anything_loads() -> None:
    # The context has no images or models: reaching either would fail with another error.
    with pytest.raises(ValueError, match="at most 4 reference images; this prompt has 5"):
        _pipeline_encoder_node(5).invoke(SimpleNamespace())


class _ReservationRecorded(Exception):
    pass


def _encode_until_reservation(num_references: int, image_size: int = 1024) -> tuple[list[int], list[SubModelType]]:
    """Runs the node until the encoder would move to the device; returns what it reserved and what it loaded."""
    from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_tokenizer

    reserved: list[int] = []
    loaded: list[SubModelType] = []

    @contextmanager
    def on_device(*, working_mem_bytes: int):
        reserved.append(working_mem_bytes)
        raise _ReservationRecorded
        yield  # pragma: no cover

    text_encoder = torch.nn.Module()
    text_encoder.language_model = torch.nn.Linear(2, 2)
    text_encoder.visual = torch.nn.Identity()
    tower = torch.nn.Linear(2, 2)
    # The released tower's attention geometry.
    tower.config = SimpleNamespace(num_heads=16, hidden_size=1152)
    infos = {
        SubModelType.TextEncoder: SimpleNamespace(
            model=text_encoder, compute_device=torch.device("cpu"), model_on_device=on_device
        ),
        SubModelType.VisionEncoder: SimpleNamespace(
            model=tower, compute_device=torch.device("cpu"), model_on_device=lambda: nullcontext((None, tower))
        ),
        SubModelType.Tokenizer: nullcontext(load_bundled_qwen3_vl_tokenizer()),
        SubModelType.Processor: SimpleNamespace(model_on_device=lambda: nullcontext((None, None))),
    }

    def load(identifier):
        loaded.append(identifier.submodel_type)
        return infos[identifier.submodel_type]

    context = SimpleNamespace(
        images=SimpleNamespace(get_pil=lambda _name: Image.new("RGB", (image_size, image_size))),
        models=SimpleNamespace(
            get_config=lambda _identifier: SimpleNamespace(
                base=BaseModelType.QwenImage21, type=ModelType.Main, format=ModelFormat.Diffusers
            ),
            load=load,
        ),
        util=SimpleNamespace(signal_progress=lambda _message: None),
    )
    with pytest.raises(_ReservationRecorded):
        _pipeline_encoder_node(num_references).invoke(context)
    return reserved, loaded


@pytest.mark.parametrize("num_references", [0, 2], ids=["text-only", "edit"])
def test_only_an_edit_loads_the_vision_tower(num_references: int) -> None:
    _, loaded = _encode_until_reservation(num_references, image_size=64)
    # The encoder is the tower-less one either way: its language model is cached once for both.
    assert loaded[0] is SubModelType.TextEncoder and SubModelType.TextEncoder not in loaded[1:]
    assert (SubModelType.VisionEncoder in loaded) is bool(num_references)
    assert (SubModelType.Processor in loaded) is bool(num_references)


class _EncodeFailed(Exception):
    pass


@pytest.mark.parametrize("fail", [False, True], ids=["encoded", "failed"])
def test_an_edit_encodes_with_the_tower_attached_and_detaches_it(fail: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through the node with a tiny Qwen3-VL split the way the pipeline loader caches it."""
    import copy

    from transformers import Qwen2VLImageProcessor, Qwen3VLModel, Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    from invokeai.app.invocations.text_encoder import qwen_image_2_1_text_encoder as node_module
    from invokeai.backend.model_manager.util.qwen3_vl import drop_qwen3vl_visual_tower
    from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_tokenizer
    from invokeai.backend.qwen_image_2_1.text_encoding import encode_prompt
    from invokeai.backend.qwen_image_2_1.vae import to_reference
    from tests.backend.model_manager.load.qwen3vl_gguf_fixture import tiny_qwen3vl_config

    monkeypatch.setattr(f"{node_module.__name__}.TorchDevice.choose_bfloat16_safe_dtype", lambda _d: torch.float32)
    tokenizer = load_bundled_qwen3_vl_tokenizer()
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
    )
    config = tiny_qwen3vl_config()
    config.text_config.vocab_size = len(tokenizer)
    torch.manual_seed(0)
    built_in = Qwen3VLModel(config).float().eval()
    tower = copy.deepcopy(built_in.visual)
    encoder = copy.deepcopy(built_in)
    drop_qwen3vl_visual_tower(encoder)

    def info(model):
        return SimpleNamespace(
            model=model, compute_device=torch.device("cpu"), model_on_device=lambda **_kw: nullcontext((None, model))
        )

    infos = {
        SubModelType.TextEncoder: info(encoder),
        SubModelType.VisionEncoder: info(tower),
        SubModelType.Tokenizer: nullcontext(tokenizer),
        SubModelType.Processor: info(processor),
    }
    saved: list[ConditioningFieldData] = []
    reference = Image.new("RGB", (64, 64), (200, 30, 30))
    context = SimpleNamespace(
        images=SimpleNamespace(get_pil=lambda _name: reference),
        models=SimpleNamespace(
            get_config=lambda _identifier: SimpleNamespace(
                base=BaseModelType.QwenImage21, type=ModelType.Main, format=ModelFormat.Diffusers
            ),
            load=lambda identifier: infos[identifier.submodel_type],
        ),
        conditioning=SimpleNamespace(save=lambda data: saved.append(data) or "cond"),
        util=SimpleNamespace(signal_progress=lambda _message: None),
    )
    node = _pipeline_encoder_node(1)
    if fail:

        def failing_encode(*_args, **_kwargs):
            raise _EncodeFailed

        monkeypatch.setattr(node_module, "encode_prompt", failing_encode)
        with pytest.raises(_EncodeFailed):
            node.invoke(context)
    else:
        node.invoke(context)
        expected = encode_prompt(
            built_in, tokenizer, node.prompt, torch.device("cpu"), processor=processor, images=[to_reference(reference)]
        )
        info_out = saved[0].conditionings[0]
        assert torch.equal(info_out.prompt_embeds, expected.embeds)
        assert torch.equal(info_out.image_pad_mask, expected.image_pad_mask)

    # The cached encoder is left as it was loaded, whether the encode finished or not.
    assert isinstance(encoder.visual, torch.nn.Identity)


@pytest.mark.parametrize("num_references", [0, 4])
def test_the_encoder_reserves_room_for_the_attention_its_references_need(
    num_references: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The measurement below is a bf16 one; on a CPU without bf16 the node would price float32, twice as much.
    monkeypatch.setattr(
        "invokeai.app.invocations.text_encoder.qwen_image_2_1_text_encoder.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.bfloat16,
    )
    reserved, _ = _encode_until_reservation(num_references)

    if num_references:
        # Measured on an RTX 4090: four 1-megapixel references peak 5.44 GiB above the weights, and the allocator
        # reserves about a fifth more. Less than that and the driver spills the encode to system memory.
        assert reserved[0] >= 1.2 * 5.44 * 2**30
    else:
        assert reserved[0] < 2**28


def test_a_float32_encoder_reserves_twice_the_bfloat16_room(monkeypatch: pytest.MonkeyPatch) -> None:
    # The encoder runs in float32 where the device has no bf16 (CPU, older MPS); its activations are twice as wide.
    reserved = {}
    for dtype in (torch.bfloat16, torch.float32):
        monkeypatch.setattr(
            "invokeai.app.invocations.text_encoder.qwen_image_2_1_text_encoder.TorchDevice.choose_bfloat16_safe_dtype",
            lambda _device, dtype=dtype: dtype,
        )
        reserved[dtype] = _encode_until_reservation(4)[0][0]
    assert reserved[torch.float32] == pytest.approx(2 * reserved[torch.bfloat16], rel=1e-6)


class _VAE:
    """Records what it encodes and answers with bf16 latents, as the real VAE runs in bf16."""

    dtype = torch.bfloat16
    config = SimpleNamespace(latents_mean=[0.3] * LATENT_CHANNELS, latents_std=[1.7] * LATENT_CHANNELS)

    def __init__(self) -> None:
        self.pixels: torch.Tensor | None = None
        self.z: torch.Tensor | None = None

    def parameters(self):
        return iter([torch.zeros(1, dtype=torch.bfloat16)])

    def encode(self, pixels: torch.Tensor):
        self.pixels = pixels
        _, _, frames, height, width = pixels.shape
        generator = torch.Generator().manual_seed(0)
        self.z = torch.randn(1, LATENT_CHANNELS, frames, height // 16, width // 16, generator=generator).bfloat16()
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: self.z))


def test_a_reference_encodes_at_its_reading_size_with_its_alpha_normalized_in_bf16(monkeypatch) -> None:
    from invokeai.app.invocations.vae import qwen_image_2_1_image_to_latents as i2l_module

    monkeypatch.setattr(i2l_module, "patch_qwen_image_vae_tiling", lambda *_args: nullcontext())
    vae = _VAE()
    saved: dict[str, torch.Tensor] = {}

    @contextmanager
    def on_device(**_kwargs):
        yield ({}, vae)

    context = SimpleNamespace(
        images=SimpleNamespace(get_pil=lambda _name: Image.new("RGBA", (500, 300), (255, 0, 0, 64))),
        models=SimpleNamespace(
            load=lambda _identifier: SimpleNamespace(
                compute_device=torch.device("cpu"), model=vae, model_on_device=on_device
            )
        ),
        config=SimpleNamespace(get=lambda: SimpleNamespace(force_tiled_decode=False, auto_tiled_decode=False)),
        tensors=SimpleNamespace(save=lambda tensor: saved.update(latents=tensor) or "latents"),
    )
    vae_field = VAEField(
        vae=ModelIdentifierField(key="v", hash="h", name="v", base=BaseModelType.QwenImage21, type=ModelType.VAE)
    )
    output = QwenImage21ImageToLatentsInvocation.model_construct(
        image=ImageField(image_name="ref.png"), vae=vae_field, tiled=False, tile_size=0, reference=True
    ).invoke(context)

    # 500x300 reads at 1312x800; the alpha is the image's own (64/255), not the canvas encode's 1.
    assert (output.width, output.height) == (1312, 800)
    assert vae.pixels is not None and vae.pixels.shape == (1, 4, 1, 800, 1312)
    assert vae.pixels[0, 3].unique().tolist() == [torch.tensor(64 / 255 * 2 - 1).bfloat16().item()]
    # Normalized in bf16, as the pipeline normalizes it -- which differs from normalizing in fp32.
    mean, std = torch.tensor(0.3).bfloat16(), torch.tensor(1.7).bfloat16()
    expected = ((vae.z - mean) / std).float()[:, :, 0]
    assert torch.equal(saved["latents"], expected)
    assert not torch.equal(expected, ((vae.z.float() - 0.3) / 1.7)[:, :, 0])
