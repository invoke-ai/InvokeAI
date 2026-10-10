"""The Qwen-Image-2.1 denoise and model-loader nodes, run against stand-ins for the model and the context.

What the GPU parity run cannot see: the order the prefix cache is filled in when guidance changes per step, where
an image-to-image run starts on each variant's schedule, and which components the loader refuses.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from invokeai.app.invocations.fields import ImageField, LatentsField, QwenImage21ConditioningField
from invokeai.app.invocations.model import ModelIdentifierField, Qwen3VLEncoderField, TransformerField, VAEField
from invokeai.app.invocations.qwen_image_2_1.qwen_image_2_1_denoise import (
    KV_BYTES_PER_PREFIX_TOKEN,
    LATENT_CHANNELS,
    QwenImage21DenoiseInvocation,
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
    def __init__(self, transformer: _Transformer) -> None:
        self.model = transformer

    @contextmanager
    def model_on_device(self, **_kwargs):
        yield ({}, self.model)


def _context(
    transformer: _Transformer,
    variant: QwenImage21VariantType,
    init_latents: torch.Tensor | None = None,
    *,
    infos: dict[str, QwenImage21ConditioningInfo] | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
):
    infos = infos or {
        "pos": QwenImage21ConditioningInfo(prompt_embeds=torch.ones(1, 3, 8)),
        "neg": QwenImage21ConditioningInfo(prompt_embeds=-torch.ones(1, 2, 8)),
    }
    conditionings = {name: ConditioningFieldData(conditionings=[info]) for name, info in infos.items()}
    tensors = tensors or {}
    return SimpleNamespace(
        models=SimpleNamespace(
            load=lambda _identifier: _TransformerInfo(transformer),
            get_config=lambda _identifier: SimpleNamespace(variant=variant),
        ),
        conditioning=SimpleNamespace(load=lambda name: conditionings[name]),
        tensors=SimpleNamespace(load=lambda name: tensors.get(name, init_latents)),
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
    cached = estimate(4096, prefix, prefix, True, True)
    recomputed = estimate(4096, prefix, prefix, True, False)
    # K and V of every prefix token in all 32 blocks, per guidance branch: ~2 GiB per 1 MP reference.
    assert cached - recomputed == 2 * prefix * KV_BYTES_PER_PREFIX_TOKEN


def test_caches_larger_than_half_the_card_are_not_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=24 * 2**30))
    cuda = torch.device("cuda")
    # Two references at CFG fit a 24 GB card; four do not.
    assert prefix_cache_fits(2 * 2 * 4200 * KV_BYTES_PER_PREFIX_TOKEN, cuda)
    assert not prefix_cache_fits(2 * 4 * 4200 * KV_BYTES_PER_PREFIX_TOKEN, cuda)


def test_without_room_for_the_caches_every_step_runs_the_whole_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{DENOISE}.prefix_cache_fits", lambda _bytes, _device: False)
    transformer = _Transformer()
    _denoise(reference_latents=LatentsField(latents_name="ref"), cfg_scale=4.0)._run_diffusion(
        _edit_context(transformer, [(2, 4)], {"ref": torch.zeros(1, LATENT_CHANNELS, 4, 8)})
    )
    assert {(cache, mode) for _prompt, cache, mode, _t in transformer.calls} == {(id(None), None)}


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


def test_the_processor_comes_from_the_qwen_image_2_1_pipeline_the_tokenizer_is_from() -> None:
    main = ModelIdentifierField(
        key="qwen21", hash="h", name="Qwen-Image-2.1", base=BaseModelType.QwenImage21, type=ModelType.Main
    )
    tokenizer = main.model_copy(update={"submodel_type": SubModelType.Tokenizer})
    node = QwenImage21TextEncoderInvocation.model_construct(
        prompt="swap them",
        qwen3_vl_encoder=Qwen3VLEncoderField(tokenizer=tokenizer, text_encoder=tokenizer, loras=[]),
        reference_images=[ImageField(image_name="ref.png")],
    )
    context = SimpleNamespace(
        models=SimpleNamespace(
            get_config=lambda _identifier: SimpleNamespace(
                base=BaseModelType.QwenImage21, type=ModelType.Main, format=ModelFormat.Diffusers
            )
        )
    )
    processor = node._processor_identifier(context)
    assert (processor.key, processor.submodel_type) == ("qwen21", SubModelType.Processor)


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
