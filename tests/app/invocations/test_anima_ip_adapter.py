"""The Anima IP-Adapter through its encoder node and the denoise node.

Denoise follows the reference node's default CFG mode: the positive pass gets the reference image, the negative pass
gets none, and the bundled LoRA is applied once for the whole run. The transformer here records what it is handed.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.anima.anima_denoise import AnimaDenoiseInvocation
from invokeai.app.invocations.anima.anima_ip_adapter import AnimaIPAdapterField, AnimaIPAdapterInvocation
from invokeai.app.invocations.fields import AnimaConditioningField
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.backend.anima.ip_adapter import AnimaIPAdapter
from invokeai.backend.model_manager.starter_models.common import siglip, siglip2_base_512
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import AnimaConditioningInfo, ConditioningFieldData

_DEPTH = 28


class _Loaded:
    def __init__(self, model):
        self.model = model
        self.compute_device = torch.device("cpu")
        self.config = SimpleNamespace(format=ModelFormat.Checkpoint)

    @contextmanager
    def model_on_device(self, working_mem_bytes=None):
        yield (None, self.model)


class _FakeTransformer(torch.nn.Module):
    """Records the IP-Adapter terms of each call; predicts zero velocity."""

    def __init__(self, depth: int = _DEPTH) -> None:
        super().__init__()
        self.has_semantic_connector = False
        self.patch_spatial = 2
        self.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(depth)])
        for block in self.blocks:
            block.cross_attn = torch.nn.Identity()
        self.calls: list[list | None] = []

    def preprocess_text_embeds(self, text_embeds, text_ids, t5xxl_weights=None, **_kwargs):
        return torch.zeros(1, 512, 1024)

    def forward(self, x, timesteps, context, ip_adapter_blocks=None, **kwargs):
        self.calls.append(ip_adapter_blocks)
        return torch.zeros_like(x)


def _adapter() -> AnimaIPAdapter:
    sd: dict[str, torch.Tensor] = {}
    for n in range(_DEPTH):
        sd |= {
            f"blocks.{n}.ip_k_proj.weight": torch.zeros(8, 4),
            f"blocks.{n}.ip_k_proj.bias": torch.zeros(8),
            f"blocks.{n}.ip_v_proj.weight": torch.zeros(8, 4),
            f"blocks.{n}.ip_v_proj.bias": torch.zeros(8),
            f"blocks.{n}.adaln_ip.1.weight": torch.zeros(8, 8),
            f"blocks.{n}.adaln_ip.1.bias": torch.zeros(8),
        }
    sd["lora.base_model.model.blocks.0.cross_attn.q_proj.lora_A.weight"] = torch.zeros(2, 8)
    sd["lora.base_model.model.blocks.0.cross_attn.q_proj.lora_B.weight"] = torch.zeros(8, 2)
    return AnimaIPAdapter.from_state_dict(sd)


def _identifier(key: str) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash="h", name=key, base=BaseModelType.Anima, type=ModelType.IPAdapter)


def _field(key: str = "ipa", tokens_name: str = "tokens", **kwargs) -> AnimaIPAdapterField:
    kwargs.setdefault("image_name", f"{tokens_name}.png")
    return AnimaIPAdapterField(tokens_name=tokens_name, ip_adapter_model=_identifier(key), **kwargs)


# Each tokens name loads as three tokens holding one value, so a block term's tokens show which images it attends.
_TOKEN_VALUES = {"tokens": 1.0, "first": 1.0, "second": 2.0}


def _attended(block_fn) -> torch.Tensor:
    """The image tokens a block term attends: the last argument `AnimaIPAdapterExtension.block_fns` binds."""
    return block_fn.args[-1]


def _denoise(
    monkeypatch,
    ip_adapter,
    *,
    steps: int = 4,
    scheduler: str = "euler",
    transformer_depth: int = _DEPTH,
    semantic_connector: bool = False,
    context: MagicMock | None = None,
):
    module = "invokeai.app.invocations.anima.anima_denoise"
    monkeypatch.setattr(f"{module}.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{module}.TorchDevice.choose_anima_inference_dtype", lambda _d: torch.float32)
    patches: list = []

    def apply_smart_model_patches(**kwargs):
        patches.extend(kwargs["patches"])
        return nullcontext()

    monkeypatch.setattr(f"{module}.LayerPatcher.apply_smart_model_patches", apply_smart_model_patches)
    monkeypatch.setattr(f"{module}.patch_anima_for_regional_prompting", lambda *_a: nullcontext())
    transformer = _FakeTransformer(transformer_depth)
    transformer.has_semantic_connector = semantic_connector
    adapters: dict[str, AnimaIPAdapter] = {}
    transformer_id = object()

    def load(identifier):
        if identifier is transformer_id:
            return _Loaded(transformer)
        assert identifier.key not in adapters, f"adapter '{identifier.key}' loaded twice"
        return _Loaded(adapters.setdefault(identifier.key, _adapter()))

    context = context or MagicMock()
    context.models.load.side_effect = load
    context.conditioning.load.side_effect = lambda _name: ConditioningFieldData(
        conditionings=[
            AnimaConditioningInfo(
                qwen3_embeds=torch.zeros(3, 1024),
                t5xxl_ids=torch.zeros(3).long(),
                qwen35_states=torch.zeros(4, 3, 2560) if semantic_connector else None,
            )
        ]
    )
    context.tensors.load.side_effect = lambda name: torch.full((1, 3, 4), _TOKEN_VALUES[name])

    invocation = AnimaDenoiseInvocation.model_construct(
        latents=None, noise=None, denoise_mask=None, denoising_start=0.0, denoising_end=1.0, add_noise=True,
        transformer=SimpleNamespace(transformer=transformer_id, loras=[]),
        positive_conditioning=AnimaConditioningField(conditioning_name="pos"),
        negative_conditioning=AnimaConditioningField(conditioning_name="neg"),
        guidance_scale=6.0, width=64, height=64, steps=steps, seed=0, control_lllite=None, scheduler=scheduler,
        ip_adapter=ip_adapter,
    )  # fmt: skip
    invocation._run_diffusion(context)
    return transformer, patches, adapters


@pytest.mark.parametrize("scheduler", ["euler", "dpmpp_2m"])
def test_the_positive_pass_gets_the_image_and_the_negative_pass_does_not(monkeypatch, scheduler) -> None:
    transformer, patches, adapters = _denoise(monkeypatch, _field(), scheduler=scheduler)

    cond, uncond = transformer.calls[0::2], transformer.calls[1::2]
    assert len(cond) == len(uncond) == 4
    assert all(blocks is not None and len(blocks) == _DEPTH and all(blocks) for blocks in cond)
    assert all(blocks is None for blocks in uncond)
    # Its bundled LoRA, once, at 1.0.
    assert patches == [(adapters["ipa"].lora_patch(), 1.0)]
    assert len(adapters["ipa"].lora_patch().layers) == 1


@pytest.mark.parametrize("scheduler", ["euler", "dpmpp_2m"])
def test_the_step_window_turns_the_term_off_outside_it(monkeypatch, scheduler) -> None:
    # 4 steps, from 50%: floor(0.5 * 4) = 2 is the first step that gets it, as for ControlNet-LLLite.
    transformer, _, _ = _denoise(monkeypatch, _field(begin_step_percent=0.5), scheduler=scheduler)

    assert [blocks is not None for blocks in transformer.calls[0::2]] == [False, False, True, True]


def test_strength_zero_is_no_adapter_at_all(monkeypatch) -> None:
    transformer, patches, adapters = _denoise(monkeypatch, [_field(strength=0.0)])

    assert all(blocks is None for blocks in transformer.calls)
    assert patches == []
    assert adapters == {}, "a strength-0 adapter should not even be loaded"


def test_two_adapters_each_bring_their_lora(monkeypatch) -> None:
    transformer, patches, adapters = _denoise(monkeypatch, [_field("b"), _field("a", end_step_percent=0.0)])

    assert patches == [(adapters["a"].lora_patch(), 1.0), (adapters["b"].lora_patch(), 1.0)]
    assert all(blocks is not None for blocks in transformer.calls[0::2])


def test_an_adapter_deeper_than_the_transformer_is_refused(monkeypatch) -> None:
    """Its bundled LoRA would address blocks the transformer does not have."""
    with pytest.raises(ValueError, match="'ipa' has 28 blocks, more than this 20-block Anima model"):
        _denoise(monkeypatch, [_field()], transformer_depth=20)


def test_images_sharing_an_adapter_are_attended_together_with_its_lora_applied_once(monkeypatch) -> None:
    transformer, patches, adapters = _denoise(
        monkeypatch, [_field(tokens_name="first", strength=0.5), _field(tokens_name="second", strength=1.5)]
    )

    assert list(adapters) == ["ipa"]
    assert patches == [(adapters["ipa"].lora_patch(), 1.0)]
    expected = torch.cat([torch.full((1, 3, 4), 0.5), torch.full((1, 3, 4), 3.0)], dim=1)
    for blocks in transformer.calls[0::2]:
        assert blocks is not None and len(blocks) == _DEPTH
        assert all(torch.equal(_attended(fn), expected) for fn in blocks)


def test_each_image_keeps_its_own_step_window(monkeypatch) -> None:
    # 4 steps, both bounds inclusive: the first image runs to ceil(0.5 * 4) = 2, the second from floor(0.25 * 4) = 1.
    transformer, _, _ = _denoise(
        monkeypatch,
        [
            _field(tokens_name="first", end_step_percent=0.5),
            _field(tokens_name="second", begin_step_percent=0.25),
        ],
    )

    first, second = torch.full((1, 3, 4), 1.0), torch.full((1, 3, 4), 2.0)
    attended = [_attended(blocks[0]) for blocks in transformer.calls[0::2]]
    assert [tokens.shape[1] for tokens in attended] == [3, 6, 6, 3]
    assert torch.equal(attended[0], first)
    assert torch.equal(attended[1], torch.cat([first, second], dim=1))
    assert torch.equal(attended[3], second)
    # Steps with the same images share one concatenation.
    assert attended[1] is attended[2]


def test_no_image_in_its_window_adds_no_term(monkeypatch) -> None:
    transformer, patches, _ = _denoise(
        monkeypatch,
        [
            # floor(1.0 * 4) = 4: past the last of the 4 steps.
            _field(tokens_name="first", begin_step_percent=1.0),
            _field(tokens_name="second", begin_step_percent=1.0),
        ],
    )

    assert all(blocks is None for blocks in transformer.calls)
    assert len(patches) == 1, "the LoRA is applied for the whole run, as for one image"


def test_a_strength_zero_image_leaves_the_other_image_of_its_adapter(monkeypatch) -> None:
    transformer, patches, adapters = _denoise(
        monkeypatch, [_field(tokens_name="first", strength=0.0), _field(tokens_name="second")]
    )

    assert patches == [(adapters["ipa"].lora_patch(), 1.0)]
    assert all(torch.equal(_attended(blocks[0]), torch.full((1, 3, 4), 2.0)) for blocks in transformer.calls[0::2])


def test_more_images_than_an_adapter_takes_are_refused(monkeypatch) -> None:
    fields = [_field(tokens_name="first", image_name=f"{n}.png") for n in range(5)]

    with pytest.raises(ValueError, match="'ipa' has 5 reference images; at most 4 can share one adapter"):
        _denoise(monkeypatch, fields)
    # The cap counts the images that do something: a strength-0 one is dropped first.
    _denoise(monkeypatch, [*fields[:4], _field(image_name="off.png", strength=0.0)])


def test_the_images_of_an_adapter_are_attended_in_the_same_order_whatever_the_collect_order(monkeypatch) -> None:
    fields = [
        _field(tokens_name="second", image_name="b.png"),
        _field(tokens_name="first", image_name="a.png", strength=0.5),
        _field(tokens_name="first", image_name="a.png"),
    ]
    orders = []
    for permutation in (fields, fields[::-1], fields[1:] + fields[:1]):
        (group,) = AnimaDenoiseInvocation._normalize_ip_adapters(permutation)
        orders.append([(f.image_name, f.strength) for f in group])

    assert orders == [[("a.png", 0.5), ("a.png", 1.0), ("b.png", 1.0)]] * 3
    transformer, _, _ = _denoise(monkeypatch, fields[::-1])
    expected = torch.cat([torch.full((1, 3, 4), 0.5), torch.full((1, 3, 4), 1.0), torch.full((1, 3, 4), 2.0)], 1)
    assert torch.equal(_attended(transformer.calls[0][0]), expected)


def test_anima_3_8b_runs_the_adapter_with_a_warning(monkeypatch) -> None:
    context = MagicMock()

    transformer, _, _ = _denoise(monkeypatch, [_field()], semantic_connector=True, context=context)

    assert all(blocks is not None for blocks in transformer.calls[0::2])
    warnings = [call.args[0] for call in context.logger.warning.call_args_list]
    assert any("for the Qwen3 Anima models (2B and 2.9B), not Anima-3.8B" in w for w in warnings)


def test_a_qwen3_anima_runs_the_adapter_without_that_warning(monkeypatch) -> None:
    context = MagicMock()

    _denoise(monkeypatch, [_field()], context=context)

    assert not any("Qwen3 Anima" in call.args[0] for call in context.logger.warning.call_args_list)


# --- encoder node -------------------------------------------------------------------------------------


def _encoder_node(**steps) -> AnimaIPAdapterInvocation:
    return AnimaIPAdapterInvocation(
        id="ipa", image={"image_name": "ref.png"}, ip_adapter_model=_identifier("ipa"), **steps
    )


def test_the_encoder_node_refuses_a_step_window_that_begins_after_it_ends() -> None:
    assert _encoder_node(begin_step_percent=0.4, end_step_percent=0.4).end_step_percent == 0.4
    with pytest.raises(ValueError, match="Begin step percent must be less than or equal to end step percent"):
        _encoder_node(begin_step_percent=0.6, end_step_percent=0.4)


def test_the_encoder_node_saves_the_tokens_without_strength(monkeypatch) -> None:
    module = "invokeai.app.invocations.anima.anima_ip_adapter"
    tokens = torch.randn(1, 1024, 768)
    monkeypatch.setattr(f"{module}.AnimaIPAdapterInvocation._encode", lambda self, context, image: tokens)
    context = MagicMock()
    context.tensors.save.return_value = "saved-tokens"
    invocation = AnimaIPAdapterInvocation.model_construct(
        image=SimpleNamespace(image_name="ref"),
        ip_adapter_model=_identifier("ipa"),
        strength=1.5,
        begin_step_percent=0.1,
        end_step_percent=0.9,
    )

    output = invocation.invoke(context)

    (saved,), _ = context.tensors.save.call_args
    assert torch.equal(saved, tokens) and saved.device.type == "cpu"
    field = output.ip_adapter
    assert (field.tokens_name, field.image_name) == ("saved-tokens", "ref")
    assert (field.strength, field.begin_step_percent, field.end_step_percent) == (1.5, 0.1, 0.9)
    assert field.ip_adapter_model.key == "ipa"


def test_a_strength_zero_node_encodes_nothing_and_denoise_ignores_it(monkeypatch) -> None:
    context = MagicMock()
    context.tensors.save.return_value = "placeholder"
    invocation = AnimaIPAdapterInvocation.model_construct(
        image=SimpleNamespace(image_name="ref"),
        ip_adapter_model=_identifier("ipa"),
        strength=0.0,
        begin_step_percent=0.0,
        end_step_percent=1.0,
    )

    field = invocation.invoke(context).ip_adapter

    context.models.load.assert_not_called()
    context.models.search_by_attrs.assert_not_called()
    context.images.get_pil.assert_not_called()
    transformer, patches, adapters = _denoise(monkeypatch, [field])
    assert all(blocks is None for blocks in transformer.calls) and patches == [] and adapters == {}


def _siglip_record(name: str, source: str) -> SimpleNamespace:
    return SimpleNamespace(key=name, name=name, source=source)


def test_the_encoder_is_found_by_its_source_and_not_confused_with_the_redux_siglip() -> None:
    renamed = _siglip_record("my siglip2", siglip2_base_512.source)
    context = MagicMock()
    context.models.search_by_attrs.return_value = [_siglip_record(siglip.name, siglip.source), renamed]

    assert AnimaIPAdapterInvocation._get_siglip_model(context) is renamed
    context._services.model_manager.install.heuristic_import.assert_not_called()


def test_a_missing_encoder_is_installed_from_its_starter_source() -> None:
    installed = _siglip_record(siglip2_base_512.name, siglip2_base_512.source)
    context = MagicMock()
    context._services.configuration.multiuser = False
    context.models.search_by_attrs.side_effect = [[], [installed]]

    assert AnimaIPAdapterInvocation._get_siglip_model(context) is installed
    (source,), kwargs = context._services.model_manager.install.heuristic_import.call_args
    assert source == siglip2_base_512.source
    assert kwargs["config"].type == ModelType.SigLIP


@pytest.mark.parametrize(
    "source",
    [
        "https://huggingface.co/google/siglip2-base-patch16-512",
        "https://huggingface.co/google/siglip2-base-patch16-512/",
        "google/siglip2-base-patch16-512:fp16",
        "Google/SigLIP2-base-patch16-512",
    ],
)
def test_the_encoder_is_found_under_another_spelling_of_its_repo(source) -> None:
    record = _siglip_record("my siglip2", source)
    context = MagicMock()
    context.models.search_by_attrs.return_value = [record]

    assert AnimaIPAdapterInvocation._get_siglip_model(context) is record
    context._services.model_manager.install.heuristic_import.assert_not_called()


@pytest.mark.parametrize(
    "source", ["google/siglip2-base-patch16-512::vision_model", "google/siglip2-base-patch16-224", "/models/siglip2"]
)
def test_another_siglip_source_is_not_taken_for_the_encoder(source) -> None:
    context = MagicMock()
    context._services.configuration.multiuser = False
    context.models.search_by_attrs.side_effect = [[_siglip_record("other", source)], []]

    with pytest.raises(RuntimeError, match="Could not install"):
        AnimaIPAdapterInvocation._get_siglip_model(context)


@pytest.mark.parametrize(("is_admin", "installs"), [(False, False), (True, True)])
def test_in_multiuser_mode_only_an_administrator_installs_a_missing_encoder(is_admin, installs) -> None:
    installed = _siglip_record(siglip2_base_512.name, siglip2_base_512.source)
    context = MagicMock()
    context._services.configuration.multiuser = True
    context._services.users.get.return_value = SimpleNamespace(is_admin=is_admin)
    context.models.search_by_attrs.side_effect = [[], [installed]]

    if installs:
        assert AnimaIPAdapterInvocation._get_siglip_model(context) is installed
    else:
        with pytest.raises(RuntimeError, match="Ask an administrator to install it from the Model Manager"):
            AnimaIPAdapterInvocation._get_siglip_model(context)
    assert context._services.model_manager.install.heuristic_import.called is installs
    context._services.users.get.assert_called_once_with(context._data.queue_item.user_id)
