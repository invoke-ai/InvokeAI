from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

from invokeai.app.invocations.text_encoder.sd3_text_encoder import Sd3TextEncoderInvocation
from invokeai.backend.model_manager.taxonomy import ModelFormat


class FakeSd3ClipTextEncoder(torch.nn.Module):
    def __init__(self, effective_device: torch.device):
        super().__init__()
        self.register_parameter("cpu_param", torch.nn.Parameter(torch.ones(1)))
        self.register_buffer("active_buffer", torch.ones(1, device=effective_device))
        self.dtype = torch.float32
        self.forward_input_device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        assert output_hidden_states
        self.forward_input_device = input_ids.device
        hidden = input_ids.unsqueeze(-1).float()
        return SimpleNamespace(hidden_states=[hidden, hidden + 1], __getitem__=lambda self, idx: hidden)


class FakeClipOutput(SimpleNamespace):
    @property
    def text_embeds(self):
        return self.hidden_states[-1]

    @property
    def pooler_output(self):
        return self.hidden_states[-1]


class FakeClipTokenizer:
    def __call__(self, prompt, padding, max_length=None, truncation=None, return_tensors=None):
        del prompt, padding, max_length, truncation, return_tensors
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def batch_decode(self, input_ids):
        del input_ids
        return ["decoded"]


class FakeSd3T5Encoder(torch.nn.Module):
    def __init__(self, effective_device: torch.device):
        super().__init__()
        self.register_parameter("cpu_param", torch.nn.Parameter(torch.ones(1)))
        self.register_buffer("active_buffer", torch.ones(1, device=effective_device))
        self.forward_input_device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(self, input_ids: torch.Tensor):
        self.forward_input_device = input_ids.device
        hidden = input_ids.unsqueeze(-1).float()
        return (hidden,)


class FakeT5Tokenizer:
    def __call__(self, prompt, padding, max_length=None, truncation=None, add_special_tokens=None, return_tensors=None):
        del prompt, padding, max_length, truncation, add_special_tokens, return_tensors
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def batch_decode(self, input_ids):
        del input_ids
        return ["decoded"]


class FakeLoadedModel:
    def __init__(self, model, config=None, compute_device=torch.device("cpu")):
        self._model = model
        self.config = config
        self._compute_device = compute_device

    @property
    def compute_device(self) -> torch.device:
        return self._compute_device

    @property
    def model(self):
        return self._model

    @contextmanager
    def model_on_device(self, working_mem_bytes=None):
        yield (None, self._model)

    def __enter__(self):
        return self._model

    def __exit__(self, exc_type, exc, tb):
        return False


def test_sd3_clip_encode_uses_compute_device(monkeypatch):
    # Regression test for #9373: the encoder's weights are offloaded to CPU, but its intended compute device is the
    # accelerator. The encode must run on the intended compute device, not the current residency.
    module_path = "invokeai.app.invocations.text_encoder.sd3_text_encoder"
    compute_device = torch.device("meta")
    text_encoder = FakeSd3ClipTextEncoder(effective_device=torch.device("cpu"))
    tokenizer = FakeClipTokenizer()

    def forward(input_ids: torch.Tensor, output_hidden_states: bool = False):
        assert output_hidden_states
        text_encoder.forward_input_device = input_ids.device
        hidden = input_ids.unsqueeze(-1).float()
        return FakeClipOutput(hidden_states=[hidden, hidden + 1])

    text_encoder.forward = forward  # type: ignore[method-assign]

    mock_context = MagicMock()
    mock_context.models.load.side_effect = [
        FakeLoadedModel(
            text_encoder, config=SimpleNamespace(format=ModelFormat.Diffusers), compute_device=compute_device
        ),
        FakeLoadedModel(tokenizer),
    ]
    mock_context.util.signal_progress = MagicMock()

    monkeypatch.setattr(f"{module_path}.CLIPTextModel", FakeSd3ClipTextEncoder)
    monkeypatch.setattr(f"{module_path}.CLIPTextModelWithProjection", FakeSd3ClipTextEncoder)
    monkeypatch.setattr(f"{module_path}.CLIPTokenizer", FakeClipTokenizer)
    monkeypatch.setattr(f"{module_path}.LayerPatcher.apply_smart_model_patches", lambda **kwargs: nullcontext())

    invocation = Sd3TextEncoderInvocation.model_construct(
        clip_l=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        clip_g=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        t5_encoder=None,
        prompt="test prompt",
    )

    invocation._clip_encode(
        context=mock_context,
        clip_model=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        lora_prefix="lora_clip_l-",
    )

    assert text_encoder.forward_input_device == compute_device


def test_sd3_t5_encode_uses_compute_device(monkeypatch):
    # Regression test for #9373: same as the CLIP case, for the T5 encode path.
    module_path = "invokeai.app.invocations.text_encoder.sd3_text_encoder"
    compute_device = torch.device("meta")
    text_encoder = FakeSd3T5Encoder(effective_device=torch.device("cpu"))
    tokenizer = FakeT5Tokenizer()

    mock_context = MagicMock()
    mock_context.models.load.side_effect = [
        FakeLoadedModel(text_encoder, compute_device=compute_device),
        FakeLoadedModel(tokenizer),
    ]
    mock_context.util.signal_progress = MagicMock()
    mock_context.logger.warning = MagicMock()

    monkeypatch.setattr(f"{module_path}.T5EncoderModel", FakeSd3T5Encoder)
    monkeypatch.setattr(f"{module_path}.T5Tokenizer", FakeT5Tokenizer)

    invocation = Sd3TextEncoderInvocation.model_construct(
        clip_l=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        clip_g=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[]),
        t5_encoder=SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace()),
        prompt="test prompt",
    )

    invocation._t5_encode(mock_context, max_seq_len=16)

    assert text_encoder.forward_input_device == compute_device


@pytest.mark.parametrize(
    ("model_class", "pooled_field"),
    [(CLIPTextModel, "pooler_output"), (CLIPTextModelWithProjection, "text_embeds")],
    ids=["clip_l_without_projection", "clip_g_with_projection"],
)
def test_sd3_clip_pooled_output_is_projected_where_the_encoder_has_a_projection(
    monkeypatch, model_class: type, pooled_field: str
):
    # SD 3 conditions on the projected pooled output. A separately installed CLIP-L is a plain CLIPTextModel,
    # whose first output is the hidden-state sequence, not a pooled vector; its pooled output is what SD 3's
    # identity CLIP-L projection would give. A CLIP-G must be projected.
    module_path = "invokeai.app.invocations.text_encoder.sd3_text_encoder"
    torch.manual_seed(0)
    config = CLIPTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=8,
        projection_dim=16,
    )
    text_encoder = model_class(config).eval()

    mock_context = MagicMock()
    mock_context.models.load.side_effect = [
        FakeLoadedModel(text_encoder, config=SimpleNamespace(format=ModelFormat.Checkpoint, name="clip")),
        FakeLoadedModel(FakeClipTokenizer()),
    ]
    monkeypatch.setattr(f"{module_path}.CLIPTokenizer", FakeClipTokenizer)
    monkeypatch.setattr(f"{module_path}.LayerPatcher.apply_smart_model_patches", lambda **kwargs: nullcontext())
    clip_model = SimpleNamespace(text_encoder=SimpleNamespace(), tokenizer=SimpleNamespace(), loras=[])
    invocation = Sd3TextEncoderInvocation.model_construct(
        clip_l=clip_model, clip_g=clip_model, t5_encoder=None, prompt="test prompt"
    )

    _, pooled = invocation._clip_encode(
        context=mock_context, clip_model=clip_model, lora_prefix="lora_clip_l-", tokenizer_max_length=3
    )

    with torch.no_grad():
        expected = getattr(text_encoder(input_ids=torch.tensor([[1, 2, 3]])), pooled_field)
    assert pooled.shape == (1, 16)
    torch.testing.assert_close(pooled, expected)


def test_each_clip_encoder_takes_only_its_own_layers_of_an_sd3_lora(monkeypatch):
    """CLIP-L and CLIP-G name their modules alike (`text_model.encoder.layers.N...`), so an SD 3 LoRA that trains both
    keeps them under separate prefixes, and each encode has to pick its own."""
    from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
        apply_custom_layers_to_model,
    )
    from invokeai.backend.patches.layers.lora_layer import LoRALayer
    from invokeai.backend.patches.lora_conversions.sd3_lora_constants import (
        SD3_LORA_CLIP_G_PREFIX,
        SD3_LORA_CLIP_L_PREFIX,
    )
    from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

    module_path = "invokeai.app.invocations.text_encoder.sd3_text_encoder"
    monkeypatch.setattr(f"{module_path}.CLIPTokenizer", FakeClipTokenizer)
    torch.manual_seed(0)
    config = CLIPTextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        vocab_size=10,
        max_position_embeddings=8,
        projection_dim=16,
    )
    encoder = CLIPTextModelWithProjection(config).eval()
    apply_custom_layers_to_model(encoder)
    q_proj = encoder.get_submodule("text_model.encoder.layers.0.self_attn.q_proj")
    weight, bias = q_proj.weight.detach().clone(), q_proj.bias.detach().clone()
    generator = torch.Generator().manual_seed(1)
    deltas = {}
    layers = {}
    for prefix in (SD3_LORA_CLIP_L_PREFIX, SD3_LORA_CLIP_G_PREFIX):
        up, down = torch.randn(16, 2, generator=generator), torch.randn(2, 16, generator=generator)
        deltas[prefix] = up @ down
        layers[f"{prefix}text_model.encoder.layers.0.self_attn.q_proj"] = LoRALayer(
            up=up, mid=None, down=down, alpha=None, bias=None
        )
    patch = ModelPatchRaw(layers=layers)
    probe = torch.randn(2, 16, generator=generator)
    seen: list[torch.Tensor] = []
    encoder.register_forward_pre_hook(lambda module, args, kwargs: seen.append(q_proj(probe)), with_kwargs=True)

    lora_info = SimpleNamespace(model=patch, model_in_ram=nullcontext)
    loaded = {
        "encoder": FakeLoadedModel(encoder, config=SimpleNamespace(format=ModelFormat.Diffusers)),
        "tokenizer": FakeLoadedModel(FakeClipTokenizer()),
    }
    mock_context = MagicMock()
    mock_context.models.load.side_effect = lambda identifier: loaded.get(identifier, lora_info)
    clip_model = SimpleNamespace(
        text_encoder="encoder", tokenizer="tokenizer", loras=[SimpleNamespace(lora="lora", weight=1.0)]
    )
    invocation = Sd3TextEncoderInvocation.model_construct(prompt="test prompt")

    for prefix in (SD3_LORA_CLIP_L_PREFIX, SD3_LORA_CLIP_G_PREFIX):
        seen.clear()
        with torch.no_grad():
            invocation._clip_encode(context=mock_context, clip_model=clip_model, lora_prefix=prefix)
            torch.testing.assert_close(seen[0], probe @ (weight + deltas[prefix]).T + bias)


def test_invoke_hands_clip_l_and_clip_g_their_own_layers(monkeypatch):
    """The two encoders differ in width (768 / 1280 in the release; 16 / 24 here), so a LoRA handed the other
    encoder's layers would only be skipped, layer by layer, as a shape mismatch -- and do nothing."""
    from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
        apply_custom_layers_to_model,
    )
    from invokeai.backend.patches.layers.lora_layer import LoRALayer
    from invokeai.backend.patches.lora_conversions.sd3_lora_constants import (
        SD3_LORA_CLIP_G_PREFIX,
        SD3_LORA_CLIP_L_PREFIX,
    )
    from invokeai.backend.patches.model_patch_raw import ModelPatchRaw

    monkeypatch.setattr("invokeai.app.invocations.text_encoder.sd3_text_encoder.CLIPTokenizer", FakeClipTokenizer)
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(1)
    encoders, expected, seen, layers = {}, {}, {}, {}
    for name, prefix, width in (("clip_l", SD3_LORA_CLIP_L_PREFIX, 16), ("clip_g", SD3_LORA_CLIP_G_PREFIX, 24)):
        config = CLIPTextConfig(
            hidden_size=width,
            intermediate_size=2 * width,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=10,
            max_position_embeddings=8,
            projection_dim=width,
        )
        encoder = CLIPTextModelWithProjection(config).eval()
        apply_custom_layers_to_model(encoder)
        q_proj = encoder.get_submodule("text_model.encoder.layers.0.self_attn.q_proj")
        up, down = torch.randn(width, 2, generator=generator), torch.randn(2, width, generator=generator)
        layers[f"{prefix}text_model.encoder.layers.0.self_attn.q_proj"] = LoRALayer(
            up=up, mid=None, down=down, alpha=None, bias=None
        )
        probe = torch.randn(2, width, generator=generator)
        expected[name] = probe @ (q_proj.weight.detach() + up @ down).T + q_proj.bias.detach()
        seen[name] = []
        encoder.register_forward_pre_hook(
            lambda module, args, kwargs, q=q_proj, x=probe, out=seen[name]: out.append(q(x)), with_kwargs=True
        )
        encoders[name] = encoder

    lora_info = SimpleNamespace(model=ModelPatchRaw(layers=layers), model_in_ram=nullcontext)
    loaded = {
        "enc_clip_l": FakeLoadedModel(encoders["clip_l"], config=SimpleNamespace(format=ModelFormat.Diffusers)),
        "enc_clip_g": FakeLoadedModel(encoders["clip_g"], config=SimpleNamespace(format=ModelFormat.Diffusers)),
        "tokenizer": FakeLoadedModel(FakeClipTokenizer()),
    }
    mock_context = MagicMock()
    mock_context.models.load.side_effect = lambda identifier: loaded.get(identifier, lora_info)
    mock_context.conditioning.save.return_value = "cond-name"
    loras = [SimpleNamespace(lora="lora", weight=1.0)]
    invocation = Sd3TextEncoderInvocation.model_construct(
        prompt="test prompt",
        clip_l=SimpleNamespace(text_encoder="enc_clip_l", tokenizer="tokenizer", loras=loras),
        clip_g=SimpleNamespace(text_encoder="enc_clip_g", tokenizer="tokenizer", loras=loras),
        t5_encoder=None,
    )

    with torch.no_grad():
        invocation.invoke(mock_context)

    for name in ("clip_l", "clip_g"):
        torch.testing.assert_close(seen[name][0], expected[name])
