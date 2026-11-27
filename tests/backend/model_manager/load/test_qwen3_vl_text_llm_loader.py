"""A single-file Qwen3-VL encoder serving as a Text LLM loads as the whole Instruct chat model.

Driven through the view loader's own entry point with the Text LLM view of an encoder record, over a real safetensors
file in ComfyUI's spelling, at toy width and in float32 on the CPU. The language model is the encoder loader's own
decode; what these pin is the head on top of it: tied to the embedding where the variant's config ties it (the 4B),
read from the file and widened otherwise (the 8B), and refused from the header, before anything is reserved, when
the loader cannot widen it.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.model_manager.configs.roles import view_as
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Qwen3VLEncoder_Config
from invokeai.backend.model_manager.load import load_default
from invokeai.backend.model_manager.load.model_loaders import krea2
from invokeai.backend.model_manager.load.model_loaders.krea2 import Qwen3VLEncoderCheckpointLoader
from invokeai.backend.model_manager.load.model_loaders.text_llm import TextLLMQwen3VLLoader
from invokeai.backend.model_manager.taxonomy import ModelType, Qwen3VLVariantType, SubModelType
from invokeai.backend.quantization.fp8_scaled import FP8_WEIGHT_DTYPES
from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_tokenizer
from invokeai.backend.util.devices import TorchDevice
from tests.fixtures.loader_seams import Seam, SeamRun, prepare
from tests.fixtures.model_roles import qwen3_vl_encoder_file
from tests.fixtures.quantized_payloads import comfy_quant_marker, mxfp8_tensors, quantize_scaled_fp8

# Whole MXFP8 tiles (128 rows, 4 blocks of 32) for the head, so one geometry serves every case.
HIDDEN = 128
VOCAB = 128
FLOAT32 = torch.float32.itemsize


def _seam(loader: type, entry: str, compute_dtype: torch.dtype = torch.float32) -> Seam:
    return Seam(
        loader=loader,
        module=krea2,
        entry=entry,
        compute_dtype=compute_dtype,
        patches_device=True,
        casts_fp8_storage=False,
    )


def _tiny_config(*, tied: bool) -> Qwen3VLConfig:
    return Qwen3VLConfig(
        text_config={
            "hidden_size": HIDDEN,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": VOCAB,
            "rope_scaling": {"rope_type": "default", "mrope_section": [8, 4, 4], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": HIDDEN,
            "deepstack_visual_indexes": [],
        },
        tie_word_embeddings=tied,
    )


def _reference(*, tied: bool) -> Qwen3VLForConditionalGeneration:
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(_tiny_config(tied=tied)).eval()


def _comfy_file(reference: Qwen3VLForConditionalGeneration) -> dict[str, torch.Tensor]:
    """The reference's tower and language model as ComfyUI spells them (`model.layers.*`, `model.visual.*`), no head."""
    return {
        key.replace("model.language_model.", "model.", 1): value.detach().clone()
        for key, value in reference.state_dict().items()
        if not key.startswith("lm_head.")
    }


def _logits(model: Qwen3VLForConditionalGeneration) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=torch.tensor([[1, 5, 9, 33, 2, 47, 17]]), use_cache=False).logits


def _view(
    tmp_path: Path,
    state_dict: dict[str, torch.Tensor],
    variant: Qwen3VLVariantType,
    *,
    cpu_only: bool | None = None,
    metadata: dict[str, str] | None = None,
) -> TextLLM_Qwen3VLEncoder_Config:
    path = tmp_path / "qwen3vl.safetensors"
    save_file(state_dict, str(path), metadata=metadata)
    record = qwen3_vl_encoder_file("vl", variant).model_copy(update={"path": str(path), "cpu_only": cpu_only})
    view = view_as(record, ModelType.TextLLM)
    assert isinstance(view, TextLLM_Qwen3VLEncoder_Config)
    return view


def _prepare(monkeypatch: pytest.MonkeyPatch, seam: Seam, *, tied: bool) -> SeamRun:
    def tiny_geometry(patch: pytest.MonkeyPatch) -> None:
        patch.setattr(Qwen3VLEncoderCheckpointLoader, "_load_te_config", lambda _self, _config: _tiny_config(tied=tied))

    return prepare(seam, monkeypatch, geometry=tiny_geometry)


def _run(monkeypatch: pytest.MonkeyPatch, *, tied: bool, compute_dtype: torch.dtype = torch.float32) -> SeamRun:
    run = _prepare(monkeypatch, _seam(TextLLMQwen3VLLoader, "_load_model", compute_dtype), tied=tied)
    # The view's language model loads through an encoder loader it builds from its own surroundings.
    run.loader._app_config = InvokeAIAppConfig()
    return run


def _encoder_reservation(monkeypatch: pytest.MonkeyPatch, view: TextLLM_Qwen3VLEncoder_Config, *, tied: bool) -> int:
    """What the encoder loader alone reserves for the same file."""
    encoder = _prepare(monkeypatch, _seam(Qwen3VLEncoderCheckpointLoader, "_load_text_encoder"), tied=tied)
    encoder.load(view_as(view, ModelType.Qwen3VLEncoder))
    assert len(encoder.reserved) == 1
    return encoder.reserved[0]


def test_a_4b_file_generates_through_its_token_embedding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reference = _reference(tied=True)
    view = _view(tmp_path, _comfy_file(reference), Qwen3VLVariantType.Qwen3VL_4B)
    run = _run(monkeypatch, tied=True)

    model = run.load(view, None)

    assert isinstance(model, Qwen3VLForConditionalGeneration)
    assert model.lm_head.weight is model.get_input_embeddings().weight
    assert isinstance(model.model.visual, torch.nn.Identity)
    expected = _logits(reference)
    assert expected.abs().max() > 0.1
    torch.testing.assert_close(_logits(model), expected)
    # Through `generate`, which takes the vendored generation config and Qwen3-VL's own input preparation.
    prompt = torch.tensor([[3, 14, 15, 9]])
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=4, do_sample=False)
        expected_tokens = reference.generate(prompt, max_new_tokens=4, do_sample=False, pad_token_id=0)
    assert generated.shape == (1, 8)
    assert torch.equal(generated, expected_tokens)
    # Nothing is widened beyond the language model, so the encoder's reservation is the whole of it.
    assert run.reserved == [_encoder_reservation(monkeypatch, view, tied=True)]


def _bf16_head(weight: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    stored = weight.to(torch.bfloat16)
    # The bf16 weight, held until it is widened to float32.
    return {"lm_head.weight": stored}, stored.float(), VOCAB * HIDDEN * 2


def _scaled_fp8_head(weight: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
    """As Comfy-Org's nvfp4 8B stores its head: e4m3 codes, a scalar scale and a marker naming the format."""
    payload = quantize_scaled_fp8(weight)
    head = {
        "lm_head.weight": payload.codes,
        "lm_head.weight_scale": payload.scale,
        "lm_head.comfy_quant": comfy_quant_marker({"format": "float8_e4m3fn"}),
    }
    # The codes, the float32 scalar, and the fold's float32 copy of the weight and its product, whole at this size.
    return head, payload.dequantized, VOCAB * HIDDEN + FLOAT32 + 2 * VOCAB * HIDDEN * FLOAT32


@pytest.mark.parametrize("stored_head", [_bf16_head, _scaled_fp8_head], ids=["bf16", "scaled-fp8"])
def test_an_8b_file_loads_its_own_lm_head_widened_to_the_compute_dtype(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stored_head
) -> None:
    reference = _reference(tied=False)
    head, decoded, held_while_widening = stored_head(reference.lm_head.weight.detach())
    with torch.no_grad():
        reference.lm_head.weight.copy_(decoded)
    view = _view(tmp_path, {**_comfy_file(reference), **head}, Qwen3VLVariantType.Qwen3VL_8B)
    run = _run(monkeypatch, tied=False)

    model = run.load(view, None)

    assert model.lm_head.weight is not model.get_input_embeddings().weight
    assert model.lm_head.weight.dtype == torch.float32
    torch.testing.assert_close(model.lm_head.weight, decoded, rtol=0, atol=0)
    torch.testing.assert_close(_logits(model), _logits(reference))
    # One reservation, made before the language model widens anything: the encoder's, plus the head widened after it.
    widened_head = VOCAB * HIDDEN * FLOAT32
    assert run.reserved == [_encoder_reservation(monkeypatch, view, tied=False) + widened_head + held_while_widening]


_MX_HEADER = {"_quantization_metadata": json.dumps({"layers": {"lm_head": {"format": "mxfp8"}}})}


def test_an_mxfp8_head_named_only_in_the_header_is_decoded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reference = _reference(tied=False)
    exponents = torch.randint(120, 131, (VOCAB, HIDDEN // 32))
    head, block_scale = mxfp8_tensors("lm_head", exponents)
    decoded = block_scale.repeat_interleave(32, dim=1)
    with torch.no_grad():
        reference.lm_head.weight.copy_(decoded)
    view = _view(tmp_path, {**_comfy_file(reference), **head}, Qwen3VLVariantType.Qwen3VL_8B, metadata=_MX_HEADER)

    model = _run(monkeypatch, tied=False).load(view, None)

    torch.testing.assert_close(model.lm_head.weight, decoded, rtol=0, atol=0)
    torch.testing.assert_close(_logits(model), _logits(reference))


@pytest.mark.parametrize(
    ("head", "metadata", "message"),
    [
        pytest.param(
            {"lm_head.weight": torch.zeros(VOCAB, HIDDEN // 2, dtype=torch.uint8)}, None, "torch.uint8", id="packed"
        ),
        pytest.param(
            {"lm_head.weight": torch.zeros(VOCAB + 1, HIDDEN, dtype=torch.bfloat16)},
            None,
            "token embedding",
            id="shape",
        ),
        pytest.param(
            {
                "lm_head.weight": torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16),
                "lm_head.bias": torch.zeros(VOCAB, dtype=torch.bfloat16),
            },
            None,
            "lm_head.bias",
            id="unread-key",
        ),
        pytest.param(
            mxfp8_tensors("lm_head", torch.full((VOCAB, HIDDEN // 32), 127))[0], None, "mxfp8", id="unnamed-mx-grid"
        ),
        pytest.param(
            {"lm_head.weight": torch.zeros(VOCAB, HIDDEN, dtype=torch.float8_e4m3fnuz)}, None, "fnuz", id="fnuz"
        ),
        pytest.param(
            {
                "lm_head.weight": torch.zeros(VOCAB, HIDDEN, dtype=torch.float8_e4m3fn),
                "lm_head.weight_scale": torch.ones(VOCAB + 3),
            },
            None,
            "neither per-tensor nor per-output-channel",
            id="scale-fits-no-layout",
        ),
        pytest.param(
            {
                **mxfp8_tensors("lm_head", torch.full((VOCAB, HIDDEN // 32), 127))[0],
                "lm_head.weight_scale": torch.zeros(VOCAB, HIDDEN // 16, dtype=torch.uint8),
            },
            _MX_HEADER,
            "does not describe",
            id="mx-grid-of-the-wrong-block-width",
        ),
    ],
)
def test_an_lm_head_the_loader_cannot_widen_is_refused_before_room_is_reserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, head: dict[str, torch.Tensor], metadata, message: str
) -> None:
    view = _view(
        tmp_path, {**_comfy_file(_reference(tied=False)), **head}, Qwen3VLVariantType.Qwen3VL_8B, metadata=metadata
    )
    run = _run(monkeypatch, tied=False)

    with pytest.raises(ValueError, match=message):
        run.load(view, None)
    assert run.reserved == []


def _fp8_gate_proj(reference: Qwen3VLForConditionalGeneration) -> dict[str, torch.Tensor]:
    payload = quantize_scaled_fp8(reference.model.language_model.layers[0].mlp.gate_proj.weight.detach())
    return {
        "model.layers.0.mlp.gate_proj.weight": payload.codes,
        "model.layers.0.mlp.gate_proj.weight_scale": payload.scale,
    }


def _fp8_policy(monkeypatch: pytest.MonkeyPatch, device: torch.device, *, supported: bool) -> None:
    """The process' device, and whether it keeps fp8 weights for its matmul and for storage."""
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: device))
    monkeypatch.setattr(TorchDevice, "choose_torch_dtype", staticmethod(lambda: torch.bfloat16))
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda d: supported and d.type != "cpu")
    monkeypatch.setattr(
        load_default, "_device_supports_fp8_storage", lambda d, _logger=None: supported and d.type != "cpu"
    )


@pytest.mark.parametrize(
    ("cpu_only", "supported", "expected"),
    [
        pytest.param(None, True, {torch.float8_e4m3fn, torch.bfloat16}, id="device keeping fp8"),
        pytest.param(None, False, {torch.bfloat16}, id="device folding fp8"),
        pytest.param(True, True, {torch.float32}, id="cpu only"),
    ],
)
def test_the_view_decodes_by_the_load_decision_of_the_device_it_runs_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_only: bool | None, supported: bool, expected: set
) -> None:
    """bfloat16 configured: a device that keeps fp8 weights keeps them in the view as in the encoder, one that does not
    folds them to bfloat16, and a CPU-only record widens everything to float32, as CPU-only Text LLMs do. Loaded, not
    run (no CPU bf16 forward)."""
    reference = _reference(tied=True)
    view = _view(
        tmp_path,
        {**_comfy_file(reference), **_fp8_gate_proj(reference)},
        Qwen3VLVariantType.Qwen3VL_4B,
        cpu_only=cpu_only,
    )
    run = _run(monkeypatch, tied=True, compute_dtype=torch.bfloat16)
    _fp8_policy(monkeypatch, torch.device("cuda"), supported=supported)

    model = run.load(view, None)

    assert {parameter.dtype for parameter in model.parameters()} == expected


def test_devices_with_different_fp8_policies_never_share_the_views_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Device caches adopt a canonical CPU copy by key. The view built where fp8 stays packed differs from the one built
    where it is folded, so their keys must differ -- and a device with the same policy keeps the same key."""
    view = _view(tmp_path, _comfy_file(_reference(tied=True)), Qwen3VLVariantType.Qwen3VL_4B)

    def key(device: torch.device, *, supported: bool, cpu_only: bool | None = None) -> str:
        run = _run(monkeypatch, tied=True, compute_dtype=torch.bfloat16)
        _fp8_policy(monkeypatch, device, supported=supported)
        return run.loader._get_cache_key(view.model_copy(update={"cpu_only": cpu_only}), None)

    keeps_fp8 = key(torch.device("cuda", 0), supported=True)
    folds_fp8 = key(torch.device("cuda", 1), supported=False)
    assert keeps_fp8 != folds_fp8
    assert key(torch.device("cuda", 2), supported=True) == keeps_fp8
    assert key(torch.device("cuda", 0), supported=True, cpu_only=True) not in {keeps_fp8, folds_fp8}


@pytest.mark.parametrize("tied", [True, False], ids=["4b", "8b"])
def test_an_unscaled_fp8_layer_never_turns_the_embedding_or_the_head_into_unscaled_fp8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tied: bool
) -> None:
    """A build with an unscaled fp8 Linear takes the FP8 Storage pass; on a 4B it must leave the table the head ties to
    alone, as the attach branch does, and the encoder with it."""
    reference = _reference(tied=tied)
    gate_proj = quantize_scaled_fp8(reference.model.language_model.layers[0].mlp.gate_proj.weight.detach()).codes
    head = {} if tied else {"lm_head.weight": reference.lm_head.weight.detach().to(torch.bfloat16)}
    variant = Qwen3VLVariantType.Qwen3VL_4B if tied else Qwen3VLVariantType.Qwen3VL_8B
    view = _view(
        tmp_path, {**_comfy_file(reference), "model.layers.0.mlp.gate_proj.weight": gate_proj, **head}, variant
    )
    run = _run(monkeypatch, tied=tied, compute_dtype=torch.bfloat16)
    _fp8_policy(monkeypatch, torch.device("cuda"), supported=True)
    monkeypatch.setattr(load_default, "should_keep_fp8_weights", lambda _device: False)

    model = run.load(view, None)

    assert model.model.language_model.layers[0].mlp.gate_proj.weight.dtype in FP8_WEIGHT_DTYPES
    assert model.get_input_embeddings().weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert (model.lm_head.weight is model.get_input_embeddings().weight) is tied
    encoder = _prepare(
        monkeypatch, _seam(Qwen3VLEncoderCheckpointLoader, "_load_text_encoder", torch.bfloat16), tied=tied
    )
    encoder.loader._torch_device = torch.device("cuda")
    assert encoder.load(view_as(view, ModelType.Qwen3VLEncoder)).get_input_embeddings().weight.dtype == torch.bfloat16


def test_whether_the_head_ties_is_read_from_the_encoders_variant_not_the_views_role(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The role is fixed at install; the record's variant can be corrected later. An 8B file without a head that was
    installed as a 4B got the role because the 4B ties, and once fixed to 8B must be refused, not served with its
    embedding as the head."""
    path = tmp_path / "qwen3vl.safetensors"
    save_file(_comfy_file(_reference(tied=False)), str(path))
    record = qwen3_vl_encoder_file("vl", Qwen3VLVariantType.Qwen3VL_4B).model_copy(update={"path": str(path)})
    view = view_as(record.model_copy(update={"variant": Qwen3VLVariantType.Qwen3VL_8B}), ModelType.TextLLM)
    run = _run(monkeypatch, tied=False)

    with pytest.raises(ValueError, match="lm_head.weight"):
        run.load(view, None)
    assert run.reserved == []


def test_the_text_side_is_its_own_tokenizer_stopping_at_the_end_of_the_turn_and_no_processor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference = _reference(tied=True)
    view = _view(tmp_path, _comfy_file(reference), Qwen3VLVariantType.Qwen3VL_4B)
    run = _run(monkeypatch, tied=True)

    model = run.load(view, None)
    tokenizer = run.load(view, SubModelType.Tokenizer)

    # The process-wide instance is the encoders'; their truncating encodes would race Expand Prompt's on it.
    assert tokenizer is not load_bundled_qwen3_vl_tokenizer()
    end_of_turn, end_of_text = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    assert model.generation_config.eos_token_id == [end_of_turn, end_of_text]
    assert model.generation_config.pad_token_id == end_of_text
    with pytest.raises(ValueError, match="cannot read images"):
        run.load(view, SubModelType.Processor)
