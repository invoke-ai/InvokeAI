"""A single-file Qwen3-VL encoder serving as a Text LLM loads as the whole Instruct chat model.

Driven through the view loader's own entry point with the Text LLM view of an encoder record, over a real safetensors
file in ComfyUI's spelling, at toy width and in float32 on the CPU. The language model is the encoder loader's own
decode; what these pin is the head on top of it: tied to the embedding where the variant's config ties it (the 4B),
read from the file and widened otherwise (the 8B), and refused from the header, before anything is reserved, when
the loader cannot widen it. A view whose role reads images also runs the file's vision tower and serves a processor.
"""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.backend.model_manager.configs.roles import view_as
from invokeai.backend.model_manager.configs.text_llm import TextLLM_Qwen3VLEncoder_Config
from invokeai.backend.model_manager.load.model_loaders import krea2
from invokeai.backend.model_manager.load.model_loaders.krea2 import Qwen3VLEncoderCheckpointLoader
from invokeai.backend.model_manager.load.model_loaders.text_llm import TextLLMQwen3VLLoader
from invokeai.backend.model_manager.taxonomy import ModelType, Qwen3VLVariantType, SubModelType
from invokeai.backend.qwen3_vl.qwen3_vl_assets import load_bundled_qwen3_vl_tokenizer
from invokeai.backend.text_llm_pipeline import TextLLMPipeline
from invokeai.backend.util.devices import TorchDevice
from tests.fixtures.loader_seams import Seam, SeamRun, prepare
from tests.fixtures.model_roles import qwen3_vl_encoder_file
from tests.fixtures.quantized_payloads import comfy_quant_marker, mxfp8_tensors, quantize_scaled_fp8

# Whole MXFP8 tiles (128 rows, 4 blocks of 32) for the head, so one geometry serves every case.
HIDDEN = 128
VOCAB = 128
FLOAT32 = torch.float32.itemsize
# Inside the toy vocabulary, where the release's ids (151652 and up) are not.
IMAGE_TOKEN, VISION_START, VISION_END = 120, 122, 123


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
        image_token_id=IMAGE_TOKEN,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
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


def _logits_with_an_image(model: Qwen3VLForConditionalGeneration) -> torch.Tensor:
    """A 64 x 64 image as the processor would send it: 4 x 4 patches of 16 px, two frames deep, merged 2 x 2 into
    four image tokens."""
    generator = torch.Generator().manual_seed(1)
    pixel_values = torch.randn(16, 3 * 2 * 16 * 16, generator=generator)
    input_ids = torch.tensor([[1, VISION_START, *[IMAGE_TOKEN] * 4, VISION_END, 5, 9]])
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            # 1 marks image tokens, as the processor's own output does.
            mm_token_type_ids=(input_ids == IMAGE_TOKEN).long(),
            use_cache=False,
        ).logits


def _view(
    tmp_path: Path,
    state_dict: dict[str, torch.Tensor],
    variant: Qwen3VLVariantType,
    *,
    cpu_only: bool | None = None,
    metadata: dict[str, str] | None = None,
    reads_images: bool | None = None,
) -> TextLLM_Qwen3VLEncoder_Config:
    path = tmp_path / "qwen3vl.safetensors"
    save_file(state_dict, str(path), metadata=metadata)
    record = qwen3_vl_encoder_file("vl", variant, reads_images).model_copy(
        update={"path": str(path), "cpu_only": cpu_only}
    )
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
    ],
)
def test_an_lm_head_the_loader_cannot_widen_is_refused_before_room_is_reserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, head: dict[str, torch.Tensor], metadata, message: str
) -> None:
    view = _view(tmp_path, {**_comfy_file(_reference(tied=False)), **head}, Qwen3VLVariantType.Qwen3VL_8B)
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


@pytest.mark.parametrize("cpu_only", [None, True], ids=["on the device", "cpu only"])
def test_a_cpu_only_view_computes_in_float32_with_its_fp8_weights_folded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_only: bool | None
) -> None:
    """bfloat16 configured on a device that keeps fp8 weights for its matmul: the view on that device keeps them, a
    CPU-only one widens everything to float32, as CPU-only Text LLMs do. Loaded, not run (no CPU bf16 forward)."""
    reference = _reference(tied=True)
    view = _view(
        tmp_path,
        {**_comfy_file(reference), **_fp8_gate_proj(reference)},
        Qwen3VLVariantType.Qwen3VL_4B,
        cpu_only=cpu_only,
    )
    run = _run(monkeypatch, tied=True, compute_dtype=torch.bfloat16)
    accelerator = torch.device("cuda")
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: accelerator))
    run.loader._torch_device = accelerator
    monkeypatch.setattr(krea2, "should_keep_fp8_weights", lambda device: device.type != "cpu")
    monkeypatch.setattr(krea2, "_device_supports_fp8_storage", lambda device, _logger=None: device.type != "cpu")

    model = run.load(view, None)

    dtypes = {parameter.dtype for parameter in model.parameters()}
    if cpu_only:
        assert dtypes == {torch.float32}
    else:
        assert torch.float8_e4m3fn in dtypes and torch.float32 not in dtypes


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


_TOWER_QKV = "model.visual.blocks.0.attn.qkv"


def _tower_as_stored(reference: Qwen3VLForConditionalGeneration, stored: str) -> tuple[dict[str, torch.Tensor], int]:
    """The reference's file with its tower stored as `stored`, the reference's tower set to what that decodes to, and
    the bytes loading the tower holds beyond its float32 result."""
    file = _comfy_file(reference)
    tower = {key: tensor for key, tensor in file.items() if key.startswith("model.visual.")}
    if stored == "float32":
        return file, 0
    if stored == "bf16":
        with torch.no_grad():
            for parameter in reference.model.visual.parameters():
                parameter.copy_(parameter.to(torch.bfloat16).float())
        file.update({key: tensor.to(torch.bfloat16) for key, tensor in tower.items()})
        # Each bf16 tensor, held until it is cast.
        return file, sum(tensor.numel() for tensor in tower.values()) * 2
    payload = quantize_scaled_fp8(file[f"{_TOWER_QKV}.weight"])
    with torch.no_grad():
        reference.model.visual.blocks[0].attn.qkv.weight.copy_(payload.dequantized)
    file[f"{_TOWER_QKV}.weight"] = payload.codes
    file[f"{_TOWER_QKV}.weight_scale"] = payload.scale
    numel = payload.codes.numel()
    # The e4m3 codes until they are folded, the fold's float32 copy of the weight and its product, and the scale.
    return file, numel + 2 * numel * FLOAT32 + FLOAT32


@pytest.mark.parametrize("stored", ["float32", "bf16", "scaled-fp8"])
def test_a_view_that_reads_images_runs_the_files_vision_tower(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stored: str
) -> None:
    reference = _reference(tied=True)
    file, held_while_casting = _tower_as_stored(reference, stored)
    view = _view(tmp_path, file, Qwen3VLVariantType.Qwen3VL_4B, reads_images=True)
    run = _run(monkeypatch, tied=True)

    model = run.load(view, None)

    assert isinstance(model.model.visual, Qwen3VLVisionModel)
    assert {parameter.dtype for parameter in model.model.visual.parameters()} == {torch.float32}
    expected = _logits_with_an_image(reference)
    assert expected.abs().max() > 0.1
    torch.testing.assert_close(_logits_with_an_image(model), expected)
    # Still the one reservation, made before the decode and now covering the tower, which loads after it.
    tower_bytes = sum(tensor.numel() for tensor in reference.model.visual.state_dict().values()) * FLOAT32
    assert run.reserved == [_encoder_reservation(monkeypatch, view, tied=True) + tower_bytes + held_while_casting]


def test_a_view_whose_tower_no_longer_loads_says_so_rather_than_running_text_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Its role says it reads images, so it serves a processor: a model without its tower would fail at the first
    image, or answer about pixels it never saw."""
    file = _comfy_file(_reference(tied=True))
    del file["model.visual.merger.linear_fc2.weight"]
    view = _view(tmp_path, file, Qwen3VLVariantType.Qwen3VL_4B, reads_images=True)

    with pytest.raises(ValueError, match="cannot load its vision tower.*Re-identify it"):
        _run(monkeypatch, tied=True).load(view, None)


def test_a_view_that_reads_images_serves_a_processor_over_its_own_tokenizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    view = _view(tmp_path, _comfy_file(_reference(tied=True)), Qwen3VLVariantType.Qwen3VL_4B, reads_images=True)
    run = _run(monkeypatch, tied=True)

    processor = run.load(view, SubModelType.Processor)

    assert isinstance(processor, Qwen3VLProcessor)
    assert processor.tokenizer is not load_bundled_qwen3_vl_tokenizer()
    assert processor.chat_template is not None
    assert processor.chat_template == processor.tokenizer.chat_template


def _tiny_qwen3_vl_with_the_release_vocabulary() -> Qwen3VLForConditionalGeneration:
    """The release's vocabulary and image token ids, and 16 px patches merged 2 x 2, around layers a few dozen wide."""
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 151936,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "deepstack_visual_indexes": [],
        },
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval()


def test_an_image_reaches_the_vision_tower_through_the_processor_the_view_serves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The chat template's image placeholder has to expand into as many image tokens as the tower returns features, or
    generate() fails: the step the Text LLM pipeline takes for Expand Prompt and Image to Prompt."""
    view = _view(tmp_path, _comfy_file(_reference(tied=True)), Qwen3VLVariantType.Qwen3VL_4B, reads_images=True)
    processor = _run(monkeypatch, tied=True).load(view, SubModelType.Processor)
    model = _tiny_qwen3_vl_with_the_release_vocabulary()
    grids: list[list[list[int]]] = []
    model.model.visual.register_forward_hook(
        lambda _module, _args, kwargs, _output: grids.append(kwargs["grid_thw"].tolist()), with_kwargs=True
    )

    result = TextLLMPipeline(model, processor.tokenizer, processor).run(
        prompt="Describe this image.",
        system_prompt="Write the prompt.",
        max_new_tokens=2,
        image=Image.new("RGB", (320, 256)),
    )

    # 320 x 256 is a whole number of 32 px merge cells and inside the processor's pixel range: 20 x 16 patches.
    assert grids == [[[1, 16, 20]]]
    assert isinstance(result, str)
