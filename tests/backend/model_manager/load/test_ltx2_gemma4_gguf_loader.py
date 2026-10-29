"""The LTX-2 Gemma-4 encoder from a GGUF conversion: the tower with its Linears packed, and the packed tokenizer.

Driven over real (tiny) GGUFs laid out like the published conversions (see ``gemma4_gguf_fixture``). The tower is
checked by its forward -- every hidden state, since LTX-2 stacks all of them -- against a dense model holding
the values the file decodes to.
"""

import dataclasses
from pathlib import Path

import gguf
import pytest
import torch

from invokeai.backend.model_manager.configs.gemma4_encoder import Gemma4Encoder_GGUF_LTX2_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import ltx2
from invokeai.backend.model_manager.load.model_loaders.ltx2 import LTX2Gemma4EncoderGGUFModel
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import HOST_DEQUANT_PEAK_COPIES, GGMLTensor
from tests.backend.model_manager.load.gemma4_gguf_fixture import (
    ELIX3R_CONFIG_KEY,
    TINY_GEMMA4,
    tokenizer_files,
    write_gemma4_gguf,
)
from tests.fixtures.loader_seams import Seam, prepare

SEAM = Seam(
    loader=LTX2Gemma4EncoderGGUFModel, module=ltx2, entry="_load_model", patches_device=True, casts_fp8_storage=False
)
ELIX3R = {
    "qtype": gguf.GGMLQuantizationType.Q4_K,
    "quantized_embedding": True,
    "asset_type": gguf.GGMLQuantizationType.I16,
    "config_key": ELIX3R_CONFIG_KEY,
    "architecture": "gemma4",
}


def _config(path: Path) -> Gemma4Encoder_GGUF_LTX2_Config:
    return Gemma4Encoder_GGUF_LTX2_Config.model_construct(path=str(path), name=path.stem)


def _load(monkeypatch, tmp_path: Path, submodel=SubModelType.TextEncoder, seam: Seam = SEAM, **write_kwargs):
    path = tmp_path / "gemma4-12b-ltx-text-encoder.gguf"
    meant = write_gemma4_gguf(path, **write_kwargs)
    run = prepare(seam, monkeypatch)
    return run.load(_config(path), submodel), meant, run


def _tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """What the file fills: the parameters and the layer scalars, which ``Gemma4TextModel`` keeps as buffers."""
    scalars = {name: buffer for name, buffer in model.named_buffers() if name.endswith("layer_scalar")}
    return {**dict(model.named_parameters()), **scalars}


def _hidden_states(model: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    ids = torch.tensor([[5, 9, 2, 33, 61, 1]])
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True).hidden_states


def _dense_reference(meant: dict[str, torch.Tensor]) -> torch.nn.Module:
    from transformers import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    reference = Gemma4TextModel(Gemma4TextConfig(**TINY_GEMMA4))
    reference.load_state_dict(meant)
    return reference.eval()


@pytest.mark.parametrize(
    ("layout", "tolerance"),
    [
        # The torch kernels scale Q8_0 codes in fp16, the reference reader in float32: after two layers that is a
        # few 1e-3 on values near 4. A weight in the wrong slot is off by tenths -- every Linear's slot is pinned here.
        ({"qtype": gguf.GGMLQuantizationType.Q8_0}, 1e-2),
        # elix3r's builds: K-quants, the embedding quantized too, I16 assets, the config under its own key. Checks
        # the K-quant kernels and the quantized embedding; the full-attention layer's softmax saturates on these
        # random blocks, so its query and key are left to the case above.
        (ELIX3R, 1e-3),
    ],
    ids=["rootport-Q8_0", "elix3r-Q4_K"],
)
def test_the_gguf_tower_computes_every_hidden_state_its_file_means(monkeypatch, tmp_path, layout, tolerance) -> None:
    """LTX-2 stacks all of them (the embedding output and every layer's), so all are compared."""
    model, meant, _ = _load(monkeypatch, tmp_path, **layout)
    apply_custom_layers_to_model(model)

    got, want = _hidden_states(model), _hidden_states(_dense_reference(meant))

    assert len(got) == TINY_GEMMA4["num_hidden_layers"] + 1
    for layer, (a, b) in enumerate(zip(got, want, strict=True)):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance, msg=f"hidden state {layer}")


def test_what_is_read_outside_a_matmul_is_unpacked_and_the_linears_stay_packed(monkeypatch, tmp_path) -> None:
    """``Gemma4RMSNorm`` reads ``weight.float()``, the layer scalars multiply activations directly, and the
    embedding is indexed: none of those can read packed storage."""
    model, _, _ = _load(monkeypatch, tmp_path, **ELIX3R)

    tensors = _tensors(model)
    assert any(name.endswith("layer_scalar") for name in tensors)
    for name, tensor in tensors.items():
        assert isinstance(tensor.data, GGMLTensor) == name.endswith("proj.weight"), name


def test_only_the_tower_is_read_from_the_file(monkeypatch, tmp_path) -> None:
    """The projection alone is ~0.8 GB in a real build; the component folder supplies it, so it is not copied."""
    read: list[str] = []
    real = ltx2.gguf_sd_loader

    def recording(*args, **kwargs):
        sd = real(*args, **kwargs)
        read.extend(sd)
        return sd

    monkeypatch.setattr(ltx2, "gguf_sd_loader", recording)
    _load(monkeypatch, tmp_path)

    assert read and all(key.startswith(("model.embed_tokens.", "model.layers.", "model.norm.")) for key in read)


def test_the_load_reserves_the_tower_as_it_will_be_held(monkeypatch, tmp_path) -> None:
    """One absolute reservation: the packed Linears, everything unpacked at its unpacked size, and the
    quantized embedding in flight -- its packed original plus one slice's decode temporaries (here the
    whole embedding fits one slice)."""
    model, _, run = _load(monkeypatch, tmp_path, **ELIX3R)
    held = {
        name: tensor.data.quantized_data.nbytes
        if isinstance(tensor.data, GGMLTensor)
        else tensor.numel() * tensor.element_size()
        for name, tensor in _tensors(model).items()
    }
    stored = {t.name: t for t in gguf.GGUFReader(tmp_path / "gemma4-12b-ltx-text-encoder.gguf").tensors}
    embedding = held["embed_tokens.weight"]

    in_flight = int(stored["model.embed_tokens.weight"].n_bytes) + HOST_DEQUANT_PEAK_COPIES * embedding
    assert run.reserved == [sum(held.values()) + in_flight]


def test_the_weights_follow_the_devices_compute_dtype(monkeypatch, tmp_path) -> None:
    """On a GPU that dtype is bf16, not the float32 the forwards above run at."""
    model, _, _ = _load(monkeypatch, tmp_path, seam=dataclasses.replace(SEAM, compute_dtype=torch.bfloat16), **ELIX3R)

    for name, tensor in _tensors(model).items():
        dtype = tensor.data.compute_dtype if isinstance(tensor.data, GGMLTensor) else tensor.dtype
        assert dtype == torch.bfloat16, name


@pytest.mark.parametrize("asset_type", [gguf.GGMLQuantizationType.I8, gguf.GGMLQuantizationType.I16])
def test_the_packed_tokenizer_encodes_like_the_folder_it_was_packed_from(monkeypatch, tmp_path, asset_type) -> None:
    from transformers import AutoTokenizer

    folder = tmp_path / "folder"
    folder.mkdir()
    for name, content in tokenizer_files().items():
        (folder / name).write_bytes(content)
    expected = AutoTokenizer.from_pretrained(folder, local_files_only=True, fix_mistral_regex=False)

    tokenizer, _, _ = _load(monkeypatch, tmp_path, submodel=SubModelType.Tokenizer, asset_type=asset_type)

    for side in ("left", "right"):
        tokenizer.padding_side = expected.padding_side = side
        for prompt in ("w1 w2 w3", "▁w1 é 日本 w4", "w57 unknownword w4", ""):
            kwargs = {"padding": "max_length", "max_length": 8, "truncation": True, "return_tensors": "pt"}
            for got, want in zip(
                tokenizer(prompt, **kwargs).values(), expected(prompt, **kwargs).values(), strict=True
            ):
                assert torch.equal(got, want), (side, prompt)
    assert (tokenizer.pad_token, tokenizer.eos_token) == ("<pad>", "<eos>")


def test_a_packed_file_that_is_not_bytes_is_refused_by_name(monkeypatch, tmp_path) -> None:
    path = tmp_path / "gemma4-12b-ltx-text-encoder.gguf"
    write_gemma4_gguf(path, asset_type=gguf.GGMLQuantizationType.I16)
    stored = {t.name: t for t in gguf.GGUFReader(path, "r+").tensors}
    stored["tokenizer_json"].data[0] = 300
    del stored

    with pytest.raises(ValueError, match="tokenizer_json"):
        prepare(SEAM, monkeypatch).load(_config(path), SubModelType.Tokenizer)


def test_each_submodel_is_sized_by_what_it_reads(tmp_path) -> None:
    """The cache reserves this before the load. The tokenizer reads two small files; sized as the whole file it
    would reserve the tower's ~9 GB a second time."""
    path = tmp_path / "gemma4-12b-ltx-text-encoder.gguf"
    write_gemma4_gguf(path, asset_type=gguf.GGMLQuantizationType.I16)
    stored = {t.name: int(t.n_bytes) for t in gguf.GGUFReader(path).tensors}
    loader = object.__new__(LTX2Gemma4EncoderGGUFModel)

    tower = sum(size for name, size in stored.items() if name.startswith("model."))
    files = sum(2 * len(content) for content in tokenizer_files().values())
    assert loader.get_size_fs(_config(path), path, SubModelType.TextEncoder) == tower
    assert loader.get_size_fs(_config(path), path, SubModelType.Tokenizer) == files
