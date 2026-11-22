"""Loading a llama.cpp UMT5 encoder GGUF for Wan 2.2 (``city96/umt5-xxl-encoder-gguf``).

Each test writes a real, tiny GGUF (see ``umt5_gguf_fixture``) and runs the production loader over
it, so the tensors arrive as the ``GGMLTensor`` wrappers the cache dequantizes at runtime.
"""

from contextlib import ExitStack
from logging import getLogger
from pathlib import Path
from unittest.mock import MagicMock, Mock

import gguf
import pytest
import torch

from invokeai.app.services.config.config_default import load_config_from_root
from invokeai.backend.model_manager.configs.wan_t5_encoder import WanT5Encoder_GGUF_Config
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache, get_model_cache_key
from invokeai.backend.model_manager.load.model_cache.shared_cpu_weights import SharedCpuWeightsStore
from invokeai.backend.model_manager.load.model_loaders.wan import WanT5EncoderGGUFLoader
from invokeai.backend.model_manager.taxonomy import SubModelType
from invokeai.backend.quantization.gguf.ggml_tensor import HOST_DEQUANT_PEAK_COPIES, GGMLTensor
from invokeai.backend.quantization.gguf.loaders import WrappedGGUFReader
from invokeai.backend.util.devices import TorchDevice
from tests.backend.model_manager.load.umt5_gguf_fixture import (
    EOS_ID,
    PAD_ID,
    PIECE_IDS,
    UNK_ID,
    write_tiny_umt5_gguf,
)


def _loader(ram_cache=None) -> WanT5EncoderGGUFLoader:
    loader = object.__new__(WanT5EncoderGGUFLoader)
    loader._ram_cache = MagicMock(execution_device=torch.device("cpu")) if ram_cache is None else ram_cache
    return loader


def _config(path: Path, *, cpu_only: bool = False) -> WanT5Encoder_GGUF_Config:
    return WanT5Encoder_GGUF_Config.model_construct(
        key="tiny-umt5", path=str(path), name="tiny umt5", cpu_only=cpu_only
    )


@pytest.fixture
def float32_on_cpu(monkeypatch):
    """Load in float32: CPU bfloat16 matmuls fault on part of the CI fleet (see tests/AGENTS.md)."""
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", lambda _device=None: torch.float32)


@pytest.fixture
def written(tmp_path) -> tuple[Path, torch.nn.Module]:
    path = tmp_path / "umt5-tiny-Q8_0.gguf"
    return path, write_tiny_umt5_gguf(path)


@pytest.fixture
def make_loader(tmp_path):
    app_config = load_config_from_root(tmp_path)
    shared_weights = SharedCpuWeightsStore()
    with ExitStack() as stack:

        def make(device: str) -> WanT5EncoderGGUFLoader:
            cache = ModelCache(
                execution_device=device,
                execution_device_working_mem_gb=0,
                enable_partial_loading=False,
                keep_ram_copy_of_weights=True,
                max_ram_cache_size_gb=1,
                shared_cpu_weights=shared_weights,
            )
            stack.callback(cache.shutdown)
            return WanT5EncoderGGUFLoader(app_config, getLogger(__name__), cache)

        yield make


@pytest.fixture
def bfloat16_host(monkeypatch):
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(TorchDevice, "choose_torch_dtype", lambda: torch.float16)
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", lambda device: torch.bfloat16)


@pytest.mark.parametrize(
    ("cache_device", "cpu_only"),
    [("cuda:0", True), ("cpu", False)],
    ids=["cpu-only-in-accelerator-cache", "cpu-cache"],
)
def test_cpu_execution_uses_float32_and_matches_the_dense_fixture(
    written, make_loader, bfloat16_host, cache_device, cpu_only
) -> None:
    path, reference = written
    loader = make_loader(cache_device)
    loaded = loader.load_model(_config(path, cpu_only=cpu_only), SubModelType.TextEncoder)

    assert loaded.compute_device == torch.device("cpu")
    with loaded as model, torch.no_grad():
        assert model.shared.weight.dtype == torch.float32
        assert model.encoder.embed_tokens.weight is model.shared.weight
        linears = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
        assert len(linears) == 7 * len(model.encoder.block)
        for linear in linears:
            assert isinstance(linear.weight, GGMLTensor)
            assert linear.weight.dtype == torch.uint8
            assert linear.weight.compute_dtype == torch.float32
            assert linear.weight.shape == (linear.out_features, linear.in_features)
        assert all(
            parameter.dtype == torch.float32
            for parameter in model.parameters()
            if not isinstance(parameter, GGMLTensor)
        )
        input_ids = torch.tensor([[5, 17, 9, 30, 6, EOS_ID, PAD_ID, PAD_ID]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
        actual = model(input_ids, attention_mask).last_hidden_state
        expected = reference(input_ids, attention_mask).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_cache_keeps_encoder_dtypes_separate_and_reuses_the_tokenizer(written, make_loader, bfloat16_host) -> None:
    path, _ = written
    loader = make_loader("cuda:0")
    accelerated = _config(path)
    cpu = _config(path, cpu_only=True)

    # Read the accelerated entry in RAM. This exercises real cache admission on a CPU build of torch.
    with loader.load_model(accelerated, SubModelType.TextEncoder).model_in_ram() as bf16:
        assert bf16.shared.weight.dtype == torch.bfloat16
        cpu_cache_loader = make_loader("cpu")
        with cpu_cache_loader.load_model(accelerated, SubModelType.TextEncoder) as peer_fp32:
            assert peer_fp32.shared.weight.dtype == torch.float32
        with loader.load_model(cpu, SubModelType.TextEncoder) as fp32:
            assert fp32 is not bf16
            assert fp32.shared.weight.dtype == torch.float32
            assert fp32.shared.weight.data_ptr() == peer_fp32.shared.weight.data_ptr()

    encoder_key = get_model_cache_key(cpu.key, SubModelType.TextEncoder)
    assert loader.ram_cache.get(f"{encoder_key}:accelerator:bfloat16").cached_model.model is bf16
    assert loader.ram_cache.get(f"{encoder_key}:cpu:float32").cached_model.model is fp32
    with loader.load_model(cpu, SubModelType.TextEncoder) as warm_fp32:
        assert warm_fp32 is fp32

    with loader.load_model(cpu, SubModelType.Tokenizer).model_in_ram() as tokenizer:
        assert tokenizer("a cat").input_ids == [PIECE_IDS["▁a"], PIECE_IDS["▁cat"], EOS_ID]
    tokenizer_key = get_model_cache_key(cpu.key, SubModelType.Tokenizer)
    assert loader.ram_cache.get(tokenizer_key).cached_model.model is tokenizer
    with loader.load_model(accelerated, SubModelType.Tokenizer).model_in_ram() as warm_tokenizer:
        assert warm_tokenizer is tokenizer


def test_cpu_only_reload_bypasses_a_pinned_accelerator_fp32_entry(written, make_loader, bfloat16_host, monkeypatch):
    path, _ = written
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", lambda device: torch.float16)
    loader = make_loader("cuda:0")
    original = _config(path)
    accelerator = loader.load_model(original, SubModelType.TextEncoder)

    assert accelerator.compute_device == torch.device("cuda:0")
    with accelerator.model_in_ram() as old_model:
        assert old_model.shared.weight.dtype == torch.float32
        # An in-flight holder keeps the old record alive through invalidation.
        assert loader.ram_cache.drop_model(original.key) == 0
        cpu_config = original.model_copy(update={"cpu_only": True})
        cpu = loader.load_model(cpu_config, SubModelType.TextEncoder)

        assert cpu.compute_device == torch.device("cpu")
        with cpu as new_model:
            assert new_model is not old_model
            assert new_model.shared.weight.dtype == torch.float32
            assert new_model.shared.weight.data_ptr() != old_model.shared.weight.data_ptr()
        assert accelerator.compute_device == torch.device("cuda:0")

    with loader.load_model(cpu_config, SubModelType.TextEncoder) as warm_model:
        assert warm_model is new_model


@pytest.mark.parametrize(
    "matrix_type",
    [
        gguf.GGMLQuantizationType.Q8_0,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
        gguf.GGMLQuantizationType.F32,
    ],
    ids=lambda matrix_type: matrix_type.name,
)
@pytest.mark.parametrize(
    ("cache_device", "cpu_only"),
    [("cpu", False), ("cuda:0", True)],
    ids=["cpu-cache", "cpu-only-in-accelerator-cache"],
)
def test_cold_load_reserves_held_weights_and_unpack_workspace(
    tmp_path, monkeypatch, make_loader, bfloat16_host, matrix_type, cache_device, cpu_only
) -> None:
    path = tmp_path / "reserve.gguf"
    reference = write_tiny_umt5_gguf(path, matrix_type=matrix_type)
    held = sum(parameter.nbytes for parameter in reference.parameters())
    with WrappedGGUFReader(path) as reader:
        embedding_bytes = next(tensor.n_bytes for tensor in reader.tensors if tensor.name == "token_embd.weight")
        projections = [
            tensor
            for tensor in reader.tensors
            if tensor.name.rsplit(".", 2)[-2]
            in {"attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate", "ffn_up", "ffn_down"}
        ]
        if matrix_type is gguf.GGMLQuantizationType.Q8_0:
            held -= sum(int(tensor.n_elements) * 4 - int(tensor.n_bytes) for tensor in projections)
            in_flight = embedding_bytes + HOST_DEQUANT_PEAK_COPIES * reference.shared.weight.nbytes
        else:
            in_flight = max(tensor.n_bytes for tensor in reader.tensors)

    loader = make_loader(cache_device)
    reserve = Mock(wraps=loader.ram_cache.make_room)
    monkeypatch.setattr(loader.ram_cache, "make_room", reserve)
    loaded = loader.load_model(_config(path, cpu_only=cpu_only), SubModelType.TextEncoder)
    assert loaded.compute_device == torch.device("cpu")
    with loaded as model:
        assert model.shared.weight.dtype == torch.float32
    assert loaded.weight_bytes == held
    assert [call.args[0] for call in reserve.call_args_list] == [max(path.stat().st_size, held), held + in_flight]


@pytest.mark.parametrize(
    "matrix_type",
    [gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16],
    ids=lambda matrix_type: matrix_type.name,
)
def test_unquantized_matrices_are_unpacked_to_the_compute_dtype(tmp_path, float32_on_cpu, matrix_type) -> None:
    """city96 also ships F16 (10.6 GiB) and F32 (21.2 GiB) builds. Kept packed, the F32 one would stay
    that size in memory and every unquantized matrix would be converted on every forward; unpacked, they
    are the compute dtype's plain weights, and the encoding is the same."""
    path = tmp_path / f"umt5-tiny-{matrix_type.name}.gguf"
    reference = write_tiny_umt5_gguf(path, matrix_type=matrix_type)
    model = _loader()._load_model(_config(path), SubModelType.TextEncoder).eval()

    packed = [name for name, param in model.named_parameters() if isinstance(param, GGMLTensor)]
    assert packed == []
    assert {param.dtype for param in model.parameters()} == {torch.float32}
    input_ids = torch.tensor([[5, 17, 9, 30, 6, EOS_ID]])
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids).last_hidden_state, reference(input_ids).last_hidden_state)


def test_a_device_without_bfloat16_computes_in_float32_not_float16(written, make_loader, monkeypatch) -> None:
    """UMT5's feed-forward activations overflow float16, which transformers guards against by keeping
    ``wo`` in float32 -- not possible for a packed ``wo``, so the loader never picks float16."""
    path, _ = written
    monkeypatch.setattr(TorchDevice, "choose_torch_device", lambda: torch.device("cuda:1"))
    monkeypatch.setattr(TorchDevice, "choose_torch_dtype", lambda: torch.float16)
    safe_dtypes = {torch.device("cuda:0"): torch.float16, torch.device("cuda:1"): torch.bfloat16}
    monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", safe_dtypes.__getitem__)

    loader = make_loader("cuda:0")
    with loader.load_model(_config(path), SubModelType.TextEncoder).model_in_ram() as model:
        assert model.shared.weight.dtype == torch.float32
        assert {module.weight.compute_dtype for module in model.modules() if isinstance(module, torch.nn.Linear)} == {
            torch.float32
        }


def test_the_tokenizer_comes_from_the_gguf_metadata(written) -> None:
    """What the Wan node asks of it: runs of spaces collapse, ``</s>`` ends the prompt, padding uses
    ``<pad>``. A character outside the vocabulary becomes ``<unk>`` -- found by its token type, since
    the real file has no ``unknown_token_id`` -- and a sentinel typed into the prompt stays whole."""
    path, _ = written
    tokenizer = _loader()._load_model(_config(path), SubModelType.Tokenizer)
    words = [PIECE_IDS[piece] for piece in ("▁a", "▁cat", "▁on", "▁the", "▁mat")]

    assert tokenizer("a cat  on   the mat").input_ids == [*words, EOS_ID]
    assert tokenizer("a cat on the mat").input_ids == [*words, EOS_ID]
    padded = tokenizer(["a cat"], padding="max_length", max_length=6, truncation=True, return_tensors="pt")
    assert padded.input_ids.tolist() == [[PIECE_IDS["▁a"], PIECE_IDS["▁cat"], EOS_ID, PAD_ID, PAD_ID, PAD_ID]]
    assert padded.attention_mask.tolist() == [[1, 1, 1, 0, 0, 0]]
    assert tokenizer("a 猫").input_ids == [PIECE_IDS["▁a"], PIECE_IDS["▁"], UNK_ID, EOS_ID]
    # The `▁` prefix goes on every word run, also one that follows a sentinel without a space.
    assert tokenizer("<extra_id_0>a").input_ids == [PIECE_IDS["<extra_id_0>"], PIECE_IDS["▁a"], EOS_ID]
    # A prompt over the length is cut, and `</s>` still ends it.
    truncated = tokenizer(["a cat on the mat"], padding="max_length", max_length=4, truncation=True)
    assert truncated.input_ids == [[PIECE_IDS["▁a"], PIECE_IDS["▁cat"], PIECE_IDS["▁on"], EOS_ID]]
    assert truncated.attention_mask == [[1, 1, 1, 1]]


def test_a_gguf_without_tokenizer_fields_fails_naming_the_field(tmp_path, make_loader) -> None:
    path = tmp_path / "no-tokenizer.gguf"
    write_tiny_umt5_gguf(path, with_tokenizer=False)

    with pytest.raises(ValueError, match="tokenizer.ggml"):
        make_loader("cpu").load_model(_config(path), SubModelType.Tokenizer)


def test_an_incomplete_encoder_never_enters_the_cache(tmp_path, make_loader, float32_on_cpu) -> None:
    path = tmp_path / "missing-norm.gguf"
    write_tiny_umt5_gguf(path, omit_tensor="enc.output_norm.weight")
    loader = make_loader("cpu")
    config = _config(path)

    with pytest.raises(RuntimeError, match=r"encoder\.final_layer_norm\.weight"):
        loader.load_model(config, SubModelType.TextEncoder)
    assert loader.ram_cache.drop_model(config.key) == 0


def test_tokenizer_makes_no_ram_reservation(written, make_loader, bfloat16_host, monkeypatch) -> None:
    """The cache makes room for this size before a load; the tokenizer reads only the header, and
    sizing it by the file would evict gigabytes to load a few megabytes."""
    path, _ = written
    loader = make_loader("cuda:0")
    reserve = Mock(wraps=loader.ram_cache.make_room)
    monkeypatch.setattr(loader.ram_cache, "make_room", reserve)

    with loader.load_model(_config(path), SubModelType.Tokenizer).model_in_ram() as tokenizer:
        assert tokenizer("a").input_ids == [PIECE_IDS["▁a"], EOS_ID]
    reserve.assert_called_once_with(0)
