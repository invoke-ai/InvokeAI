"""Exercise each Z-Image Qwen3 embedding loader through its real CPU cache and files."""

from unittest.mock import Mock

import gguf
import pytest
import torch
import transformers
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_GGUF_Config, Qwen3Encoder_SDNQ_Config
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.load.model_loaders import z_image
from invokeai.backend.quantization.sdnq.sdnq_tensor import SDNQTensor
from tests.fixtures.quantized_payloads import write_gguf

VOCAB, HIDDEN = 32, 128


@pytest.mark.parametrize("kind", ["gguf", "sdnq", "diffusers_sdnq"])
def test_qwen3_loader_reserves_live_storage_before_embedding_decode(monkeypatch, tmp_path, kind):
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        tie_word_embeddings=True,
    )
    sd = {k: v.detach().clone() for k, v in Qwen3ForCausalLM(config).state_dict().items() if k != "lm_head.weight"}
    sd["unused_payload"] = torch.ones(13)
    embed = sd.pop("model.embed_tokens.weight")
    if kind == "gguf":
        path = tmp_path / "tiny.gguf"
        write_gguf(path, sd, q8_0={"model.embed_tokens.weight": embed}, architecture="qwen3")
        reader = gguf.GGUFReader(path)
        checkpoint_bytes = sum(int(t.n_bytes) for t in reader.tensors)
        del reader
    else:
        path = tmp_path / "tiny.safetensors"
        sd["model.embed_tokens.weight"] = torch.full((VOCAB, HIDDEN), 10, dtype=torch.int8)
        sd["model.embed_tokens.scale"] = torch.tensor([0.1])
        # A second wrapper's independent auxiliary storage stays live during embedding decode.
        sd["model.layers.0.self_attn.q_proj.weight"] = torch.ones((HIDDEN, HIDDEN), dtype=torch.int8)
        sd["model.layers.0.self_attn.q_proj.scale"] = torch.tensor([0.2])
        sd["model.layers.0.self_attn.q_proj.zero_point"] = torch.tensor([1.0])
        checkpoint_bytes = sum(t.nbytes for t in sd.values())
        save_file(sd, path)

    observed_buffers = []

    class ModelWithExtraParameter(Qwen3ForCausalLM):
        def load_state_dict(self, *args, **kwargs):
            result = super().load_state_dict(*args, **kwargs)
            assert self.lm_head.weight.is_meta
            # This storage is not in the checkpoint and cannot be counted through its aliases.
            self.extra = torch.nn.Parameter(torch.ones(19), requires_grad=False)
            self.register_buffer("extra_buffer", torch.ones(23), persistent=False)
            observed_buffers.append(sum(b.untyped_storage().nbytes() for b in self.buffers() if not b.is_meta))
            return result

    monkeypatch.setattr(transformers, "Qwen3ForCausalLM", ModelWithExtraParameter)
    monkeypatch.setattr(z_image.TorchDevice, "choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(z_image.TorchDevice, "choose_bfloat16_safe_dtype", lambda _: torch.float32)
    cache = ModelCache(
        execution_device_working_mem_gb=0,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=False,
        max_ram_cache_size_gb=1,
        execution_device="cpu",
        shared_cpu_weights=None,
    )
    cache.make_room = Mock(wraps=cache.make_room)
    if kind != "gguf":
        real_decode = SDNQTensor.get_dequantized_tensor

        def decode_after_reservation(weight):
            assert cache.make_room.call_count == 1
            return real_decode(weight)

        monkeypatch.setattr(SDNQTensor, "get_dequantized_tensor", decode_after_reservation)

    try:
        if kind == "gguf":
            loader = object.__new__(z_image.Qwen3EncoderGGUFLoader)
            loader._ram_cache = cache
            model = loader._load_from_gguf(Qwen3Encoder_GGUF_Config.model_construct(path=str(path)))
        elif kind == "sdnq":
            loader = object.__new__(z_image.Qwen3EncoderSDNQLoader)
            loader._ram_cache = cache
            model = loader._load_from_sdnq(Qwen3Encoder_SDNQ_Config.model_construct(path=str(path)))
        else:
            loader = object.__new__(z_image.ZImageDiffusersModel)
            loader._ram_cache = cache
            model = loader._load_sdnq_text_encoder(path)

        output_bytes = VOCAB * HIDDEN * torch.float32.itemsize
        decode_budget = 7 * output_bytes if kind == "gguf" else 6 * output_bytes
        cache.make_room.assert_called_once_with(
            checkpoint_bytes + 19 * 4 + observed_buffers[0] + decode_budget,
            host_bytes_needed=decode_budget,
        )
        assert model.lm_head.weight is model.model.embed_tokens.weight
        assert model.model.embed_tokens(torch.tensor([[0, 1]])).shape == (1, 2, HIDDEN)
    finally:
        cache.shutdown()
