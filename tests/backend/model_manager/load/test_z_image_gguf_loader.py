"""Z-Image GGUF transformers, driven over a real (tiny) GGUF file through the loader and the diffusers forward.

The file is laid out the way the published Z-Image Turbo GGUFs are (read from the Q4_K_S header): original
(non-diffusers) keys with a fused ``qkv``, the main ``layers`` GGML-quantized, the refiners, embedders and final
layer in BF16, every norm and bias in F32 -- and both pad tokens F32 with GGUF shape ``[dim, 1]``, which the
reader hands over as ``(1, dim)``. gguf-py cannot write K-quants, so ``Q8_0`` stands in for the release's
Q4_K/Q5_K; which GGML type the Linears carry is not what these tests are about.
"""

from pathlib import Path

import diffusers
import gguf
import numpy as np
import torch
from diffusers import ZImageTransformer2DModel
from gguf.quants import dequantize, quantize

from invokeai.backend.model_manager.configs.main import Main_GGUF_ZImage_Config
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders import z_image
from invokeai.backend.model_manager.load.model_loaders.z_image import ZImageGGUFCheckpointModel
from tests.fixtures.loader_seams import Seam, prepare

# Every quantized Linear input a multiple of 32, the Q8_0 block size: dim 96 makes the feed-forward width
# int(96 / 3 * 8) = 256. head_dim is 32 and the RoPE axes sum to it, as the released (32, 48, 48) sums to 3840 / 30.
TINY_CONFIG = {
    "all_patch_size": (2,),
    "all_f_patch_size": (1,),
    "in_channels": 16,
    "dim": 96,
    "n_layers": 1,
    "n_refiner_layers": 1,
    "n_heads": 3,
    "n_kv_heads": 3,
    "cap_feat_dim": 32,
    "axes_dims": [8, 12, 12],
}

SEAM = Seam(
    loader=ZImageGGUFCheckpointModel,
    module=z_image,
    patches_device=True,
    sets_torch_dtype=False,
    casts_fp8_storage=False,
)


def _original_key(key: str) -> str:
    """The checkpoint's own name for a diffusers key (the fused ``qkv`` is assembled by the caller)."""
    for diffusers_prefix, original_prefix in (
        ("all_x_embedder.2-1.", "x_embedder."),
        ("all_final_layer.2-1.", "final_layer."),
    ):
        if key.startswith(diffusers_prefix):
            return original_prefix + key[len(diffusers_prefix) :]
    return (
        key.replace(".attention.to_out.0.", ".attention.out.")
        .replace(".attention.norm_q.", ".attention.q_norm.")
        .replace(".attention.norm_k.", ".attention.k_norm.")
    )


def write_z_image_gguf(path: Path, seed: int = 0) -> dict[str, torch.Tensor]:
    """Write a randomly initialised tiny transformer to ``path``; return, by diffusers key, what each tensor means."""
    torch.manual_seed(seed)
    model = ZImageTransformer2DModel(**TINY_CONFIG)
    # The pad tokens start as zeros, which would make a lost or garbled one indistinguishable from a correct one.
    with torch.no_grad():
        for param in model.parameters():
            if not param.any():
                param.normal_(std=0.02)
    sd = {key: value.detach().to(torch.float32) for key, value in model.state_dict().items()}

    original: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if ".attention.to_q." in key:
            prefix, suffix = key.split(".attention.to_q.")
            parts = [sd[f"{prefix}.attention.{name}.{suffix}"] for name in ("to_q", "to_k", "to_v")]
            original[f"{prefix}.attention.qkv.{suffix}"] = torch.cat(parts, dim=0)
        elif ".attention.to_k." not in key and ".attention.to_v." not in key:
            original[_original_key(key)] = value

    writer = gguf.GGUFWriter(str(path), "lumina2")
    meant: dict[str, torch.Tensor] = {}
    for name, tensor in original.items():
        data = tensor.numpy()
        if data.ndim == 1 or name.endswith("_pad_token"):
            # A (1, dim) array is written as GGUF shape [dim, 1], as the released pad tokens are.
            writer.add_tensor(name, data)
            meant[name] = tensor
            continue
        stored_as = gguf.GGMLQuantizationType.Q8_0 if name.startswith("layers.") else gguf.GGMLQuantizationType.BF16
        raw = quantize(data, stored_as)
        # `raw_shape` is the packed byte shape; gguf derives the logical one from it.
        writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=stored_as)
        meant[name] = torch.from_numpy(np.array(dequantize(raw, stored_as))).reshape(tensor.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Back to diffusers keys for the reference model.
    by_diffusers_key: dict[str, torch.Tensor] = {}
    for key in sd:
        if ".attention.to_" in key and ".attention.to_out." not in key:
            prefix, rest = key.split(".attention.")
            name, suffix = rest.split(".", 1)
            rows = TINY_CONFIG["dim"]
            index = ("to_q", "to_k", "to_v").index(name)
            by_diffusers_key[key] = meant[f"{prefix}.attention.qkv.{suffix}"][index * rows : (index + 1) * rows]
        else:
            by_diffusers_key[key] = meant[_original_key(key)]
    return by_diffusers_key


def _load(monkeypatch, tmp_path: Path) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    path = tmp_path / "z-image-turbo-Q8_0.gguf"
    meant = write_z_image_gguf(path)
    config = Main_GGUF_ZImage_Config.model_construct(path=str(path), name="z-image-turbo")
    run = prepare(
        SEAM,
        monkeypatch,
        # The loader builds the released geometry; this file holds the tiny one.
        geometry=lambda patch: patch.setattr(
            diffusers, "ZImageTransformer2DModel", lambda **_kwargs: ZImageTransformer2DModel(**TINY_CONFIG)
        ),
    )
    return run.load(config), meant


def _forward(model: torch.nn.Module) -> torch.Tensor:
    """One image of 2x2 patches and a three-token caption: both are padded to 32 tokens, and with a single
    item there is no attention mask, so every real token attends to the pad tokens."""
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        (out,) = model(
            x=[torch.randn(TINY_CONFIG["in_channels"], 1, 4, 4, generator=generator)],
            t=torch.tensor([0.5]),
            cap_feats=[torch.randn(3, TINY_CONFIG["cap_feat_dim"], generator=generator)],
            return_dict=False,
        )
    return out[0]


def test_a_gguf_with_two_dimensional_pad_tokens_computes_what_its_file_means(monkeypatch, tmp_path) -> None:
    """Not bit-exact: the torch kernels multiply Q8_0 codes by their fp16 block scale in fp16, the reference reader
    in float32. A pad token that reached the forward packed fails outright; a wrong one is off by far more."""
    model, meant = _load(monkeypatch, tmp_path)
    apply_custom_layers_to_model(model)
    reference = ZImageTransformer2DModel(**TINY_CONFIG)
    reference.load_state_dict(meant)

    assert torch.allclose(_forward(model), _forward(reference), atol=1e-3, rtol=1e-3)
