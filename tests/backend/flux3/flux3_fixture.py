"""A tiny FLUX.3 transformer, written out in Black Forest Labs' native key layout.

Shared by the conversion, loader and identification tests. The native names are enumerated here from the
layout every release uses (``tests/backend/model_manager/load/state_dicts/flux3_klein_keys.py`` is the
captured real one), not derived from the converter under test; ``test_conversion`` checks the two agree.

The head width is the released 128: the rotary axes are fixed at (32, 32, 32, 32) and the loader refuses any
other width. Everything else is small. Weights are random rather than the diffusers zero-init, so a forward
through a mis-wired parameter cannot pass by multiplying by zero.
"""

from pathlib import Path

import gguf
import torch
from diffusers import Flux3Transformer2DModel
from gguf.quants import dequantize, quantize

from invokeai.backend.flux3.conversion import flux3_native_key_to_diffusers

TINY_CONFIG = {
    "in_channels": 32,
    "cond_in_channels": 32,
    "num_layers": 1,
    "num_single_layers": 2,
    "attention_head_dim": 128,
    "num_attention_heads": 2,
    "joint_attention_dim": 256,
    "timestep_channels": 32,
}

_BLOCK_PARAMS = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "attn_out.weight",
    "mlp_in.weight",
    "mlp_out.weight",
    "norm.query_norm.scale",
    "norm.key_norm.scale",
)
_ADALN = "final_layer.image.adaLN_modulation.1.weight"


def native_flux3_keys(num_layers: int, num_single_layers: int, flow_map: bool) -> list[str]:
    keys = [
        "emb_in.image.weight",
        "emb_in.image_cond.weight",
        "txt_in.weight",
        "time_in.in_layer.weight",
        "time_in.out_layer.weight",
        "final_layer.image.linear.weight",
        _ADALN,
    ]
    if flow_map:
        keys += ["ref_time_in.in_layer.weight", "ref_time_in.out_layer.weight"]
    for stage in ("early_stream_modulations", "single_stream_modulations"):
        keys += [f"{stage}.{mode}.lin.weight" for mode in ("image", "image_cond", "txt")]
    stacks = [f"content_mode_blocks.{mode}.{i}" for mode in ("image", "image_cond") for i in range(num_layers)]
    stacks += [f"txt_mode_blocks.{i}" for i in range(num_layers)]
    stacks += [f"single_blocks.{i}" for i in range(num_single_layers)]
    keys += [f"{stack}.{param}" for stack in stacks for param in _BLOCK_PARAMS]
    return keys


def tiny_flux3_model(flow_map: bool, seed: int = 0) -> Flux3Transformer2DModel:
    torch.manual_seed(seed)
    model = Flux3Transformer2DModel(**TINY_CONFIG, flow_map=flow_map).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.05)
    return model


def to_native(model: Flux3Transformer2DModel) -> dict[str, torch.Tensor]:
    """The model's weights under BFL's names, with the output adaLN in BFL's [shift, scale] order."""
    diffusers_sd = model.state_dict()
    native = {}
    for key in native_flux3_keys(TINY_CONFIG["num_layers"], TINY_CONFIG["num_single_layers"], model.config.flow_map):
        value = diffusers_sd[flux3_native_key_to_diffusers(key)].clone()
        if key == _ADALN:
            scale, shift = value.chunk(2, dim=0)
            value = torch.cat([shift, scale], dim=0)
        native[key] = value
    return native


def forward(model: torch.nn.Module, flow_map: bool, with_reference: bool = True) -> torch.Tensor:
    """One step over four image tokens, two reference tokens and three text tokens."""
    generator = torch.Generator().manual_seed(1)
    kwargs = {}
    if with_reference:
        kwargs = {
            "image_cond_hidden_states": torch.randn(1, 2, TINY_CONFIG["cond_in_channels"], generator=generator),
            "image_cond_ids": torch.tensor([[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.5, -1.0]]),
        }
    with torch.no_grad():
        return model(
            hidden_states=torch.randn(1, 4, TINY_CONFIG["in_channels"], generator=generator),
            encoder_hidden_states=torch.randn(1, 3, TINY_CONFIG["joint_attention_dim"], generator=generator),
            timestep=torch.tensor([0.7]),
            img_ids=torch.tensor([[0.0, h * 0.5, w * 0.5, 0.0] for h in range(2) for w in range(2)]),
            txt_ids=torch.tensor([[0.0, 0.0, 0.0, float(i)] for i in range(3)]),
            target_timestep=torch.tensor([0.4]) if flow_map else None,
            return_dict=False,
            **kwargs,
        )[0]


def write_flux3_gguf(
    path: Path,
    native: dict[str, torch.Tensor],
    *,
    qtype: gguf.GGMLQuantizationType = gguf.GGMLQuantizationType.Q8_0,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Write ``native`` the way the published GGUFs do: 2D weights in ``qtype``, norms in F32, the rest BF16.

    The architecture string is the published one, ``flux``. Returns the values a reader dequantizes each
    tensor to, so a reference model can be built from what the file means.
    """
    writer = gguf.GGUFWriter(str(path), "flux")
    block_size = gguf.GGML_QUANT_SIZES[qtype][0]
    meant: dict[str, torch.Tensor] = {}
    for name, tensor in native.items():
        data = tensor.detach().to(torch.float32).numpy()
        if name.endswith(".scale"):
            stored_as = gguf.GGMLQuantizationType.F32
        elif data.ndim == 2 and data.shape[-1] % block_size == 0 and not name.startswith(("final_layer", "time_in")):
            stored_as = qtype
        else:
            stored_as = gguf.GGMLQuantizationType.BF16
        raw = quantize(data, stored_as)
        writer.add_tensor(prefix + name, raw, raw_shape=raw.shape, raw_dtype=stored_as)
        meant[name] = torch.from_numpy(dequantize(raw, stored_as)).reshape(tensor.shape)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return meant
