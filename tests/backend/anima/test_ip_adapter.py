"""The SigLIP2 Anima IP-Adapter against the reference ComfyUI node (LuciferTC9527/ComfyUI-Anima_IP-Adapter).

The expected values come from `_reference_forward`, a transcription of the node's monkey-patched block forward: it
captures each cross-attention's input with a pre-hook, runs the original block, then adds the IP term with the
node's own reshapes, its `scale_mask`, and its `_LoRALinear` wrappers on the cross-attention projections.
"""

import copy

import accelerate
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from transformers import SiglipVisionConfig, SiglipVisionModel

from invokeai.backend.anima.anima_transformer import AnimaTransformer, CosmosAttention, MiniTrainDIT
from invokeai.backend.anima.ip_adapter import (
    AnimaIPAdapter,
    AnimaIPAdapterExtension,
    combine_block_fns,
    encode_image_tokens,
    siglip_pixel_values,
)
from invokeai.backend.model_manager.configs.ip_adapter import IPAdapter_Checkpoint_Anima_Config
from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.model_manager.load.model_loaders.anima import (
    ANIMA_TRANSFORMER_CONFIG,
    AnimaIPAdapterModelLoader,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.anima_lora_constants import ANIMA_LORA_TRANSFORMER_PREFIX
from tests.backend.patches.lora_conversions.lora_state_dicts.anima_ip_adapter_format import (
    NUM_BLOCKS,
    metadata,
    state_dict_keys,
)

_TINY_DIT = {
    "max_img_h": 64,
    "max_img_w": 64,
    "model_channels": 64,
    "num_blocks": 2,
    "num_heads": 4,
    "crossattn_emb_channels": 32,
    "use_adaln_lora": True,
    "adaln_lora_dim": 16,
    "rope_h_extrapolation_ratio": 4.0,
    "rope_w_extrapolation_ratio": 4.0,
}
_TOKEN_DIM = 12
_LORA_RANK = 4
_CROSS_ATTN_IN = {"q_proj": 64, "k_proj": 32, "v_proj": 32, "output_proj": 64}
_CPU = torch.device("cpu")


def _tiny_dit() -> MiniTrainDIT:
    torch.manual_seed(0)
    model = MiniTrainDIT(**_TINY_DIT).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return model


def _tiny_checkpoint(num_blocks: int = 2) -> dict[str, torch.Tensor]:
    """Checkpoint-named weights for the tiny DiT: IP projections and gates, the cross-attention LoRA, and a
    self-attention LoRA layer the reference ignores."""
    gen = torch.Generator().manual_seed(1)
    sd: dict[str, torch.Tensor] = {}
    x_dim = _TINY_DIT["model_channels"]
    for n in range(num_blocks):
        for proj in ("ip_k_proj", "ip_v_proj"):
            sd[f"blocks.{n}.{proj}.weight"] = torch.randn(x_dim, _TOKEN_DIM, generator=gen) * 0.2
            sd[f"blocks.{n}.{proj}.bias"] = torch.randn(x_dim, generator=gen) * 0.2
        sd[f"blocks.{n}.adaln_ip.1.weight"] = torch.randn(x_dim, x_dim, generator=gen) * 0.2
        sd[f"blocks.{n}.adaln_ip.1.bias"] = torch.randn(x_dim, generator=gen) * 0.2
        for sub, in_dims in (("cross_attn", _CROSS_ATTN_IN), ("self_attn", {"q_proj": x_dim})):
            for proj, in_dim in in_dims.items():
                prefix = f"lora.base_model.model.blocks.{n}.{sub}.{proj}"
                sd[f"{prefix}.lora_A.weight"] = torch.randn(_LORA_RANK, in_dim, generator=gen) * 0.2
                sd[f"{prefix}.lora_B.weight"] = torch.randn(x_dim, _LORA_RANK, generator=gen) * 0.2
    return sd


class _LoRALinear(torch.nn.Module):
    """The reference's unmerged LoRA wrapper (`nodes.py` `_LoRALinear`), at its scale alpha / rank = 1."""

    def __init__(self, base: torch.nn.Linear, lora_A: torch.Tensor, lora_B: torch.Tensor):
        super().__init__()
        self.base, self.lora_A, self.lora_B = base, lora_A, lora_B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * 1.0


def _reference_forward(model, sd, images, x, t, ctx) -> torch.Tensor:
    """The reference node's forward for `images`, a list of (tokens, strength). It takes one image; for several, the
    scaled tokens are concatenated into one key/value set, the computation the extension claims."""
    blocks = list(model.blocks)
    saved = [(b.forward, {p: getattr(b.cross_attn, p) for p in _CROSS_ATTN_IN}) for b in blocks]
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for i, blk in enumerate(blocks):
        for proj in _CROSS_ATTN_IN:
            prefix = f"lora.base_model.model.blocks.{i}.cross_attn.{proj}"
            setattr(
                blk.cross_attn,
                proj,
                _LoRALinear(
                    getattr(blk.cross_attn, proj), sd[f"{prefix}.lora_A.weight"], sd[f"{prefix}.lora_B.weight"]
                ),
            )
        hooks.append(blk.cross_attn.register_forward_pre_hook(lambda _m, args, i=i: captured.__setitem__(i, args[0])))

        def patched(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, _i=i, _blk=blk, _orig=blk.forward, **kwargs):
            result = _orig(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs)
            x_q = captured[_i]
            ip_tok = torch.cat([tokens * strength for tokens, strength in images], dim=1).to(dtype=x_q.dtype)
            B, T, H, W = x_B_T_H_W_D.shape[:4]
            n_h, h_d = _blk.cross_attn.n_heads, _blk.cross_attn.head_dim
            w = {k[len(f"blocks.{_i}.") :]: v for k, v in sd.items() if k.startswith(f"blocks.{_i}.")}
            gate_ip = F.linear(F.silu(emb_B_T_D), w["adaln_ip.1.weight"], w["adaln_ip.1.bias"])
            scale_mask = (ip_tok.abs().sum(dim=[1, 2]) > 1e-6).to(dtype=result.dtype).reshape(B, 1, 1)
            ip_q = _blk.cross_attn.q_proj(x_q).reshape(B, -1, n_h, h_d).permute(0, 2, 1, 3)
            ip_q = _blk.cross_attn.q_norm(ip_q)
            ip_k = F.linear(ip_tok, w["ip_k_proj.weight"], w["ip_k_proj.bias"]).reshape(B, -1, n_h, h_d)
            ip_v = F.linear(ip_tok, w["ip_v_proj.weight"], w["ip_v_proj.bias"]).reshape(B, -1, n_h, h_d)
            ip_attn = F.scaled_dot_product_attention(ip_q, ip_k.permute(0, 2, 1, 3), ip_v.permute(0, 2, 1, 3))
            ip_out = ip_attn.permute(0, 2, 1, 3).reshape(B, T * H * W, n_h * h_d)
            gate = gate_ip * scale_mask
            return result + gate.reshape(B, T, 1, 1, -1) * ip_out.reshape(B, T, H, W, -1)

        blk.forward = patched
    try:
        return model(x, t, ctx)
    finally:
        for hook in hooks:
            hook.remove()
        for blk, (forward, projections) in zip(blocks, saved, strict=True):
            blk.forward = forward
            for proj, module in projections.items():
                setattr(blk.cross_attn, proj, module)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(1, 16, 1, 8, 8, generator=gen)
    ctx = torch.randn(1, 8, 32, generator=gen)
    tokens = torch.randn(1, 5, _TOKEN_DIM, generator=gen)
    return x, torch.tensor([0.6]), ctx, tokens


@torch.no_grad()
def test_the_term_and_its_lora_match_the_reference_node() -> None:
    model = _tiny_dit()
    sd = _tiny_checkpoint()
    x, t, ctx, tokens = _inputs()
    strength = 0.7
    expected = _reference_forward(model, sd, [(tokens, strength)], x, t, ctx)

    adapter = AnimaIPAdapter.from_state_dict(sd)
    extension = AnimaIPAdapterExtension(adapter, [(tokens, strength)], len(model.blocks), torch.float32, _CPU)
    # As the model cache prepares it. The LoRA is patched in after `block_fns` has run, and the IP term's query has
    # to go through the patched `q_proj`.
    apply_custom_layers_to_model(model)
    block_fns = extension.block_fns(model.blocks)
    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=[(adapter.lora_patch(), 1.0)], prefix=ANIMA_LORA_TRANSFORMER_PREFIX, dtype=torch.float32
    ):
        actual = model(x, t, ctx, ip_adapter_blocks=block_fns)

    plain = model(x, t, ctx)
    assert (expected - plain).abs().max() > 1e-2, "the adapter must change the output for this test to mean anything"
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _single_image_term(ip_block, cross_attn, tokens, strength, x, emb) -> torch.Tensor:
    """The one-image block term as it was computed before images could share an adapter, written out."""
    B, L, _ = x.shape
    T = emb.shape[1]
    n_heads, head_dim = cross_attn.n_heads, cross_attn.head_dim
    tok = (tokens.to(dtype=torch.float32) * strength).to(dtype=x.dtype).expand(B, -1, -1)
    q = cross_attn.q_norm(cross_attn.q_proj(x).reshape(B, L, n_heads, head_dim).transpose(1, 2))
    k = ip_block.ip_k_proj(tok).reshape(B, -1, n_heads, head_dim).transpose(1, 2)
    v = ip_block.ip_v_proj(tok).reshape(B, -1, n_heads, head_dim).transpose(1, 2)
    ip = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, L // T, n_heads * head_dim)
    return (ip_block.adaln_ip(emb).unsqueeze(2) * ip).reshape(B, L, -1)


@torch.no_grad()
def test_one_image_is_bitwise_the_single_image_computation() -> None:
    model = _tiny_dit()
    _, _, _, tokens = _inputs()
    adapter = AnimaIPAdapter.from_state_dict(_tiny_checkpoint())
    fns = AnimaIPAdapterExtension(adapter, [(tokens, 0.7)], 2, torch.float32, _CPU).block_fns(model.blocks)
    xq, emb = torch.randn(1, 16, 64), torch.randn(1, 1, 64)

    for index in range(2):
        expected = _single_image_term(adapter.blocks[index], model.blocks[index].cross_attn, tokens, 0.7, xq, emb)
        assert torch.equal(fns[index](xq, emb), expected)  # type: ignore[misc]


@torch.no_grad()
def test_images_sharing_an_adapter_are_attended_together_each_at_its_strength() -> None:
    """Two images of one adapter: one key/value set of both images' tokens, each scaled by its own strength. Not the
    sum of two one-image terms, which would double the gate."""
    model = _tiny_dit()
    sd = _tiny_checkpoint()
    x, t, ctx, tokens = _inputs()
    other = torch.randn(1, 7, _TOKEN_DIM, generator=torch.Generator().manual_seed(3))
    images = [(tokens, 0.7), (other, 1.6)]
    expected = _reference_forward(model, sd, images, x, t, ctx)
    adapter = AnimaIPAdapter.from_state_dict(sd)
    extension = AnimaIPAdapterExtension(adapter, images, len(model.blocks), torch.float32, _CPU)
    apply_custom_layers_to_model(model)

    def run(block_fns) -> torch.Tensor:
        with LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=[(adapter.lora_patch(), 1.0)],
            prefix=ANIMA_LORA_TRANSFORMER_PREFIX,
            dtype=torch.float32,
        ):
            return model(x, t, ctx, ip_adapter_blocks=block_fns)

    torch.testing.assert_close(run(extension.block_fns(model.blocks)), expected, rtol=1e-5, atol=1e-5)
    # Each strength scales its own image: swapping them is a different result.
    swapped = _reference_forward(model, sd, [(tokens, 1.6), (other, 0.7)], x, t, ctx)
    assert (swapped - expected).abs().max() > 1e-3
    summed = combine_block_fns(
        [
            AnimaIPAdapterExtension(adapter, [image], len(model.blocks), torch.float32, _CPU).block_fns(model.blocks)
            for image in images
        ]
    )
    assert (run(summed) - expected).abs().max() > 1e-3, "concatenation must differ from summing for this to mean much"


@torch.no_grad()
def test_a_subset_of_images_attends_only_their_tokens() -> None:
    model = _tiny_dit()
    _, _, _, tokens = _inputs()
    other = torch.randn(1, 7, _TOKEN_DIM, generator=torch.Generator().manual_seed(3))
    adapter = AnimaIPAdapter.from_state_dict(_tiny_checkpoint())
    extension = AnimaIPAdapterExtension(adapter, [(tokens, 0.7), (other, 1.6)], 2, torch.float32, _CPU)
    alone = AnimaIPAdapterExtension(adapter, [(other, 1.6)], 2, torch.float32, _CPU)
    xq, emb = torch.randn(1, 16, 64), torch.randn(1, 1, 64)

    subset = extension.block_fns(model.blocks, [1])

    for index in range(2):
        assert torch.equal(subset[index](xq, emb), alone.block_fns(model.blocks)[index](xq, emb))  # type: ignore[misc]


@torch.no_grad()
def test_no_term_for_any_block_is_bitwise_the_plain_forward() -> None:
    model = _tiny_dit()
    x, t, ctx, _ = _inputs()

    assert combine_block_fns([]) is None
    assert torch.equal(model(x, t, ctx, ip_adapter_blocks=[None, None]), model(x, t, ctx))


@torch.no_grad()
def test_several_adapters_add_their_terms() -> None:
    model = _tiny_dit()
    x, t, ctx, tokens = _inputs()
    adapter = AnimaIPAdapter.from_state_dict(_tiny_checkpoint())
    first, second = (
        AnimaIPAdapterExtension(adapter, [(tokens * scale, 1.0)], 2, torch.float32, _CPU).block_fns(model.blocks)
        for scale in (1.0, -0.5)
    )
    xq = torch.randn(1, 16, 64)
    emb = torch.randn(1, 1, 64)

    combined = combine_block_fns([first, second])

    assert combined is not None
    for index in range(2):
        torch.testing.assert_close(combined[index](xq, emb), first[index](xq, emb) + second[index](xq, emb))


@torch.no_grad()
def test_a_depth_expanded_model_gets_the_term_on_the_blocks_the_adapter_was_trained_on() -> None:
    """Anima-2.9B inserted its 12 new blocks at these positions (`block_layout`); the other 28 are Anima base's,
    in order."""
    new_blocks = {2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36}
    base_block_at = {position: n for n, position in enumerate(sorted(set(range(40)) - new_blocks))}
    blocks = [torch.nn.Module() for _ in range(40)]
    for block in blocks:
        block.cross_attn = CosmosAttention(8, 8, 2, 4)
    # Each adapter block's term is its own index: the values are all ones, and the gate is the block index.
    sd: dict[str, torch.Tensor] = {}
    for n in range(NUM_BLOCKS):
        sd |= {
            f"blocks.{n}.ip_k_proj.weight": torch.zeros(8, 4),
            f"blocks.{n}.ip_k_proj.bias": torch.zeros(8),
            f"blocks.{n}.ip_v_proj.weight": torch.zeros(8, 4),
            f"blocks.{n}.ip_v_proj.bias": torch.ones(8),
            f"blocks.{n}.adaln_ip.1.weight": torch.zeros(8, 8),
            f"blocks.{n}.adaln_ip.1.bias": torch.full((8,), float(n)),
        }
    adapter = AnimaIPAdapter.from_state_dict(sd)
    extension = AnimaIPAdapterExtension(adapter, [(torch.ones(1, 3, 4), 1.0)], 40, torch.float32, _CPU)

    fns = extension.block_fns(blocks)

    assert {i for i, fn in enumerate(fns) if fn is None} == new_blocks
    xq, emb = torch.randn(1, 6, 8), torch.randn(1, 1, 8)
    for position, n in base_block_at.items():
        torch.testing.assert_close(fns[position](xq, emb), torch.full((1, 6, 8), float(n)))  # type: ignore[misc]


def test_the_bundled_lora_is_its_cross_attention_part_on_real_linears() -> None:
    sd = {key: torch.empty(shape, device="meta") for key, shape in state_dict_keys.items()}
    with accelerate.init_empty_weights():
        adapter = AnimaIPAdapter.from_state_dict(sd, metadata)
        transformer = AnimaTransformer(**ANIMA_TRANSFORMER_CONFIG)
    modules = dict(transformer.named_modules())

    layers = adapter.lora_patch().layers

    assert adapter.num_blocks == NUM_BLOCKS
    assert set(layers) == {
        f"{ANIMA_LORA_TRANSFORMER_PREFIX}blocks.{n}.cross_attn.{proj}"
        for n in range(NUM_BLOCKS)
        for proj in ("q_proj", "k_proj", "v_proj", "output_proj")
    }
    for key, layer in layers.items():
        linear = modules[key[len(ANIMA_LORA_TRANSFORMER_PREFIX) :]]
        assert isinstance(linear, torch.nn.Linear)
        assert isinstance(layer, LoRALayer)
        assert (layer.up.shape[0], layer.down.shape[1]) == (linear.out_features, linear.in_features)
        # No alpha in the file: the reference's alpha / rank, 32 / 32.
        assert layer.scale() == 1.0


@pytest.mark.parametrize(
    ("extra_key", "extra_metadata", "message"),
    [
        ("null_tokens", {}, "null_tokens"),
        (None, {"ip_norm_keys": "True"}, "ip_norm_keys"),
        (None, {"ip_inject_before_mlp": "True"}, "ip_inject_before_mlp"),
    ],
)
def test_a_variant_this_adapter_does_not_compute_is_refused(extra_key, extra_metadata, message) -> None:
    sd = _tiny_checkpoint()
    if extra_key is not None:
        sd[extra_key] = torch.zeros(1, 5, _TOKEN_DIM)

    with pytest.raises(ValueError, match=message):
        AnimaIPAdapter.from_state_dict(sd, {**metadata, **extra_metadata})


def test_the_reference_image_is_letterboxed_on_black_and_normalized() -> None:
    image = Image.new("RGB", (200, 100), (255, 0, 0))

    pixels = siglip_pixel_values(image)

    assert pixels.shape == (1, 3, 512, 512)
    # 200x100 scales to 512x256, centred: rows 128-383 hold the image, the rest is black.
    black, red = torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, -1.0, -1.0])
    for row in (0, 127, 384, 511):
        assert torch.equal(pixels[0, :, row, :], black[:, None].expand(3, 512)), row
    for row in (128, 256, 383):
        torch.testing.assert_close(pixels[0, :, row, :], red[:, None].expand(3, 512))


def test_the_shared_encoder_runs_in_float32_without_being_cast() -> None:
    torch.manual_seed(0)
    config = SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16
    )
    encoder = SiglipVisionModel(config).eval().to(torch.bfloat16)
    pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = copy.deepcopy(encoder).float()(pixels).last_hidden_state

    tokens = encode_image_tokens(encoder, pixels, torch.device("cpu"))

    assert tokens.dtype == torch.float32
    torch.testing.assert_close(tokens, expected)
    assert all(p.dtype == torch.bfloat16 for p in encoder.parameters())


def test_the_loaded_adapter_holds_real_weights_the_model_cache_can_move(monkeypatch, tmp_path) -> None:
    """Building the module under `init_empty_weights` and assigning the checkpoint inside it once left every weight on
    the meta device; it only surfaced when the model cache moved the adapter to the GPU."""
    sd = _tiny_checkpoint()
    path = tmp_path / "anima-ip-adapter.safetensors"
    save_file(sd, path, metadata=metadata)
    module = "invokeai.backend.model_manager.load.model_loaders.anima"
    monkeypatch.setattr(f"{module}.TorchDevice.choose_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(f"{module}.TorchDevice.choose_anima_inference_dtype", lambda _d: torch.float32)
    loader = object.__new__(AnimaIPAdapterModelLoader)

    adapter = loader._load_model(IPAdapter_Checkpoint_Anima_Config.model_construct(path=str(path), name="ipa"))

    assert isinstance(adapter, AnimaIPAdapter)
    tensors = {**adapter.state_dict(keep_vars=True), **dict(adapter.named_buffers())}
    assert tensors and not any(t.is_meta for t in tensors.values())
    torch.testing.assert_close(adapter.blocks[1].ip_v_proj.weight, sd["blocks.1.ip_v_proj.weight"])
    layers = adapter.lora_patch().layers.values()
    assert all(isinstance(layer, LoRALayer) for layer in layers)
    lora_tensors = [t for layer in layers for t in (layer.up, layer.down)]  # type: ignore[attr-defined]
    assert len(lora_tensors) == 2 * 2 * 4 and all(t.device.type == "cpu" for t in lora_tensors)
    # What the model cache does with it before a denoise.
    apply_custom_layers_to_model(adapter)
    CachedModelWithPartialLoad(adapter, torch.device("cpu"), keep_ram_copy=True).full_load_to_vram()
