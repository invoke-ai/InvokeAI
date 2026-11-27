"""The FLUX.2 Fun control branch, hooked into the real transformer, computes what VideoX-Fun's formulation does.

`_videox_forward` is the recipe written out against diffusers' `Flux2Transformer2DModel`: the base preamble, the control
chain threading its own text stream from the base's, and an explicit loop over the double-stream blocks adding each
hint after its listed layer to the whole image stream, reference tokens included. It shares nothing with the hooks
beyond the modules' own forwards. Everything runs in float32 on the CPU.
"""

import pytest
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

from invokeai.backend.fun_controlnet.flux2 import (
    FLUX2_FUN_CONTROL_IN_DIM,
    FLUX2_FUN_CONTROL_LAYERS,
    Flux2FunControlBranch,
)
from invokeai.backend.fun_controlnet.hooks import install_fun_control
from tests.backend.model_manager.load.state_dicts.flux2_fun_controlnet_keys import state_dict_keys

HIDDEN = 16  # two heads of 8
HEAD_DIM = 8
TEXT_DIM = 12
GENERATED_TOKENS = 16  # a 4x4 packed grid
REFERENCE_TOKENS = 4  # one 2x2 reference image
TEXT_TOKENS = 6
SCALE = 0.75


def _randomized(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Every parameter from N(0, 0.1). diffusers and VideoX-Fun zero-initialise projections (VideoX-Fun `after_proj`
    and `before_proj` included), which would leave a hint that is zero and a parity test that cannot fail."""
    torch.manual_seed(seed)
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    return module.eval()


@pytest.fixture
def transformer() -> Flux2Transformer2DModel:
    """The real FLUX.2 transformer at toy width: eight double-stream blocks, so the real layer map (0, 2, 4, 6) fits."""
    model = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=8,
        num_layers=8,
        num_single_layers=2,
        attention_head_dim=HEAD_DIM,
        num_attention_heads=HIDDEN // HEAD_DIM,
        joint_attention_dim=TEXT_DIM,
        timestep_guidance_channels=16,
        axes_dims_rope=(2, 2, 2, 2),
    )
    return _randomized(model, seed=0)


def _branch(control_layers=FLUX2_FUN_CONTROL_LAYERS) -> Flux2FunControlBranch:
    branch = Flux2FunControlBranch(
        hidden_size=HIDDEN,
        num_attention_heads=HIDDEN // HEAD_DIM,
        attention_head_dim=HEAD_DIM,
        control_layers=control_layers,
    )
    return _randomized(branch, seed=1)


@pytest.fixture
def branch() -> Flux2FunControlBranch:
    return _branch()


def _ids(t: int, side: int) -> torch.Tensor:
    ids = torch.zeros(side, side, 4, dtype=torch.long)
    ids[..., 0] = t
    ids[..., 1] = torch.arange(side)[:, None]
    ids[..., 2] = torch.arange(side)[None, :]
    return ids.reshape(1, side * side, 4)


@pytest.fixture
def inputs(transformer) -> dict:
    """One forward's arguments with a reference image: its tokens follow the generated ones, at T=10, as
    `Flux2RefImageExtension` lays them out."""
    torch.manual_seed(2)
    text_ids = torch.zeros(1, TEXT_TOKENS, 4, dtype=torch.long)
    text_ids[..., 3] = torch.arange(TEXT_TOKENS)
    return {
        "hidden_states": torch.randn(1, GENERATED_TOKENS + REFERENCE_TOKENS, transformer.config.in_channels),
        "encoder_hidden_states": torch.randn(1, TEXT_TOKENS, TEXT_DIM),
        "timestep": torch.tensor([0.5]),
        "img_ids": torch.cat([_ids(0, 4), _ids(10, 2)], dim=1),
        "txt_ids": text_ids,
        "guidance": torch.tensor([3.5]),
    }


@pytest.fixture
def control_context() -> torch.Tensor:
    """The generated tokens' context, zero-padded for the reference tokens as VideoX-Fun pads it."""
    torch.manual_seed(3)
    generated = torch.randn(1, GENERATED_TOKENS, FLUX2_FUN_CONTROL_IN_DIM)
    return torch.cat([generated, torch.zeros(1, REFERENCE_TOKENS, FLUX2_FUN_CONTROL_IN_DIM)], dim=1)


def _videox_forward(model, branch, control_context, scale, **inputs) -> tuple[torch.Tensor, list[torch.Tensor]]:
    timestep = inputs["timestep"] * 1000
    guidance = inputs["guidance"] * 1000
    temb = model.time_guidance_embed(timestep, guidance)
    shared = {
        "temb_mod_img": model.double_stream_modulation_img(temb),
        "temb_mod_txt": model.double_stream_modulation_txt(temb),
    }
    hidden = model.x_embedder(inputs["hidden_states"])
    text = model.context_embedder(inputs["encoder_hidden_states"])
    image_rope = model.pos_embed(inputs["img_ids"][0])
    text_rope = model.pos_embed(inputs["txt_ids"][0])
    rope = (torch.cat([text_rope[0], image_rope[0]]), torch.cat([text_rope[1], image_rope[1]]))
    shared["image_rotary_emb"] = rope

    # `Flux2ControlTransformer2DModel.forward_control`: block 0 adds the embedded image stream, every block threads
    # the text stream it was handed by the previous one, and each block's `after_proj` is a hint.
    c = branch.control_img_in(control_context)
    control_text = text
    hints = []
    for index, block in enumerate(branch.control_blocks):
        if index == 0:
            c = block.before_proj(c) + hidden
        control_text, c = block(hidden_states=c, encoder_hidden_states=control_text, **shared)
        hints.append(block.after_proj(c))

    # `BaseFlux2TransformerBlock.forward`: the hint goes onto the block's image-stream output, once.
    for index, block in enumerate(model.transformer_blocks):
        text, hidden = block(hidden_states=hidden, encoder_hidden_states=text, **shared)
        if index in branch.control_layers:
            hidden = hidden + hints[branch.control_layers.index(index)] * scale

    joint = torch.cat([text, hidden], dim=1)
    single_mod = model.single_stream_modulation(temb)
    for block in model.single_transformer_blocks:
        joint = block(hidden_states=joint, encoder_hidden_states=None, temb_mod=single_mod, image_rotary_emb=rope)
    return model.proj_out(model.norm_out(joint[:, TEXT_TOKENS:], temb)), hints


def _forward(model, inputs, **extra) -> torch.Tensor:
    return model(**inputs, **extra, return_dict=False)[0]


def _hook_count(model) -> int:
    return sum(len(block._forward_hooks) + len(block._forward_pre_hooks) for block in model.transformer_blocks)


@torch.no_grad()
def test_the_hooked_forward_computes_the_videox_formulation_with_a_reference_image(
    transformer, branch, inputs, control_context
) -> None:
    expected, hints = _videox_forward(transformer, branch, control_context, SCALE, **inputs)
    plain = _forward(transformer, inputs)

    with install_fun_control(transformer, branch, control_context, scale=SCALE):
        hooked = _forward(transformer, inputs)

    assert torch.equal(hooked, expected)
    assert not torch.equal(hooked, plain)
    # The context is zero for the reference tokens, their hints are not: the stream they are added to carries the
    # reference image, so leaving those rows out would be a different model.
    assert all(hint[:, GENERATED_TOKENS:].abs().sum() > 0 for hint in hints)


@torch.no_grad()
def test_a_branch_on_the_wrong_layers_no_longer_matches(transformer, branch, inputs, control_context) -> None:
    """The parity above can tell where the hints land: the same weights after the odd base blocks compute something
    else, so a layer map that drifted from (0, 2, 4, 6) would fail it."""
    expected, _ = _videox_forward(transformer, branch, control_context, SCALE, **inputs)
    misplaced = Flux2FunControlBranch.from_state_dict(
        branch.state_dict(), dtype=torch.float32, control_layers=(1, 3, 5, 7), attention_head_dim=HEAD_DIM
    )

    with install_fun_control(transformer, misplaced, control_context, scale=SCALE):
        assert not torch.allclose(_forward(transformer, inputs), expected)


@torch.no_grad()
def test_hints_land_on_the_image_stream_of_the_listed_layers_only(transformer, branch, inputs, control_context) -> None:
    """Each double-stream block's output is recorded before the hint hook sees it and after: the text stream comes
    through untouched everywhere, and the image stream moves by exactly the scaled hint at the listed layers."""
    _, hints = _videox_forward(transformer, branch, control_context, SCALE, **inputs)
    before: dict[int, tuple] = {}
    after: dict[int, tuple] = {}
    blocks = list(transformer.transformer_blocks)
    handles = [
        block.register_forward_hook(lambda _m, _a, out, i=i: before.__setitem__(i, out))
        for i, block in enumerate(blocks)
    ]
    try:
        with install_fun_control(transformer, branch, control_context, scale=SCALE):
            handles += [
                block.register_forward_hook(lambda _m, _a, out, i=i: after.__setitem__(i, out))
                for i, block in enumerate(blocks)
            ]
            _forward(transformer, inputs)
    finally:
        for handle in handles:
            handle.remove()

    for index in range(len(blocks)):
        text_before, image_before = before[index]
        text_after, image_after = after[index]
        assert text_after is text_before
        if index in branch.control_layers:
            assert torch.equal(image_after, image_before + hints[branch.control_layers.index(index)] * SCALE)
        else:
            assert image_after is image_before


@torch.no_grad()
def test_scale_zero_and_an_inactive_handle_leave_the_forward_untouched(
    transformer, branch, inputs, control_context
) -> None:
    plain = _forward(transformer, inputs)

    with install_fun_control(transformer, branch, control_context, scale=0.0):
        assert torch.equal(_forward(transformer, inputs), plain)

    with install_fun_control(transformer, branch, control_context, scale=SCALE) as hooks:
        hooks.active = False
        assert torch.equal(_forward(transformer, inputs), plain)
        hooks.active = True
        assert not torch.equal(_forward(transformer, inputs), plain)
        hooks.active = False
        assert torch.equal(_forward(transformer, inputs), plain)


@torch.no_grad()
def test_the_hooks_come_off_on_exit_and_when_the_body_raises(transformer, branch, inputs, control_context) -> None:
    plain = _forward(transformer, inputs)

    with install_fun_control(transformer, branch, control_context, scale=SCALE):
        assert _hook_count(transformer) == 1 + len(branch.control_layers)
    assert _hook_count(transformer) == 0

    with pytest.raises(RuntimeError, match="mid-step"):
        with install_fun_control(transformer, branch, control_context, scale=SCALE):
            _forward(transformer, inputs)
            raise RuntimeError("mid-step")
    assert _hook_count(transformer) == 0
    assert torch.equal(_forward(transformer, inputs), plain)


@torch.no_grad()
def test_the_branch_refuses_to_run_beside_the_reference_kv_cache(transformer, branch, inputs, control_context) -> None:
    """diffusers' FLUX.2 can cache the reference tokens' keys and values per base block. The control blocks would read
    and write those layers, so the forward stops at block 0 instead -- and the hooks still come off."""
    with pytest.raises(ValueError, match="KV cache"):
        with install_fun_control(transformer, branch, control_context, scale=SCALE):
            _forward(transformer, inputs, kv_cache_mode="extract", num_ref_tokens=REFERENCE_TOKENS)
    assert _hook_count(transformer) == 0


@torch.no_grad()
def test_the_chain_runs_in_the_branch_dtype_and_hands_hints_back_in_the_base_dtype(
    transformer, branch, inputs, control_context
) -> None:
    """Everything crossing into the chain -- the two modulations, the text stream, the image stream -- is cast to the
    branch's dtype and the hint back. Checked with a float64 branch against the float32 base: the same arithmetic in a
    dtype every CPU has a GEMM for."""
    with install_fun_control(transformer, branch, control_context, scale=SCALE):
        expected = _forward(transformer, inputs)
    wider = Flux2FunControlBranch.from_state_dict(branch.state_dict(), dtype=torch.float64, attention_head_dim=HEAD_DIM)

    with install_fun_control(transformer, wider, control_context, scale=SCALE):
        hooked = _forward(transformer, inputs)

    assert hooked.dtype == torch.float32
    assert torch.allclose(hooked, expected, atol=1e-4, rtol=1e-4)


def _as_released(state_dict: dict) -> dict:
    """The tensors under the names alibaba-pai's file uses for the blocks."""
    return {name.replace("control_blocks.", "control_transformer_blocks.", 1): t for name, t in state_dict.items()}


@torch.no_grad()
def test_from_state_dict_rebuilds_the_branch_from_the_released_names(
    transformer, branch, inputs, control_context
) -> None:
    loaded = Flux2FunControlBranch.from_state_dict(
        _as_released(branch.state_dict()), dtype=torch.float32, attention_head_dim=HEAD_DIM
    )

    assert loaded.control_layers == FLUX2_FUN_CONTROL_LAYERS
    assert not loaded.training
    with install_fun_control(transformer, branch, control_context, scale=SCALE):
        expected = _forward(transformer, inputs)
    with install_fun_control(transformer, loaded, control_context, scale=SCALE):
        assert torch.equal(_forward(transformer, inputs), expected)


def test_the_released_layout_reads_as_the_dev_geometry() -> None:
    """Against the header of alibaba-pai's file, on the meta device so nothing of the 8.2 GB is allocated: the width,
    head count and MLP width come off the tensors, and every one of the 76 tensors finds its parameter."""
    released = {
        name: torch.empty(shape, dtype=torch.bfloat16, device="meta")
        for name, (shape, _dtype) in state_dict_keys.items()
    }

    branch = Flux2FunControlBranch.from_state_dict(released, dtype=torch.bfloat16)

    first = branch.control_blocks[0]
    assert (branch.hidden_size, branch.control_in_dim, first.attn.heads, first.mlp_hidden_dim) == (6144, 260, 48, 18432)
    assert branch.control_layers == FLUX2_FUN_CONTROL_LAYERS
    assert {name: list(t.shape) for name, t in branch.state_dict().items()} == {
        name.replace("control_transformer_blocks.", "control_blocks."): shape
        for name, (shape, _dtype) in state_dict_keys.items()
    }


def _without(state_dict: dict, prefix: str) -> dict:
    return {name: tensor for name, tensor in state_dict.items() if not name.startswith(prefix)}


@pytest.mark.parametrize(
    ("rewrite", "message"),
    [
        (lambda sd: _without(sd, "control_transformer_blocks.3."), "3 control blocks"),
        (lambda sd: {**sd, "control_img_in.weight": torch.zeros(HIDDEN, 132)}, "takes 132"),
        (lambda sd: _without(sd, "control_img_in."), "control_img_in.weight"),
        (lambda sd: _without(sd, "control_transformer_blocks.0.ff.linear_in."), "feed-forward"),
        (lambda sd: _without(sd, "control_transformer_blocks.2.after_proj."), "missing"),
        (
            lambda sd: {**sd, "control_transformer_blocks.1.before_proj.weight": torch.zeros(HIDDEN, HIDDEN)},
            "unexpected",
        ),
    ],
    ids=["block-count", "qwen-in-dim", "no-projection", "no-feed-forward", "missing-key", "stray-key"],
)
def test_from_state_dict_refuses_what_is_not_the_flux2_recipe(branch, rewrite, message) -> None:
    state_dict = rewrite(_as_released(branch.state_dict()))
    with pytest.raises(ValueError, match=message):
        Flux2FunControlBranch.from_state_dict(state_dict, dtype=torch.float32, attention_head_dim=HEAD_DIM)


def test_a_klein_width_transformer_is_refused_at_install(branch, control_context) -> None:
    """The dev branch is 6144 wide and Klein 3072 or 4096; here the same mismatch at toy widths."""
    narrower = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=8,
        num_layers=8,
        num_single_layers=1,
        attention_head_dim=HEAD_DIM,
        num_attention_heads=1,
        joint_attention_dim=TEXT_DIM,
        timestep_guidance_channels=16,
        axes_dims_rope=(2, 2, 2, 2),
    )

    with pytest.raises(ValueError, match="16 wide"), install_fun_control(narrower, branch, control_context, scale=1.0):
        pass
    assert _hook_count(narrower) == 0
