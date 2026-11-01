"""The Qwen-Image Fun control branch, hooked into the real transformer, computes what VideoX-Fun's formulation does.

`_videox_forward` is the recipe written out: the base preamble, the control chain threading its own text stream from
the base's, and an explicit loop over the base blocks adding each hint after its listed layer. It shares nothing with
the hooks beyond the modules' own forwards. Everything runs in float32 on the CPU.
"""

import weakref

import accelerate
import pytest
import torch
from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask

from invokeai.backend.fun_controlnet.hooks import install_fun_control, should_apply
from invokeai.backend.fun_controlnet.qwen_image import QWEN_IMAGE_FUN_CONTROL_IN_DIM, QwenImageFunControlBranch
from tests.backend.model_manager.load.state_dicts.qwen_image_fun_controlnet_keys import state_dict_keys

LATENT_SIDE = 8  # 16 image tokens once packed 2x2
IMAGE_TOKENS = (LATENT_SIDE // 2) ** 2
TEXT_TOKENS = 6
SCALE = 0.85


@pytest.fixture
def inputs(tiny_qwen_image_transformer) -> dict:
    """One forward's arguments, with two padded text tokens so the mask is threaded too."""
    torch.manual_seed(2)
    config = tiny_qwen_image_transformer.config
    mask = torch.ones(1, TEXT_TOKENS, dtype=torch.bool)
    mask[0, -2:] = False
    return {
        "hidden_states": torch.randn(1, IMAGE_TOKENS, config.in_channels),
        "encoder_hidden_states": torch.randn(1, TEXT_TOKENS, config.joint_attention_dim),
        "encoder_hidden_states_mask": mask,
        "timestep": torch.tensor([0.5]),
        "img_shapes": [[(1, LATENT_SIDE // 2, LATENT_SIDE // 2)]],
    }


@pytest.fixture
def control_context() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randn(1, IMAGE_TOKENS, QWEN_IMAGE_FUN_CONTROL_IN_DIM)


def _videox_forward(model, branch, control_context, scale, **inputs) -> tuple[torch.Tensor, list[torch.Tensor]]:
    hidden = model.img_in(inputs["hidden_states"])
    timestep = inputs["timestep"].to(hidden.dtype)
    text = model.txt_in(model.txt_norm(inputs["encoder_hidden_states"]))
    text_seq_len, _, mask = compute_text_seq_len_from_mask(text, inputs["encoder_hidden_states_mask"])
    temb = model.time_text_embed(timestep, hidden)
    rope = model.pos_embed(inputs["img_shapes"], max_txt_seq_len=text_seq_len, device=hidden.device)
    shared = {"encoder_hidden_states_mask": mask, "temb": temb, "image_rotary_emb": rope}

    c = branch.control_img_in(control_context)
    control_text = text
    hints = []
    for index, block in enumerate(branch.control_blocks):
        if index == 0:
            c = block.before_proj(c) + hidden
        control_text, c = block(hidden_states=c, encoder_hidden_states=control_text, **shared)
        hints.append(block.after_proj(c))

    for index, block in enumerate(model.transformer_blocks):
        text, hidden = block(hidden_states=hidden, encoder_hidden_states=text, **shared)
        if index in branch.control_layers:
            hidden = hidden + hints[branch.control_layers.index(index)] * scale
    return model.proj_out(model.norm_out(hidden, temb)), hints


def _forward(model, inputs) -> torch.Tensor:
    return model(**inputs, return_dict=False)[0]


def _hook_count(model) -> int:
    return sum(len(block._forward_hooks) + len(block._forward_pre_hooks) for block in model.transformer_blocks)


@torch.no_grad()
def test_the_hooked_forward_computes_the_videox_formulation(
    tiny_qwen_image_transformer, tiny_branch, inputs, control_context
) -> None:
    model, branch = tiny_qwen_image_transformer, tiny_branch
    expected, _ = _videox_forward(model, branch, control_context, SCALE, **inputs)
    plain = _forward(model, inputs)

    with install_fun_control(model, branch, control_context, scale=SCALE):
        hooked = _forward(model, inputs)

    assert torch.equal(hooked, expected)
    assert not torch.equal(hooked, plain)


@torch.no_grad()
def test_hints_land_on_the_image_stream_of_the_listed_layers_only(
    tiny_qwen_image_transformer, tiny_branch, inputs, control_context
) -> None:
    """Each base block's output is recorded before the hint hook sees it and after: the text stream comes through
    untouched everywhere, and the image stream moves by exactly the scaled hint at the listed layers."""
    model, branch = tiny_qwen_image_transformer, tiny_branch
    _, hints = _videox_forward(model, branch, control_context, SCALE, **inputs)
    before: dict[int, tuple] = {}
    after: dict[int, tuple] = {}
    blocks = list(model.transformer_blocks)
    handles = [
        block.register_forward_hook(lambda _m, _a, out, i=i: before.__setitem__(i, out))
        for i, block in enumerate(blocks)
    ]
    try:
        with install_fun_control(model, branch, control_context, scale=SCALE):
            handles += [
                block.register_forward_hook(lambda _m, _a, out, i=i: after.__setitem__(i, out))
                for i, block in enumerate(blocks)
            ]
            _forward(model, inputs)
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
    tiny_qwen_image_transformer, tiny_branch, inputs, control_context
) -> None:
    model, branch = tiny_qwen_image_transformer, tiny_branch
    plain = _forward(model, inputs)

    with install_fun_control(model, branch, control_context, scale=0.0):
        assert torch.equal(_forward(model, inputs), plain)

    with install_fun_control(model, branch, control_context, scale=SCALE) as hooks:
        hooks.active = False
        assert torch.equal(_forward(model, inputs), plain)
        hooks.active = True
        assert not torch.equal(_forward(model, inputs), plain)
        hooks.active = False
        # Nothing of the active forward is left for the inactive one.
        assert torch.equal(_forward(model, inputs), plain)


@torch.no_grad()
def test_the_projected_context_is_released_before_the_last_control_block(
    tiny_qwen_image_transformer, tiny_branch, inputs, control_context
) -> None:
    """The chain owns its input: the projected context is freed once block 0 has joined it with the image stream, not
    held through every block -- an image-length hidden state more than the hint reservation allows for."""
    projected: list[weakref.ref] = []
    alive_at_last_block: list[bool] = []
    taps = [
        tiny_branch.control_img_in.register_forward_hook(lambda _m, _a, out: projected.append(weakref.ref(out))),
        tiny_branch.control_blocks[-1].register_forward_pre_hook(
            lambda *_args: alive_at_last_block.append(projected[-1]() is not None)
        ),
    ]
    try:
        with install_fun_control(tiny_qwen_image_transformer, tiny_branch, control_context, scale=SCALE):
            _forward(tiny_qwen_image_transformer, inputs)
    finally:
        for tap in taps:
            tap.remove()

    assert alive_at_last_block == [False]


@torch.no_grad()
def test_the_hooks_come_off_on_exit_and_when_the_body_raises(
    tiny_qwen_image_transformer, tiny_branch, inputs, control_context
) -> None:
    model, branch = tiny_qwen_image_transformer, tiny_branch
    plain = _forward(model, inputs)

    with install_fun_control(model, branch, control_context, scale=SCALE):
        assert _hook_count(model) == 1 + len(branch.control_layers)
    assert _hook_count(model) == 0

    with pytest.raises(RuntimeError, match="mid-step"):
        with install_fun_control(model, branch, control_context, scale=SCALE):
            _forward(model, inputs)
            raise RuntimeError("mid-step")
    assert _hook_count(model) == 0
    assert torch.equal(_forward(model, inputs), plain)


def test_a_branch_for_another_transformer_is_refused_at_install(
    tiny_qwen_image_transformer, tiny_branch, control_context
) -> None:
    model = tiny_qwen_image_transformer
    wider = QwenImageFunControlBranch(
        hidden_size=32, num_attention_heads=4, attention_head_dim=8, control_layers=(0, 2)
    )
    deeper = QwenImageFunControlBranch(
        hidden_size=16, num_attention_heads=2, attention_head_dim=8, control_layers=(0, 4)
    )

    with pytest.raises(ValueError, match="32 wide"), install_fun_control(model, wider, control_context, scale=1.0):
        pass
    with pytest.raises(ValueError, match="4 blocks"), install_fun_control(model, deeper, control_context, scale=1.0):
        pass
    with (
        pytest.raises(ValueError, match="132"),
        install_fun_control(model, tiny_branch, control_context[..., :64], scale=1.0),
    ):
        pass
    assert _hook_count(model) == 0


def test_should_apply_counts_steps_inclusively_at_both_ends() -> None:
    assert [should_apply(step, 10, 0.3, 0.8) for step in range(10)] == [False] * 3 + [True] * 6 + [False]
    assert all(should_apply(step, 4, 0.0, 1.0) for step in range(4))


def test_the_branch_names_and_shapes_its_tensors_as_the_checkpoint_does() -> None:
    """Against the header of alibaba-pai's file: 174 tensors, `before_proj` on block 0 only. Built on the meta device,
    so nothing of the 3.5 GB is allocated."""
    with accelerate.init_empty_weights():
        branch = QwenImageFunControlBranch(hidden_size=3072, num_attention_heads=24, attention_head_dim=128)

    assert {name: list(tensor.shape) for name, tensor in branch.state_dict().items()} == {
        name: shape for name, (shape, _dtype) in state_dict_keys.items()
    }


@torch.no_grad()
def test_from_state_dict_rebuilds_the_branch_that_wrote_it(
    tiny_qwen_image_transformer, tiny_branch, inputs, control_context
) -> None:
    model = tiny_qwen_image_transformer
    state_dict = {name: tensor.clone() for name, tensor in tiny_branch.state_dict().items()}

    loaded = QwenImageFunControlBranch.from_state_dict(
        state_dict, dtype=torch.float32, control_layers=(0, 2), attention_head_dim=8
    )

    assert loaded.control_layers == (0, 2)
    assert (loaded.hidden_size, loaded.control_in_dim, loaded.control_blocks[0].attn.heads) == (16, 132, 2)
    assert not loaded.training
    with install_fun_control(model, tiny_branch, control_context, scale=SCALE):
        expected = _forward(model, inputs)
    with install_fun_control(model, loaded, control_context, scale=SCALE):
        assert torch.equal(_forward(model, inputs), expected)


@torch.no_grad()
def test_the_chain_runs_in_the_branch_dtype_and_hands_hints_back_in_the_base_dtype(
    tiny_qwen_image_transformer, tiny_branch, inputs, control_context
) -> None:
    """The loader follows the device's bfloat16-safe dtype while the node runs bfloat16, so a branch may be loaded in
    another dtype than the activations it meets. Everything crossing into the chain is cast to the branch's dtype and
    the hint back, or a modulation Linear would see a foreign dtype. Checked with a float64 branch against the float32
    base: the same arithmetic in a dtype every CPU has a GEMM for."""
    model = tiny_qwen_image_transformer
    with install_fun_control(model, tiny_branch, control_context, scale=SCALE):
        expected = _forward(model, inputs)
    wider = QwenImageFunControlBranch.from_state_dict(
        tiny_branch.state_dict(), dtype=torch.float64, control_layers=(0, 2), attention_head_dim=8
    )

    with install_fun_control(model, wider, control_context, scale=SCALE):
        hooked = _forward(model, inputs)

    assert hooked.dtype == torch.float32
    assert torch.allclose(hooked, expected, atol=1e-4, rtol=1e-4)


def test_from_state_dict_casts_to_the_requested_dtype(tiny_branch) -> None:
    loaded = QwenImageFunControlBranch.from_state_dict(
        tiny_branch.state_dict(), dtype=torch.float16, control_layers=(0, 2), attention_head_dim=8
    )
    assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float16}


def _without(state_dict: dict, prefix: str) -> dict:
    return {name: tensor for name, tensor in state_dict.items() if not name.startswith(prefix)}


@pytest.mark.parametrize(
    ("rewrite", "message"),
    [
        (lambda sd: _without(sd, "control_blocks.1."), "1 control blocks"),
        (lambda sd: {**sd, "control_img_in.weight": torch.zeros(16, 64)}, "takes 64"),
        (lambda sd: _without(sd, "control_img_in."), "control_img_in.weight"),
        (lambda sd: _without(sd, "control_blocks.1.after_proj."), "missing"),
        (lambda sd: {**sd, "control_blocks.1.before_proj.weight": torch.zeros(16, 16)}, "unexpected"),
    ],
    ids=["block-count", "in-dim", "no-projection", "missing-key", "stray-key"],
)
def test_from_state_dict_refuses_what_is_not_the_qwen_image_recipe(tiny_branch, rewrite, message) -> None:
    state_dict = rewrite(dict(tiny_branch.state_dict()))
    with pytest.raises(ValueError, match=message):
        QwenImageFunControlBranch.from_state_dict(
            state_dict, dtype=torch.float32, control_layers=(0, 2), attention_head_dim=8
        )
