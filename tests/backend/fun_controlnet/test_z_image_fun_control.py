"""The Z-Image Fun control, hooked into the real transformer, computes what VideoX-Fun's control transformer does.

`_videox_forward` restates VideoX-Fun's `ZImageControlTransformer2DModel.forward` at batch size 1 against diffusers'
`ZImageTransformer2DModel`: the base preamble, the control context patched, embedded and padded with the base's
`x_pad_token` (`patchify`), the refiner chain with its hints after the base refiner blocks (`forward_control_2_0_refiner`)
or the plain control refiner (`forward_control_1_0`), the main chain over `[control | caption]`
(`forward_control_2_0_layers`), and an explicit loop over the base blocks adding each hint after its listed layer. The
control blocks run in VideoX-Fun's own stacked formulation. Its masks are all-True at batch size 1; the restatement
passes `None`, as diffusers does. Where the hints land, which chain refines the control and how many context channels
it reads come from `VIDEOX_CONFIGS`, VideoX-Fun's four config files restated, never from the implementation's variants;
the restatement shares only the modules' own forwards with the hooks. Everything runs in float32 on the CPU, every
parameter drawn at random, the zero-initialised projections and the pad tokens included.
"""

import weakref

import accelerate
import pytest
import torch
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel, ZImageTransformerBlock

from invokeai.backend.fun_controlnet.hooks import install_fun_control
from invokeai.backend.fun_controlnet.z_image import (
    Z_IMAGE_FUN_CONTROL_1_0,
    Z_IMAGE_FUN_CONTROL_2_0,
    Z_IMAGE_FUN_CONTROL_2_1,
    Z_IMAGE_FUN_CONTROL_2_1_LITE,
    ZImageFunControl,
    ZImageFunControlVariant,
)
from invokeai.backend.z_image.z_image_transformer_patch import patch_transformer_for_regional_prompting
from tests.backend.model_manager.load.state_dicts.z_image_fun_controlnet_1_0_keys import (
    state_dict_keys as keys_1_0,
)
from tests.backend.model_manager.load.state_dicts.z_image_fun_controlnet_2_1_keys import (
    state_dict_keys as keys_2_1,
)
from tests.backend.model_manager.load.state_dicts.z_image_fun_controlnet_2_1_lite_keys import (
    state_dict_keys as keys_2_1_lite,
)

HIDDEN = 64  # two heads of 32
HEADS = 2
HEAD_DIM = HIDDEN // HEADS
CAPTION_DIM = 32
LATENT_SIDE = 10  # 5x5 = 25 image tokens: the transformer pads them to 32 with seven `x_pad_token` rows
IMAGE_TOKENS = (LATENT_SIDE // 2) ** 2
CAPTION_TOKENS = 20  # padded to 32 with `cap_pad_token`
SCALE = 0.75

VARIANTS = [Z_IMAGE_FUN_CONTROL_1_0, Z_IMAGE_FUN_CONTROL_2_0, Z_IMAGE_FUN_CONTROL_2_1, Z_IMAGE_FUN_CONTROL_2_1_LITE]

# VideoX-Fun `config/z_image/z_image_control.yaml`, `_2.0.yaml`, `_2.1.yaml` and `_2.1_lite.yaml`, as written there.
# Keys a file leaves out take `ZImageControlTransformer2DModel`'s defaults: no control refiner, and refiner hints after
# base refiner blocks `range(n_refiner_layers)`.
VIDEOX_CONFIGS: dict[str, dict] = {
    "1.0": {"control_layers_places": [0, 5, 10, 15, 20, 25], "control_in_dim": 16},
    "2.0": {
        "control_layers_places": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        "control_refiner_layers_places": [0, 1],
        "add_control_noise_refiner": True,
        "add_control_noise_refiner_correctly": False,
        "control_in_dim": 33,
    },
    "2.1": {
        "control_layers_places": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        "control_refiner_layers_places": [0, 1],
        "add_control_noise_refiner": True,
        "add_control_noise_refiner_correctly": True,
        "control_in_dim": 33,
    },
    "2.1 Lite": {
        "control_layers_places": [0, 10, 20],
        "control_refiner_layers_places": [0, 1],
        "add_control_noise_refiner": True,
        "add_control_noise_refiner_correctly": True,
        "control_in_dim": 33,
    },
}
REFINER_BLOCKS = 2


def _config(variant: ZImageFunControlVariant) -> dict:
    config = {
        "control_refiner_layers_places": list(range(REFINER_BLOCKS)),
        "add_control_noise_refiner": False,
        "add_control_noise_refiner_correctly": False,
    }
    return config | VIDEOX_CONFIGS[variant.name]


def _randomized(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Every parameter from N(0, 0.1). diffusers zero-initialises the pad tokens and VideoX-Fun the projections, which
    would leave a pad row or a hint that is zero and a parity test that cannot fail."""
    torch.manual_seed(seed)
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    return module.eval()


@pytest.fixture(scope="module")
def transformer() -> ZImageTransformer2DModel:
    """The real Z-Image transformer at toy width: 30 main blocks and two refiner blocks, so every variant's real layer
    map fits."""
    model = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=HIDDEN,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=HEADS,
        n_kv_heads=HEADS,
        norm_eps=1e-05,
        qk_norm=True,
        cap_feat_dim=CAPTION_DIM,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[8, 12, 12],
        axes_lens=[128, 32, 32],
    )
    return _randomized(model, seed=0)


def _control(variant: ZImageFunControlVariant, seed: int = 1) -> ZImageFunControl:
    return _randomized(ZImageFunControl(variant, hidden_size=HIDDEN, num_attention_heads=HEADS), seed)


@pytest.fixture
def inputs(transformer) -> dict:
    torch.manual_seed(2)
    return {
        "x": [torch.randn(16, 1, LATENT_SIDE, LATENT_SIDE)],
        "t": torch.tensor([0.4]),
        "cap_feats": [torch.randn(CAPTION_TOKENS, CAPTION_DIM)],
    }


@pytest.fixture
def context_4d() -> torch.Tensor:
    """A control context `[control | keep mask | masked image]`, every channel non-zero so a dropped one shows."""
    torch.manual_seed(3)
    return torch.randn(1, 33, LATENT_SIDE, LATENT_SIDE)


def _forward(model, inputs) -> torch.Tensor:
    # Indexed rather than `return_dict=False`, which regional prompting's patched forward does not take.
    return model(**inputs)[0][0]


def _hook_count(model) -> int:
    blocks = [*model.noise_refiner, *model.layers]
    return sum(len(block._forward_hooks) + len(block._forward_pre_hooks) for block in blocks)


def _videox_control_blocks(blocks, c, x, **kwargs) -> tuple[list[torch.Tensor], torch.Tensor]:
    """`ZImageControlTransformerBlock.forward` as VideoX-Fun writes it: every block stacks its hint and its stream
    onto what it was given, and the next one unbinds them again."""
    for index, block in enumerate(blocks):
        if index == 0:
            c = block.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = ZImageTransformerBlock.forward(block, c, **kwargs)
        all_c += [block.after_proj(c), c]
        c = torch.stack(all_c)
    *hints, c = torch.unbind(c)
    return hints, c


def _videox_patchify(context: torch.Tensor, pad_to: int) -> tuple[torch.Tensor, torch.Tensor]:
    """VideoX-Fun's `patchify` of a `(C, H, W)` context: 2x2 patches in `(pH, pW, C)` order, padded with copies of the
    last token, and the mask of those padding rows."""
    channels, height, width = context.shape
    tokens = context.view(channels, height // 2, 2, width // 2, 2).permute(1, 3, 2, 4, 0).reshape(-1, 4 * channels)
    padding = pad_to - len(tokens)
    padded = torch.cat([tokens, tokens[-1:].repeat(padding, 1)])
    return padded, torch.cat([torch.zeros(len(tokens), dtype=torch.bool), torch.ones(padding, dtype=torch.bool)])


def _videox_forward(model, control, config, context_4d, scale, *, x, t, cap_feats, drop_refiner_hints=False):
    places, refiner_places = config["control_layers_places"], config["control_refiner_layers_places"]
    adaln = model.t_embedder(t * model.t_scale)
    (xs, caps, sizes, x_pos, cap_pos, x_pad, cap_pad) = model.patchify_and_embed(x, cap_feats, 2, 1)

    hidden = model.all_x_embedder["2-1"](xs[0])
    hidden[x_pad[0]] = model.x_pad_token
    hidden = hidden[None]
    x_freqs = model.rope_embedder(x_pos[0])[None]

    # `patchify` + embed: the control context's padding rows are the base's `x_pad_token`.
    tokens, pad_mask = _videox_patchify(context_4d[0, : config["control_in_dim"]], hidden.shape[1])
    c = control.control_all_x_embedder["2-1"](tokens)
    c[pad_mask] = model.x_pad_token
    c = c[None]

    refiner_hints = None
    if config["add_control_noise_refiner"]:
        # `forward_control_2_0_refiner`: `add_control_noise_refiner_correctly` picks the refiner blocks (2.1) or the
        # main control blocks (2.0) for the chain over the image stream.
        correctly = config["add_control_noise_refiner_correctly"]
        chain = control.control_noise_refiner if correctly else control.control_layers
        refiner_hints, c = _videox_control_blocks(
            chain, c, hidden, attn_mask=None, freqs_cis=x_freqs, adaln_input=adaln
        )
    for index, block in enumerate(model.noise_refiner):
        hidden = block(hidden, None, x_freqs, adaln)
        if refiner_hints is not None and index in refiner_places and not drop_refiner_hints:
            hidden = hidden + refiner_hints[refiner_places.index(index)] * scale

    caption = model.cap_embedder(caps[0])
    caption[cap_pad[0]] = model.cap_pad_token
    caption = caption[None]
    # diffusers numbers the caption's padding rows on from its tokens and appends position-0 ids it then cuts off.
    cap_freqs = model.rope_embedder(cap_pos[0])[None, : caption.shape[1]]
    for block in model.context_refiner:
        caption = block(caption, None, cap_freqs)

    if not config["add_control_noise_refiner"]:
        # `forward_control_1_0`: plain refiner blocks over the control context alone.
        for block in control.control_noise_refiner:
            c = block(c, None, x_freqs, adaln)

    unified = torch.cat([hidden, caption], dim=1)
    unified_freqs = torch.cat([x_freqs, cap_freqs], dim=1)
    hints, _ = _videox_control_blocks(
        control.control_layers,
        torch.cat([c, caption], dim=1),
        unified,
        attn_mask=None,
        freqs_cis=unified_freqs,
        adaln_input=adaln,
    )
    for index, block in enumerate(model.layers):
        unified = block(unified, None, unified_freqs, adaln)
        if index in places:
            unified = unified + hints[places.index(index)] * scale
    unified = model.all_final_layer["2-1"](unified, adaln)
    return model.unpatchify(list(unified.unbind(dim=0)), sizes, 2, 1)[0]


@torch.no_grad()
@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_the_hooked_forward_computes_the_videox_formulation(transformer, inputs, context_4d, variant) -> None:
    control = _control(variant)
    expected = _videox_forward(transformer, control, _config(variant), context_4d, SCALE, **inputs)
    plain = _forward(transformer, inputs)

    with install_fun_control(transformer, control, control.pack(context_4d), scale=SCALE):
        hooked = _forward(transformer, inputs)

    assert torch.equal(hooked, expected)
    assert not torch.allclose(hooked, plain)


@torch.no_grad()
def test_both_guidance_passes_take_the_same_context(transformer, inputs, context_4d) -> None:
    """The positive and the negative pass of a step run one after the other with the hooks installed once; each must
    match its own restatement, with nothing of the first carried into the second."""
    control = _control(Z_IMAGE_FUN_CONTROL_2_1)
    # A longer caption: the unified stream, and with it the control chain's, is longer on this pass.
    negative = {**inputs, "cap_feats": [torch.randn(CAPTION_TOKENS + 20, CAPTION_DIM)]}

    with install_fun_control(transformer, control, control.pack(context_4d), scale=SCALE):
        positive_out = _forward(transformer, inputs)
        negative_out = _forward(transformer, negative)

    config = _config(Z_IMAGE_FUN_CONTROL_2_1)
    assert torch.equal(positive_out, _videox_forward(transformer, control, config, context_4d, SCALE, **inputs))
    assert torch.equal(negative_out, _videox_forward(transformer, control, config, context_4d, SCALE, **negative))


@torch.no_grad()
def test_the_parity_can_tell_where_the_hints_land(transformer, inputs, context_4d) -> None:
    """The comparison above is not vacuous: the Lite weights placed after base layers 0, 2 and 4 -- the map Invoke used
    for every build before -- compute something else, and so does a forward without the refiner hints."""
    control = _control(Z_IMAGE_FUN_CONTROL_2_1_LITE)
    misplaced = ZImageFunControl(
        ZImageFunControlVariant("misplaced", 33, (0, 2, 4), "refiner"), hidden_size=HIDDEN, num_attention_heads=HEADS
    ).eval()
    misplaced.load_state_dict(control.state_dict())
    config = _config(Z_IMAGE_FUN_CONTROL_2_1_LITE)
    expected = _videox_forward(transformer, control, config, context_4d, SCALE, **inputs)

    with install_fun_control(transformer, misplaced, misplaced.pack(context_4d), scale=SCALE):
        assert not torch.allclose(_forward(transformer, inputs), expected)
    with install_fun_control(transformer, control, control.pack(context_4d), scale=SCALE):
        hooked = _forward(transformer, inputs)
    assert not torch.allclose(
        hooked, _videox_forward(transformer, control, config, context_4d, SCALE, drop_refiner_hints=True, **inputs)
    )


@torch.no_grad()
@pytest.mark.parametrize("variant", [Z_IMAGE_FUN_CONTROL_1_0, Z_IMAGE_FUN_CONTROL_2_1], ids=lambda v: v.name)
def test_hints_land_on_the_listed_blocks_only(transformer, inputs, context_4d, variant) -> None:
    """Each base block's output is recorded before the hint hook sees it and after: it moves at the listed refiner and
    main blocks and nowhere else, and 1.0 leaves the refiner untouched."""
    control = _control(variant)
    blocks = {f"refiner.{i}": b for i, b in enumerate(transformer.noise_refiner)}
    blocks.update({f"layer.{i}": b for i, b in enumerate(transformer.layers)})
    before: dict[str, torch.Tensor] = {}
    after: dict[str, torch.Tensor] = {}
    handles = [
        block.register_forward_hook(lambda _m, _a, out, name=name: before.__setitem__(name, out))
        for name, block in blocks.items()
    ]
    try:
        with install_fun_control(transformer, control, control.pack(context_4d), scale=SCALE):
            handles += [
                block.register_forward_hook(lambda _m, _a, out, name=name: after.__setitem__(name, out))
                for name, block in blocks.items()
            ]
            _forward(transformer, inputs)
    finally:
        for handle in handles:
            handle.remove()

    config = _config(variant)
    refiner_places = config["control_refiner_layers_places"] if config["add_control_noise_refiner"] else []
    hinted = {f"refiner.{i}" for i in refiner_places} | {f"layer.{i}" for i in config["control_layers_places"]}
    moved = {name for name in blocks if after[name] is not before[name]}
    assert moved == hinted
    assert all(not torch.equal(after[name], before[name]) for name in hinted)


@torch.no_grad()
def test_scale_zero_and_an_inactive_handle_leave_the_forward_untouched(transformer, inputs, context_4d) -> None:
    control = _control(Z_IMAGE_FUN_CONTROL_2_1)
    context = control.pack(context_4d)
    plain = _forward(transformer, inputs)

    with install_fun_control(transformer, control, context, scale=0.0):
        assert torch.equal(_forward(transformer, inputs), plain)

    with install_fun_control(transformer, control, context, scale=SCALE) as hooks:
        hooks.active = False
        assert torch.equal(_forward(transformer, inputs), plain)
        hooks.active = True
        assert not torch.equal(_forward(transformer, inputs), plain)
        hooks.active = False
        # Nothing of the active forward is left for the inactive one.
        assert torch.equal(_forward(transformer, inputs), plain)


@torch.no_grad()
@pytest.mark.parametrize(
    ("variant", "expected_hooks"),
    # Two entry pre-hooks, then one post-hook per hinted block: the refiner pair (none for 1.0) and the main layers.
    [
        (Z_IMAGE_FUN_CONTROL_1_0, 2 + 6),
        (Z_IMAGE_FUN_CONTROL_2_1, 2 + 2 + 15),
        (Z_IMAGE_FUN_CONTROL_2_1_LITE, 2 + 2 + 3),
    ],
    ids=lambda v: getattr(v, "name", str(v)),
)
def test_the_hooks_come_off_on_exit_and_when_the_body_raises(
    transformer, inputs, context_4d, variant, expected_hooks
) -> None:
    control = _control(variant)
    context = control.pack(context_4d)
    plain = _forward(transformer, inputs)

    with install_fun_control(transformer, control, context, scale=SCALE):
        assert _hook_count(transformer) == expected_hooks
    assert _hook_count(transformer) == 0

    with pytest.raises(RuntimeError, match="mid-step"):
        with install_fun_control(transformer, control, context, scale=SCALE):
            _forward(transformer, inputs)
            raise RuntimeError("mid-step")
    assert _hook_count(transformer) == 0
    assert torch.equal(_forward(transformer, inputs), plain)


@torch.no_grad()
@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_the_chain_inputs_are_released_before_the_last_control_block(transformer, inputs, context_4d, variant) -> None:
    """Each chain owns its input, and the stream the refiner stage hands on is held only until the main stage takes
    it: by the last control block the embedded context, the handed-on stream and the main chain's joined input are all
    freed. Any of them still alive is an image-length hidden state more than the hint reservation allows for."""
    control = _control(variant)
    refs: dict[str, weakref.ref] = {}
    alive_at_last_block: list[dict[str, bool]] = []
    # The block whose output the refiner stage hands on: 2.0 runs its refiner chain over the main control blocks.
    config = _config(variant)
    refined_by_main_blocks = config["add_control_noise_refiner"] and not config["add_control_noise_refiner_correctly"]
    refiner_last = control.control_layers[-1] if refined_by_main_blocks else control.control_noise_refiner[-1]

    def remember(name: str, *, first_call_only: bool = False):
        # Returns nothing: a forward hook's return value would replace the block's output.
        def hook(_module, _args, output) -> None:
            if not (first_call_only and name in refs):
                refs[name] = weakref.ref(output)

        return hook

    def remember_joined(_module, args) -> None:
        refs["joined"] = weakref.ref(args[0])

    def check(*_args) -> None:
        alive_at_last_block.append({name: ref() is not None for name, ref in refs.items()})

    taps = [
        control.control_all_x_embedder["2-1"].register_forward_hook(remember("embedded")),
        # The first call is the refiner stage's; 2.0 calls the same block again in the main stage.
        refiner_last.register_forward_hook(remember("handed on", first_call_only=True)),
        # The last call is the main stage's: block 0's projection takes the joined stream.
        control.control_layers[0].before_proj.register_forward_pre_hook(remember_joined),
        control.control_layers[-1].register_forward_pre_hook(check),
    ]
    try:
        with install_fun_control(transformer, control, control.pack(context_4d), scale=SCALE):
            _forward(transformer, inputs)
    finally:
        for tap in taps:
            tap.remove()

    assert alive_at_last_block[-1] == {"embedded": False, "handed on": False, "joined": False}


@torch.no_grad()
def test_a_main_stage_without_its_refiner_stage_is_refused(transformer, inputs, context_4d) -> None:
    """The main chain starts from the control stream the refiner stage refined. A forward that reaches `layers[0]`
    without having run `noise_refiner[0]` with the control active has none, and must not run on a stale one."""
    control = _control(Z_IMAGE_FUN_CONTROL_2_1)

    with install_fun_control(transformer, control, control.pack(context_4d), scale=SCALE) as hooks:
        hooks.active = False
        # Switched on after the refiner stage's entry has passed, inactive, on its way.
        tap = transformer.noise_refiner[0].register_forward_pre_hook(lambda *_args: setattr(hooks, "active", True))
        try:
            with pytest.raises(RuntimeError, match="stage 1 was entered without"):
                _forward(transformer, inputs)
        finally:
            tap.remove()


@torch.no_grad()
def test_a_control_context_of_another_size_is_refused(transformer, inputs) -> None:
    """Encoded at another size, the control tokens do not line up with the image stream's."""
    control = _control(Z_IMAGE_FUN_CONTROL_2_1)
    other_size = control.pack(torch.randn(1, 33, LATENT_SIDE + 4, LATENT_SIDE))

    with pytest.raises(ValueError, match="35 image tokens"):
        with install_fun_control(transformer, control, other_size, scale=SCALE):
            _forward(transformer, inputs)
    assert _hook_count(transformer) == 0


def test_a_control_for_another_transformer_is_refused_at_install(transformer, context_4d) -> None:
    wider = ZImageFunControl(Z_IMAGE_FUN_CONTROL_2_1, hidden_size=128, num_attention_heads=4)
    control = _control(Z_IMAGE_FUN_CONTROL_2_1)

    with (
        pytest.raises(ValueError, match="128 wide"),
        install_fun_control(transformer, wider, control.pack(context_4d), scale=1.0),
    ):
        pass
    with (
        pytest.raises(ValueError, match="132"),
        install_fun_control(transformer, control, control.pack(context_4d)[..., :64], scale=1.0),
    ):
        pass
    assert _hook_count(transformer) == 0


@torch.no_grad()
def test_the_control_blocks_take_the_regional_bias_on_the_positive_pass_only(transformer, inputs, context_4d) -> None:
    """The control blocks take what the base block took: the regional attention bias the patched forward hands
    `layers[0]` on the positive pass, nothing on the negative pass, and nothing in the refiner stage, which regional
    prompting does not mask."""
    control = _control(Z_IMAGE_FUN_CONTROL_2_1)
    cap = inputs["cap_feats"][0]
    negative = {**inputs, "cap_feats": [torch.randn(CAPTION_TOKENS, CAPTION_DIM)]}
    side = IMAGE_TOKENS + CAPTION_TOKENS
    restricting = torch.ones(side, side, dtype=torch.float16)
    restricting[:IMAGE_TOKENS, IMAGE_TOKENS:] = 0
    restricting[IMAGE_TOKENS:, :IMAGE_TOKENS] = 0
    masks: dict[str, list] = {"refiner": [], "main": [], "base": []}
    taps = [
        control.control_noise_refiner[0].register_forward_pre_hook(
            lambda _m, _args, kwargs: masks["refiner"].append(kwargs["attn_mask"]), with_kwargs=True
        ),
        control.control_layers[0].register_forward_pre_hook(
            lambda _m, _args, kwargs: masks["main"].append(kwargs["attn_mask"]), with_kwargs=True
        ),
        transformer.layers[0].register_forward_pre_hook(lambda _m, args: masks["base"].append(args[1])),
    ]
    try:
        with install_fun_control(transformer, control, control.pack(context_4d), scale=SCALE):
            plain = _forward(transformer, inputs)
            with patch_transformer_for_regional_prompting(
                transformer, restricting, IMAGE_TOKENS, positive_cap_feats=cap
            ):
                positive = _forward(transformer, inputs)
                _forward(transformer, negative)
    finally:
        for tap in taps:
            tap.remove()

    assert not torch.allclose(positive, plain, atol=1e-3)
    assert masks["refiner"] == [None, None, None]
    assert masks["base"][0] is None and masks["base"][2] is None
    assert masks["base"][1] is not None
    assert torch.equal(masks["main"][1], masks["base"][1])
    assert masks["main"][0] is None and masks["main"][2] is None


@torch.no_grad()
def test_an_unrestricted_regional_mask_leaves_the_controlled_forward_as_it_was(transformer) -> None:
    """With nothing to restrict and no padding the regional bias is all zeros, so the patched forward computes the
    plain one's control. Unpadded on purpose: the bias also keeps real tokens from attending to padding rows, which the
    plain forward does not."""
    control = _control(Z_IMAGE_FUN_CONTROL_2_1)
    torch.manual_seed(4)
    side, caption_tokens = 16, 32  # 64 image tokens and a 32-token caption, both multiples of 32
    inputs = {
        "x": [torch.randn(16, 1, side, side)],
        "t": torch.tensor([0.4]),
        "cap_feats": [torch.randn(caption_tokens, CAPTION_DIM)],
    }
    image_tokens = (side // 2) ** 2
    unrestricted = torch.ones(image_tokens + caption_tokens, image_tokens + caption_tokens, dtype=torch.float16)

    with install_fun_control(transformer, control, control.pack(torch.randn(1, 33, side, side)), scale=SCALE):
        plain = _forward(transformer, inputs)
        with patch_transformer_for_regional_prompting(
            transformer, unrestricted, image_tokens, positive_cap_feats=inputs["cap_feats"][0]
        ):
            regional = _forward(transformer, inputs)

    torch.testing.assert_close(regional, plain, rtol=1e-4, atol=1e-5)


def test_pack_patches_the_context_as_the_transformer_patches_its_latents(transformer, context_4d) -> None:
    for variant in (Z_IMAGE_FUN_CONTROL_1_0, Z_IMAGE_FUN_CONTROL_2_1):
        control = ZImageFunControl(variant, hidden_size=HIDDEN, num_attention_heads=HEADS)
        channels = _config(variant)["control_in_dim"]
        expected, _, _ = transformer._patchify_image(context_4d[0, :channels].unsqueeze(1), 2, 1)

        assert torch.equal(control.pack(context_4d)[0], expected)


def _as_meta(keys: dict) -> dict[str, torch.Tensor]:
    return {name: torch.empty(shape, dtype=torch.bfloat16, device="meta") for name, (shape, _dtype) in keys.items()}


@pytest.mark.parametrize(
    ("variant", "keys"),
    [
        (Z_IMAGE_FUN_CONTROL_1_0, keys_1_0),
        (Z_IMAGE_FUN_CONTROL_2_1, keys_2_1),
        (Z_IMAGE_FUN_CONTROL_2_1_LITE, keys_2_1_lite),
        # The 2.0 file has the 2.1 keys; its refiner blocks were never trained and are not built.
        (
            Z_IMAGE_FUN_CONTROL_2_0,
            {name: value for name, value in keys_2_1.items() if not name.startswith("control_noise_refiner.")},
        ),
    ],
    ids=["1.0", "2.1", "2.1 Lite", "2.0"],
)
def test_each_variant_names_and_shapes_its_tensors_as_the_released_file_does(variant, keys) -> None:
    """Against the headers of alibaba-pai's files, on the meta device so nothing of the 6.7 GB is allocated."""
    with accelerate.init_empty_weights():
        control = ZImageFunControl(variant, hidden_size=3840, num_attention_heads=30)

    assert {name: list(tensor.shape) for name, tensor in control.state_dict().items()} == {
        name: shape for name, (shape, _dtype) in keys.items()
    }


def test_the_1_0_header_reads_as_its_variant_at_full_width() -> None:
    """1.0 is the one layout the loader can tell apart without reading a value, so its header goes through the whole
    `from_state_dict` on the meta device."""
    control = ZImageFunControl.from_state_dict(_as_meta(keys_1_0), dtype=torch.bfloat16)

    assert control.variant == Z_IMAGE_FUN_CONTROL_1_0
    assert (control.hidden_size, control.control_in_dim, control.control_layers[0].attention.heads) == (3840, 64, 30)


def _state_dict(variant: ZImageFunControlVariant, seed: int = 1) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in _control(variant, seed).state_dict().items()}


def _with_untrained_refiner(sd: dict) -> dict:
    return {
        name: torch.zeros_like(tensor) if name.startswith("control_noise_refiner.") and "_proj." in name else tensor
        for name, tensor in sd.items()
    }


@torch.no_grad()
@pytest.mark.parametrize(
    ("released", "variant"),
    [
        (lambda: _state_dict(Z_IMAGE_FUN_CONTROL_1_0), Z_IMAGE_FUN_CONTROL_1_0),
        (lambda: _with_untrained_refiner(_state_dict(Z_IMAGE_FUN_CONTROL_2_1)), Z_IMAGE_FUN_CONTROL_2_0),
        (lambda: _state_dict(Z_IMAGE_FUN_CONTROL_2_1), Z_IMAGE_FUN_CONTROL_2_1),
        (lambda: _state_dict(Z_IMAGE_FUN_CONTROL_2_1_LITE), Z_IMAGE_FUN_CONTROL_2_1_LITE),
    ],
    ids=["1.0", "2.0", "2.1", "2.1 Lite"],
)
def test_from_state_dict_reads_the_variant_off_the_tensors(transformer, inputs, context_4d, released, variant) -> None:
    """2.0 and 2.1 have the same keys; 2.0's refiner projections are all zero, and it runs its refiner chain over the
    main control blocks. Each loaded control computes its variant's VideoX-Fun forward."""
    sd = released()

    loaded = ZImageFunControl.from_state_dict(sd, dtype=torch.float32, attention_head_dim=HEAD_DIM)

    assert loaded.variant == variant
    assert not loaded.training
    assert len(loaded.control_noise_refiner) == (0 if variant is Z_IMAGE_FUN_CONTROL_2_0 else 2)
    with install_fun_control(transformer, loaded, loaded.pack(context_4d), scale=SCALE):
        hooked = _forward(transformer, inputs)
    assert torch.equal(hooked, _videox_forward(transformer, loaded, _config(variant), context_4d, SCALE, **inputs))


def test_from_state_dict_casts_to_the_requested_dtype() -> None:
    loaded = ZImageFunControl.from_state_dict(
        _state_dict(Z_IMAGE_FUN_CONTROL_2_1_LITE), dtype=torch.float16, attention_head_dim=HEAD_DIM
    )
    assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float16}


@torch.no_grad()
def test_the_chain_runs_in_the_control_dtype_and_hands_hints_back_in_the_base_dtype(
    transformer, inputs, context_4d
) -> None:
    """The loader follows the device's bfloat16-safe dtype while the node runs bfloat16, so the control may meet
    activations of another dtype. Checked with a float64 control against the float32 base: the same arithmetic in a
    dtype every CPU has a GEMM for."""
    control = _control(Z_IMAGE_FUN_CONTROL_2_1)
    with install_fun_control(transformer, control, control.pack(context_4d), scale=SCALE):
        expected = _forward(transformer, inputs)
    wider = ZImageFunControl.from_state_dict(control.state_dict(), dtype=torch.float64, attention_head_dim=HEAD_DIM)

    with install_fun_control(transformer, wider, wider.pack(context_4d), scale=SCALE):
        hooked = _forward(transformer, inputs)

    assert hooked.dtype == torch.float32
    assert torch.allclose(hooked, expected, atol=1e-4, rtol=1e-4)


def _without(sd: dict, prefix: str) -> dict:
    return {name: tensor for name, tensor in sd.items() if not name.startswith(prefix)}


@pytest.mark.parametrize(
    ("rewrite", "message"),
    [
        (lambda: _without(_state_dict(Z_IMAGE_FUN_CONTROL_2_1), "control_all_x_embedder."), "is missing"),
        (lambda: _without(_state_dict(Z_IMAGE_FUN_CONTROL_2_1), "control_layers.14."), "14 control blocks"),
        (
            lambda: _without(_state_dict(Z_IMAGE_FUN_CONTROL_2_1), "control_noise_refiner.1.after_proj."),
            "incomplete refiner projections",
        ),
        (
            lambda: _with_untrained_refiner(_state_dict(Z_IMAGE_FUN_CONTROL_2_1_LITE)),
            "3 control blocks and untrained",
        ),
        (
            lambda: {
                **_state_dict(Z_IMAGE_FUN_CONTROL_1_0),
                "control_all_x_embedder.2-1.weight": torch.zeros(HIDDEN, 132),
            },
            "takes 132 features, with 6 control blocks",
        ),
        (
            lambda: _without(_state_dict(Z_IMAGE_FUN_CONTROL_2_1), "control_layers.3.feed_forward."),
            "missing",
        ),
        (
            lambda: {**_state_dict(Z_IMAGE_FUN_CONTROL_2_1), "x_pad_token": torch.zeros(1, HIDDEN)},
            "unexpected",
        ),
    ],
    ids=[
        "no-embedder",
        "block-count",
        "partial-projections",
        "untrained-lite",
        "1.0-with-33-channels",
        "missing",
        "stray",
    ],
)
def test_from_state_dict_refuses_what_is_not_a_published_layout(rewrite, message) -> None:
    with pytest.raises(ValueError, match=message):
        ZImageFunControl.from_state_dict(rewrite(), dtype=torch.float32, attention_head_dim=HEAD_DIM)


def test_a_width_the_heads_do_not_divide_is_refused() -> None:
    with pytest.raises(ValueError, match="128-wide attention heads"):
        ZImageFunControl.from_state_dict(_state_dict(Z_IMAGE_FUN_CONTROL_2_1), dtype=torch.float32)
