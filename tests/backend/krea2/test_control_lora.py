"""Krea-2 Control LoRA: the widened input projection as a hook on `img_in`, and the checkpoint conversion.

The numerics are pinned against the reference pipeline's model surgery (`ControlInputLayer` replacing the input
projection, fed `cat([x, c])`), restated here on a deep copy of the model.
"""

import copy

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel

from invokeai.backend.krea2.control_lora import Krea2ControlLoRAExtension, krea2_control_lora_from_state_dict
from invokeai.backend.krea2.reference_edit import forward_with_references
from invokeai.backend.krea2.sampling_utils import prepare_position_ids, prepare_position_ids_with_references
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.lora_conversions.krea2_lora_constants import KREA2_LORA_TRANSFORMER_PREFIX
from invokeai.backend.patches.lora_conversions.krea2_lora_conversion_utils import lora_model_from_krea2_state_dict

TEXT_LEN = 3
GRID = (2, 3)
TARGET_LEN = GRID[0] * GRID[1]
HIDDEN = 16


def _tiny_transformer() -> Krea2Transformer2DModel:
    torch.manual_seed(0)
    model = Krea2Transformer2DModel(
        in_channels=64,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        timestep_embed_dim=8,
        text_hidden_dim=8,
        num_text_layers=3,
        text_num_attention_heads=2,
        text_num_key_value_heads=2,
        text_intermediate_size=16,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    # diffusers zero-initializes the modulation tables; random weights everywhere keep every path live.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape) * 0.3)
    return model


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return {
        "x": torch.randn(1, TARGET_LEN, 64, generator=generator),
        "control": torch.randn(1, TARGET_LEN, 64, generator=generator),
        "text": torch.randn(1, TEXT_LEN, 3, 8, generator=generator),
        "references": torch.randn(1, 4, 64, generator=generator),
        "weight": torch.randn(HIDDEN, 128, generator=generator) * 0.2,
        "bias": torch.randn(HIDDEN, generator=generator),
    }


def _forward(model: torch.nn.Module, hidden_states: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    return model(
        hidden_states=hidden_states,
        encoder_hidden_states=text,
        timestep=torch.tensor([0.6]),
        position_ids=prepare_position_ids(TEXT_LEN, *GRID, torch.device("cpu")),
        return_dict=False,
    )[0]


@torch.no_grad()
def test_an_armed_forward_is_the_reference_widened_projection_and_an_unarmed_one_is_stock() -> None:
    model = _tiny_transformer()
    inputs = _inputs()
    stock = _forward(model, inputs["x"], inputs["text"])
    surgery = copy.deepcopy(model)
    surgery.img_in = torch.nn.Linear(128, HIDDEN)
    surgery.img_in.weight.copy_(inputs["weight"])
    surgery.img_in.bias.copy_(inputs["bias"])
    expected = _forward(surgery, torch.cat([inputs["x"], inputs["control"]], dim=-1), inputs["text"])
    extension = Krea2ControlLoRAExtension(inputs["weight"], inputs["bias"], inputs["control"])

    with extension.install(model):
        unarmed = _forward(model, inputs["x"], inputs["text"])
        with extension.armed():
            armed = _forward(model, inputs["x"], inputs["text"])

    assert torch.equal(unarmed, stock)
    torch.testing.assert_close(armed, expected)
    assert not torch.allclose(armed, stock)


@torch.no_grad()
def test_only_the_target_tokens_are_reprojected_next_to_joint_references() -> None:
    model = _tiny_transformer()
    inputs = _inputs()
    stock_references = model.img_in(inputs["references"])
    extension = Krea2ControlLoRAExtension(inputs["weight"], inputs["bias"], inputs["control"])
    projected: list[torch.Tensor] = []

    with extension.install(model):
        # Registered after the extension's hook, so it sees what the blocks receive.
        recorder = model.img_in.register_forward_hook(lambda _m, _a, output: projected.append(output))
        with extension.armed():
            forward_with_references(
                model,
                inputs["x"],
                inputs["text"],
                torch.tensor([0.6]),
                prepare_position_ids_with_references(TEXT_LEN, *GRID, [(2, 2)], torch.device("cpu")),
                inputs["references"],
            )
        recorder.remove()

    expected_target = F.linear(torch.cat([inputs["x"], inputs["control"]], dim=-1), inputs["weight"], inputs["bias"])
    torch.testing.assert_close(projected[0][:, :TARGET_LEN], expected_target)
    torch.testing.assert_close(projected[0][:, TARGET_LEN:], stock_references)


@torch.no_grad()
def test_the_hook_comes_off_the_transformer_when_the_forward_raises() -> None:
    model = _tiny_transformer()
    inputs = _inputs()
    stock = _forward(model, inputs["x"], inputs["text"])
    extension = Krea2ControlLoRAExtension(inputs["weight"], inputs["bias"], inputs["control"])

    with pytest.raises(RuntimeError, match="cancelled"):
        with extension.install(model), extension.armed():
            raise RuntimeError("cancelled")

    with extension.armed():
        assert torch.equal(_forward(model, inputs["x"], inputs["text"]), stock)


@pytest.mark.parametrize(
    ("control", "message"),
    [
        (torch.zeros(1, TARGET_LEN, 32), "takes 128 features"),
        (torch.zeros(1, TARGET_LEN + 1, 64), "only 6 image tokens"),
    ],
)
@torch.no_grad()
def test_control_tokens_that_do_not_fit_the_input_are_refused(control: torch.Tensor, message: str) -> None:
    model = _tiny_transformer()
    inputs = _inputs()
    extension = Krea2ControlLoRAExtension(inputs["weight"], inputs["bias"], control)

    with extension.install(model), extension.armed(), pytest.raises(ValueError, match=message):
        _forward(model, inputs["x"], inputs["text"])


def test_the_checkpoint_converts_to_its_native_krea2_lora_plus_the_projection() -> None:
    generator = torch.Generator().manual_seed(2)
    halves = {
        f"blocks.{block}.{module}.{half}": torch.randn(shape, generator=generator)
        for block in (0, 1)
        for module in ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate", "mlp.gate", "mlp.up", "mlp.down")
        for half, shape in (("A", (4, 16)), ("B", (16, 4)))
    }
    projection = {"first.weight": torch.randn(16, 128, generator=generator), "first.bias": torch.randn(16)}
    native = lora_model_from_krea2_state_dict(
        {f"{key[:-2]}.lora_{key[-1]}.weight": value for key, value in halves.items()}, alpha=None
    )

    patch = krea2_control_lora_from_state_dict({**halves, **projection})

    assert patch.layers.keys() == native.layers.keys()
    assert len(patch.layers) == 16
    # Every layer lands on a Linear of the real transformer, not on a name that maps onto nothing.
    transformer = _tiny_transformer()
    for key in patch.layers:
        assert isinstance(transformer.get_submodule(key.removeprefix(KREA2_LORA_TRANSFORMER_PREFIX)), torch.nn.Linear)
    for key, layer in patch.layers.items():
        assert isinstance(layer, LoRALayer)
        assert torch.equal(layer.up, native.layers[key].up) and torch.equal(layer.down, native.layers[key].down)
        assert layer.scale() == 1.0
    assert torch.equal(patch.input_projection_weight, projection["first.weight"])
    assert torch.equal(patch.input_projection_bias, projection["first.bias"])
    # The model cache bills and moves the projection with the block layers.
    assert patch.calc_size() == native.calc_size() + (16 * 128 + 16) * 4
    patch.to(dtype=torch.float16)
    assert patch.input_projection_weight.dtype == patch.input_projection_bias.dtype == torch.float16
