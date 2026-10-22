"""Identification of Qwen-Image-2.1 single-file transformers, and of its LoRAs against every other family's.

Qwen-Image-2.1 reuses Qwen-Image's module names almost everywhere, so both directions matter: a 2.1 file must
not be claimed as Qwen-Image, and the quantizations the loader cannot build must be refused at install
(`InvalidMatchError`), not registered as a model that fails at the first render.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import InvalidMatchError, NotAMatchError
from invokeai.backend.model_manager.configs.lora import LoRA_LyCORIS_QwenImage21_Config, LoRA_LyCORIS_QwenImage_Config
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_QwenImage21_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, QwenImage21VariantType

_FIELDS = {
    "hash": "blake3:fakehash",
    "path": "/fake/model.safetensors",
    "file_size": 1000,
    "name": "model",
    "description": "test",
    "source": "test",
    "source_type": "path",
    "key": "test-key",
}


def _transformer(**extra: Any) -> dict[str, Any]:
    """The keys the probe reads, in ComfyUI's layout (prefixed, fused gate_up)."""
    sd: dict[str, Any] = {
        "model.diffusion_model.img_in.weight": torch.zeros(8, 4, dtype=torch.bfloat16),
        "model.diffusion_model.txt_in.text_norm.weight": torch.zeros(8, dtype=torch.bfloat16),
        "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight": torch.zeros(16, 8, dtype=torch.bfloat16),
    }
    sd.update(extra)
    return sd


def _identify(sd: dict[str, Any], path: Path) -> Main_Checkpoint_QwenImage21_Config:
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    with (
        patch("invokeai.backend.model_manager.configs.main.raise_if_not_file"),
        patch("invokeai.backend.model_manager.configs.main.raise_for_override_fields"),
    ):
        return Main_Checkpoint_QwenImage21_Config.from_model_on_disk(mod, dict(_FIELDS))


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("qwen_image_2.1_bf16.safetensors", QwenImage21VariantType.Base),
        ("Qwen-Image-2.1-Turbo_fp8_scaled.safetensors", QwenImage21VariantType.Turbo),
    ],
)
def test_a_single_file_transformer_takes_its_variant_from_the_name(name: str, variant: QwenImage21VariantType) -> None:
    config = _identify(_transformer(), Path(name))
    assert config.base is BaseModelType.QwenImage21
    assert config.variant is variant


def test_qwen_image_is_not_claimed() -> None:
    # Qwen-Image's txt_in is a plain Linear beside a top-level txt_norm, and its MLP is `img_mlp.net`.
    v1 = {
        "img_in.weight": torch.zeros(8, 4),
        "txt_norm.weight": torch.zeros(8),
        "txt_in.weight": torch.zeros(8, 8),
        "transformer_blocks.0.img_mlp.net.0.proj.weight": torch.zeros(16, 8),
    }
    with pytest.raises(NotAMatchError):
        _identify(v1, Path("qwen_image_2512.safetensors"))


def test_an_nvfp4_file_is_refused_at_install() -> None:
    sd = _transformer(**{"model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight_scale_2": torch.ones(())})
    with pytest.raises(InvalidMatchError, match="nvfp4"):
        _identify(sd, Path("qwen_image_2.1_nvfp4.safetensors"))


def test_a_lora_with_norm_diffs_is_qwen_image_2_1(tmp_path: Path) -> None:
    # ComfyUI's Turbo LoRA: besides the Linears, full diffs on the q/k norms and the text norm.
    sd = _comfy_gate_up_lora() | {
        "diffusion_model.transformer_blocks.1.attn.norm_q.diff": torch.zeros(128),
        "diffusion_model.transformer_blocks.1.attn.norm_k.diff": torch.zeros(128),
        "diffusion_model.txt_in.text_norm.diff": torch.zeros(4096),
    }
    assert _probe(LoRA_LyCORIS_QwenImage21_Config, sd, tmp_path)


def test_a_torchao_file_is_refused_at_install() -> None:
    # unsloth's FP8 build: torchao Float8Tensors, saved as the payload and a per-row scale, no `.weight`.
    layer = "transformer_blocks.0.attn.to_q"
    sd = _transformer(
        **{
            f"{layer}._weight_qdata": torch.zeros(8, 8, dtype=torch.float8_e4m3fn),
            f"{layer}._weight_scale": torch.ones(8, 1),
        }
    )
    with pytest.raises(InvalidMatchError, match="torchao"):
        _identify(sd, Path("Qwen-Image-2.1-FP8.safetensors"))


def _int8_gate_up(marker: dict | None) -> dict[str, Any]:
    layer = "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up"
    sd: dict[str, Any] = {
        f"{layer}.weight": torch.zeros(16, 8, dtype=torch.int8),
        f"{layer}.weight_scale": torch.ones(16, 1),
    }
    if marker is not None:
        sd[f"{layer}.comfy_quant"] = torch.frombuffer(bytearray(json.dumps(marker).encode()), dtype=torch.uint8).clone()
    return sd


def test_int8_weights_without_a_convrot_marker_are_refused_at_install(tmp_path: Path) -> None:
    # The markers are read from the file's header, so the file has to exist.
    sd = _transformer(**_int8_gate_up(marker=None))
    path = tmp_path / "qwen_image_2.1_int8_dynamic.safetensors"
    save_file(sd, path)
    with pytest.raises(InvalidMatchError, match="int8"):
        _identify(sd, path)


def test_comfy_int8_convrot_installs(tmp_path: Path) -> None:
    sd = _transformer(**_int8_gate_up(marker={"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}))
    path = tmp_path / "qwen_image_2.1_int8_convrot.safetensors"
    save_file(sd, path)
    assert _identify(sd, path).base is BaseModelType.QwenImage21


def _attention_lora(width: int) -> dict[str, torch.Tensor]:
    """PEFT's common `to_q/to_k/to_v` targets: module names Qwen-Image and Qwen-Image-2.1 share."""
    sd: dict[str, torch.Tensor] = {}
    for block in range(2):
        for projection in ("to_q", "to_k", "to_v"):
            prefix = f"transformer.transformer_blocks.{block}.attn.{projection}"
            sd[f"{prefix}.lora_A.weight"] = torch.zeros(4, width)
            sd[f"{prefix}.lora_B.weight"] = torch.zeros(width, 4)
    return sd


def _probe(config_class, sd: dict[str, torch.Tensor], tmp_path: Path) -> bool:
    path = tmp_path / "lora.safetensors"
    path.touch()
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = sd
    fields = {**_FIELDS, "path": str(path), "source": str(path)}
    try:
        config_class.from_model_on_disk(mod, fields)
    except NotAMatchError:
        return False
    return True


def _comfy_gate_up_lora() -> dict[str, torch.Tensor]:
    """As the published Qwen-Image-2.1 LoRAs are written: ComfyUI's prefix and its fused gated MLP."""
    prefix = "diffusion_model.transformer_blocks.0.img_mlp.gate_up"
    return {f"{prefix}.lora_A.weight": torch.zeros(4, 4096), f"{prefix}.lora_B.weight": torch.zeros(24576, 4)}


def test_an_attention_only_lora_is_told_apart_by_its_width(tmp_path: Path) -> None:
    assert _probe(LoRA_LyCORIS_QwenImage_Config, _attention_lora(3072), tmp_path)
    assert not _probe(LoRA_LyCORIS_QwenImage_Config, _attention_lora(4096), tmp_path)
    assert _probe(LoRA_LyCORIS_QwenImage21_Config, _attention_lora(4096), tmp_path)
    assert not _probe(LoRA_LyCORIS_QwenImage21_Config, _attention_lora(3072), tmp_path)


def test_a_lora_on_the_gated_mlp_is_qwen_image_2_1(tmp_path: Path) -> None:
    kohya = {
        "lora_unet_transformer_blocks_0_img_mlp_gate_up.lora_down.weight": torch.zeros(4, 3072),
        "lora_unet_transformer_blocks_0_img_mlp_gate_up.lora_up.weight": torch.zeros(3072, 4),
    }
    for sd in (kohya, _comfy_gate_up_lora()):
        assert not _probe(LoRA_LyCORIS_QwenImage_Config, sd, tmp_path)
        assert _probe(LoRA_LyCORIS_QwenImage21_Config, sd, tmp_path)


_SHAPES = {"attn.to_out.0": (4096, 4096), "img_mlp.proj": (12288, 4096), "img_mlp.out": (4096, 12288)}


def _lora_on(*modules: str, kohya: bool = False) -> dict[str, torch.Tensor]:
    """A low-rank LoRA on the given block modules only, in diffusers/PEFT or Kohya naming."""
    sd: dict[str, torch.Tensor] = {}
    for module in modules:
        out_features, in_features = _SHAPES[module]
        name = f"transformer_blocks.0.{module}"
        if kohya:
            prefix, down, up = "lora_unet_" + name.replace(".", "_"), "lora_down", "lora_up"
        else:
            prefix, down, up = f"transformer.{name}", "lora_A", "lora_B"
        sd[f"{prefix}.{down}.weight"] = torch.zeros(4, in_features)
        sd[f"{prefix}.{up}.weight"] = torch.zeros(out_features, 4)
    return sd


_OUTPUT_OR_MLP = {
    # No to_q/to_k/to_v and no gate_up: the attention output is told apart by its width, the MLP halves by name.
    "attention-output": _lora_on("attn.to_out.0"),
    "mlp-halves": _lora_on("img_mlp.proj", "img_mlp.out"),
    "mlp-halves-kohya": _lora_on("img_mlp.proj", "img_mlp.out", kohya=True),
    # Each half on its own: `proj` is a name Qwen-Image's MLP has only inside `img_mlp.net.0.proj`.
    "mlp-proj": _lora_on("img_mlp.proj"),
    "mlp-proj-kohya": _lora_on("img_mlp.proj", kohya=True),
    "mlp-out": _lora_on("img_mlp.out"),
}


def _attention_lokr(width: int, split_w2: bool = False) -> dict[str, torch.Tensor]:
    """An attention-only LoKR: the weight is kron(w1, w2), so its input width is 4 x w2's. LyCORIS usually keeps
    w1 whole and stores w2 as a low-rank pair, w2_a @ w2_b."""
    prefix = "transformer.transformer_blocks.0.attn.to_q"
    sd = {f"{prefix}.lokr_w1": torch.zeros(4, 4)}
    if split_w2:
        sd[f"{prefix}.lokr_w2_a"] = torch.zeros(width // 4, 8)
        sd[f"{prefix}.lokr_w2_b"] = torch.zeros(8, width // 4)
    else:
        sd[f"{prefix}.lokr_w2"] = torch.zeros(width // 4, width // 4)
    return sd


@pytest.mark.parametrize("sd", _OUTPUT_OR_MLP.values(), ids=_OUTPUT_OR_MLP.keys())
def test_a_lora_on_the_attention_output_or_the_mlp_halves_is_qwen_image_2_1(
    sd: dict[str, torch.Tensor], tmp_path: Path
) -> None:
    assert not _probe(LoRA_LyCORIS_QwenImage_Config, sd, tmp_path)
    assert _probe(LoRA_LyCORIS_QwenImage21_Config, sd, tmp_path)


@pytest.mark.parametrize("split_w2", [False, True], ids=["whole", "split-w2"])
def test_an_attention_only_lokr_is_told_apart_by_its_width(split_w2: bool, tmp_path: Path) -> None:
    assert _probe(LoRA_LyCORIS_QwenImage_Config, _attention_lokr(3072, split_w2), tmp_path)
    assert not _probe(LoRA_LyCORIS_QwenImage21_Config, _attention_lokr(3072, split_w2), tmp_path)
    assert not _probe(LoRA_LyCORIS_QwenImage_Config, _attention_lokr(4096, split_w2), tmp_path)
    assert _probe(LoRA_LyCORIS_QwenImage21_Config, _attention_lokr(4096, split_w2), tmp_path)
    # A factored width is weaker evidence: beside a module only Qwen-Image has, it stays Qwen-Image's.
    with_qwen_image_mlp = _attention_lokr(4096) | {
        "transformer.transformer_blocks.0.img_mlp.net.0.proj.lokr_w1": torch.zeros(4, 4),
        "transformer.transformer_blocks.0.img_mlp.net.0.proj.lokr_w2": torch.zeros(3072, 768),
    }
    assert _probe(LoRA_LyCORIS_QwenImage_Config, with_qwen_image_mlp, tmp_path)


def _z(*shape: int) -> torch.Tensor:
    """Zeros of the shape, in one element of storage: identification reads shapes, never values."""
    return torch.zeros(()).expand(shape)


def _attention_layers(kind: str, width: int) -> dict[str, torch.Tensor]:
    """A LoRA of one layer kind on the image attention of a model `width` features wide."""
    sd: dict[str, torch.Tensor] = {}
    for projection in ("to_q", "to_k", "to_v"):
        prefix = f"transformer.transformer_blocks.0.attn.{projection}"
        if kind == "loha":
            for factor in ("w1", "w2"):
                sd[f"{prefix}.hada_{factor}_a"] = _z(width, 4)
                sd[f"{prefix}.hada_{factor}_b"] = _z(4, width)
        elif kind == "full":
            sd[f"{prefix}.diff"] = _z(width, width)
        elif kind == "lycoris-dora":
            sd |= {f"{prefix}.lora_A.weight": _z(4, width), f"{prefix}.lora_B.weight": _z(width, 4)}
            sd[f"{prefix}.dora_scale"] = _z(1, width)
    return sd


@pytest.mark.parametrize("kind", ["loha", "full", "lycoris-dora"])
def test_an_attention_only_lora_of_another_kind_is_told_apart_by_its_width(kind: str, tmp_path: Path) -> None:
    # A LoHA states its input width on either `_b` matrix, a full layer as its second dimension.
    assert _probe(LoRA_LyCORIS_QwenImage21_Config, _attention_layers(kind, 4096), tmp_path)
    assert not _probe(LoRA_LyCORIS_QwenImage21_Config, _attention_layers(kind, 3072), tmp_path)


_OTHER_KINDS = {
    "lokr": _attention_lokr(4096),
    "lokr-factored-w1": {
        "transformer.transformer_blocks.0.attn.to_q.lokr_w1_a": _z(4, 2),
        "transformer.transformer_blocks.0.attn.to_q.lokr_w1_b": _z(2, 4),
        "transformer.transformer_blocks.0.attn.to_q.lokr_w2": _z(1024, 1024),
    },
    "loha": _attention_layers("loha", 4096),
    "full": _attention_layers("full", 4096),
    "lycoris-dora": _attention_layers("lycoris-dora", 4096),
    "peft-dora-gate-up": _comfy_gate_up_lora()
    | {"diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_magnitude_vector.weight": _z(24576)},
}


@pytest.mark.parametrize(
    "sd",
    [_comfy_gate_up_lora(), _attention_lora(4096), *_OUTPUT_OR_MLP.values(), *_OTHER_KINDS.values()],
    ids=["gate_up", "attention", *_OUTPUT_OR_MLP.keys(), *_OTHER_KINDS.keys()],
)
def test_a_qwen_image_2_1_lora_is_claimed_by_its_config_alone(sd: dict[str, torch.Tensor], tmp_path: Path) -> None:
    assert _probe(LoRA_LyCORIS_QwenImage21_Config, sd, tmp_path)
    # Identification tries every config; two that accept one file leave the outcome to iteration order.
    others = [
        c
        for c in Config_Base.CONFIG_CLASSES
        if c.__name__.startswith("LoRA_") and c is not LoRA_LyCORIS_QwenImage21_Config
    ]
    assert others
    assert [c.__name__ for c in others if _probe(c, sd, tmp_path)] == []


def _pair(name: str, out_features: int, in_features: int = 4096) -> dict[str, torch.Tensor]:
    return {f"{name}.lora_A.weight": _z(4, in_features), f"{name}.lora_B.weight": _z(out_features, 4)}


@pytest.mark.parametrize(
    "extra",
    [
        # Its text embedder's width.
        _pair("transformer.context_embedder", 4096, 12288),
        # Its double blocks' text stream, which a single-stream Qwen-Image-2.1 block does not have.
        _pair("transformer.transformer_blocks.0.attn.add_q_proj", 4096)
        | _pair("transformer.transformer_blocks.0.attn.to_add_out", 4096),
    ],
    ids=["text-embedder", "text-stream"],
)
def test_a_flux2_klein_9b_lora_is_not_qwen_image_2_1(extra: dict[str, torch.Tensor], tmp_path: Path) -> None:
    # Klein 9B is as wide (4096) and names its double blocks' image attention the same.
    assert not _probe(LoRA_LyCORIS_QwenImage21_Config, _attention_lora(4096) | extra, tmp_path)


def test_a_klein_lora_on_its_double_blocks_image_attention_reads_as_qwen_image_2_1(tmp_path: Path) -> None:
    """A known ambiguity, pinned: a FLUX.2 Klein 9B LoRA on its double blocks' image attention has a 2.1 LoRA's
    names and width. Its `proj_out` and `norm_out.linear` share the names but not the output widths (128 and 8192
    against 64 and 4096): beside that attention they are taken along, on their own they are no 2.1 evidence."""
    output_layers = _pair("transformer.proj_out", 128) | _pair("transformer.norm_out.linear", 8192)
    assert _probe(LoRA_LyCORIS_QwenImage21_Config, _attention_lora(4096) | output_layers, tmp_path)
    assert not _probe(LoRA_LyCORIS_QwenImage21_Config, output_layers, tmp_path)


# Qwen-Image-2.1 LoRAs that apply in full, with the prefix their layout uses.
_WHOLE_LORAS = {
    "comfy-gate-up": (_comfy_gate_up_lora, "diffusion_model."),
    "peft-attention": (lambda: _attention_lora(4096), "transformer."),
    # Its width is a product of factor sizes, the weakest evidence: its module names alone make it 2.1's.
    "lokr-attention": (lambda: _attention_lokr(4096, split_w2=True), "transformer."),
}


def _part_that_does_not_apply(defect: str, whole: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    # Block 2: a module of no whole LoRA, so the part is not merged into one of its layers.
    block_0, block_2 = f"{prefix}transformer_blocks.0", f"{prefix}transformer_blocks.2"
    other_prefix = "transformer." if prefix == "diffusion_model." else "diffusion_model."
    return {
        # A text encoder's layer: no Qwen-Image-2.1 transformer module.
        "foreign-layer": lambda: _pair("text_encoder.layers.0.mlp.fc1", 4096),
        "block-out-of-range": lambda: _pair(f"{prefix}transformer_blocks.40.attn.to_q", 4096),
        # LoKR on the fused gate_up, which cannot be split into gate_layer and proj.
        "lokr-gate-up": lambda: {
            f"{block_2}.img_mlp.gate_up.lokr_w1": _z(4, 4),
            f"{block_2}.img_mlp.gate_up.lokr_w2": _z(6144, 1024),
        },
        "half-pair": lambda: {f"{block_2}.attn.to_q.lora_A.weight": _z(4, 4096)},
        "incomplete-lokr": lambda: {f"{block_2}.attn.to_q.lokr_w1": _z(4, 4)},
        "alpha-only": lambda: {f"{block_2}.attn.to_q.alpha": torch.tensor(4.0)},
        # The whole LoRA's first block once more, under the other prefix.
        "two-prefixes": lambda: {
            other_prefix + key.removeprefix(prefix): value for key, value in whole.items() if key.startswith(block_0)
        },
        # LyCORIS's DoRA on a LoKR layer, which no layer class builds.
        "lokr-dora-scale": lambda: {
            f"{block_2}.attn.to_q.lokr_w1": _z(4, 4),
            f"{block_2}.attn.to_q.lokr_w2": _z(1024, 1024),
            f"{block_2}.attn.to_q.dora_scale": _z(1, 4096),
        },
        # A gate_up beside the gate_layer it is split into.
        "gate-up-and-gate-layer": lambda: _pair(f"{block_0}.img_mlp.gate_up", 24576)
        | _pair(f"{block_0}.img_mlp.gate_layer", 12288),
        "diff-and-pair": lambda: _pair(f"{block_2}.attn.to_q", 4096) | {f"{block_2}.attn.to_q.diff": _z(4096, 4096)},
        "odd-row-gate-up": lambda: _pair(f"{block_2}.img_mlp.gate_up", 24575),
    }[defect]()


_DEFECTS = [
    "foreign-layer",
    "block-out-of-range",
    "lokr-gate-up",
    "half-pair",
    "incomplete-lokr",
    "alpha-only",
    "two-prefixes",
    "lokr-dora-scale",
    "gate-up-and-gate-layer",
    "diff-and-pair",
    "odd-row-gate-up",
]


@pytest.mark.parametrize("defect", _DEFECTS)
@pytest.mark.parametrize("layout", _WHOLE_LORAS)
def test_a_qwen_image_2_1_lora_that_applies_only_in_part_is_claimed_by_no_config(
    layout: str, defect: str, tmp_path: Path
) -> None:
    """Claimed by no LoRA config, so it installs as an unknown model: not as a 2.1 LoRA that fails at the first
    generation, nor as another family's."""
    whole, prefix = _WHOLE_LORAS[layout]
    sd = whole()
    sd |= _part_that_does_not_apply(defect, sd, prefix)
    probes = [c for c in Config_Base.CONFIG_CLASSES if c.__name__.startswith("LoRA_")]
    assert LoRA_LyCORIS_QwenImage21_Config in probes
    assert [c.__name__ for c in probes if _probe(c, sd, tmp_path)] == []
