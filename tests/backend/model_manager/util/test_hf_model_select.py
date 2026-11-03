from pathlib import Path
from typing import List

import pytest

from invokeai.backend.model_manager.taxonomy import ModelRepoVariant
from invokeai.backend.model_manager.util.select_hf_files import filter_files


# This is the full list of model paths returned by the HF API for sdxl-base
@pytest.fixture
def sdxl_base_files() -> List[Path]:
    return [
        Path(x)
        for x in [
            ".gitattributes",
            "01.png",
            "LICENSE.md",
            "README.md",
            "comparison.png",
            "model_index.json",
            "pipeline.png",
            "scheduler/scheduler_config.json",
            "sd_xl_base_1.0.safetensors",
            "sd_xl_base_1.0_0.9vae.safetensors",
            "sd_xl_offset_example-lora_1.0.safetensors",
            "text_encoder/config.json",
            "text_encoder/flax_model.msgpack",
            "text_encoder/model.fp16.safetensors",
            "text_encoder/model.onnx",
            "text_encoder/model.safetensors",
            "text_encoder/openvino_model.bin",
            "text_encoder/openvino_model.xml",
            "text_encoder_2/config.json",
            "text_encoder_2/flax_model.msgpack",
            "text_encoder_2/model.fp16.safetensors",
            "text_encoder_2/model.onnx",
            "text_encoder_2/model.onnx_data",
            "text_encoder_2/model.safetensors",
            "text_encoder_2/openvino_model.bin",
            "text_encoder_2/openvino_model.xml",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "tokenizer_2/merges.txt",
            "tokenizer_2/special_tokens_map.json",
            "tokenizer_2/tokenizer_config.json",
            "tokenizer_2/vocab.json",
            "unet/config.json",
            "unet/diffusion_flax_model.msgpack",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/model.onnx",
            "unet/model.onnx_data",
            "unet/openvino_model.bin",
            "unet/openvino_model.xml",
            "vae/config.json",
            "vae/diffusion_flax_model.msgpack",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "vae_1_0/config.json",
            "vae_1_0/diffusion_pytorch_model.fp16.safetensors",
            "vae_1_0/diffusion_pytorch_model.safetensors",
            "vae_decoder/config.json",
            "vae_decoder/model.onnx",
            "vae_decoder/openvino_model.bin",
            "vae_decoder/openvino_model.xml",
            "vae_encoder/config.json",
            "vae_encoder/model.onnx",
            "vae_encoder/openvino_model.bin",
            "vae_encoder/openvino_model.xml",
        ]
    ]


# This are what we expect to get when various diffusers variants are requested
@pytest.mark.parametrize(
    "variant,expected_list",
    [
        (
            None,
            [
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "text_encoder_2/config.json",
                "text_encoder_2/model.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "tokenizer_2/merges.txt",
                "tokenizer_2/special_tokens_map.json",
                "tokenizer_2/tokenizer_config.json",
                "tokenizer_2/vocab.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
                "vae_1_0/config.json",
                "vae_1_0/diffusion_pytorch_model.safetensors",
            ],
        ),
        (
            ModelRepoVariant.Default,
            [
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "text_encoder_2/config.json",
                "text_encoder_2/model.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "tokenizer_2/merges.txt",
                "tokenizer_2/special_tokens_map.json",
                "tokenizer_2/tokenizer_config.json",
                "tokenizer_2/vocab.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
                "vae_1_0/config.json",
                "vae_1_0/diffusion_pytorch_model.safetensors",
            ],
        ),
        (
            ModelRepoVariant.OpenVINO,
            [
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/openvino_model.bin",
                "text_encoder/openvino_model.xml",
                "text_encoder_2/config.json",
                "text_encoder_2/openvino_model.bin",
                "text_encoder_2/openvino_model.xml",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "tokenizer_2/merges.txt",
                "tokenizer_2/special_tokens_map.json",
                "tokenizer_2/tokenizer_config.json",
                "tokenizer_2/vocab.json",
                "unet/config.json",
                "unet/openvino_model.bin",
                "unet/openvino_model.xml",
                "vae_decoder/config.json",
                "vae_decoder/openvino_model.bin",
                "vae_decoder/openvino_model.xml",
                "vae_encoder/config.json",
                "vae_encoder/openvino_model.bin",
                "vae_encoder/openvino_model.xml",
            ],
        ),
        (
            ModelRepoVariant.FP16,
            [
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.fp16.safetensors",
                "text_encoder_2/config.json",
                "text_encoder_2/model.fp16.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "tokenizer_2/merges.txt",
                "tokenizer_2/special_tokens_map.json",
                "tokenizer_2/tokenizer_config.json",
                "tokenizer_2/vocab.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.fp16.safetensors",
                "vae_1_0/config.json",
                "vae_1_0/diffusion_pytorch_model.fp16.safetensors",
            ],
        ),
        (
            ModelRepoVariant.ONNX,
            [
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.onnx",
                "text_encoder_2/config.json",
                "text_encoder_2/model.onnx",
                "text_encoder_2/model.onnx_data",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "tokenizer_2/merges.txt",
                "tokenizer_2/special_tokens_map.json",
                "tokenizer_2/tokenizer_config.json",
                "tokenizer_2/vocab.json",
                "unet/config.json",
                "unet/model.onnx",
                "unet/model.onnx_data",
                "vae_decoder/config.json",
                "vae_decoder/model.onnx",
                "vae_encoder/config.json",
                "vae_encoder/model.onnx",
            ],
        ),
        (
            ModelRepoVariant.Flax,
            [
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/flax_model.msgpack",
                "text_encoder_2/config.json",
                "text_encoder_2/flax_model.msgpack",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "tokenizer_2/merges.txt",
                "tokenizer_2/special_tokens_map.json",
                "tokenizer_2/tokenizer_config.json",
                "tokenizer_2/vocab.json",
                "unet/config.json",
                "unet/diffusion_flax_model.msgpack",
                "vae/config.json",
                "vae/diffusion_flax_model.msgpack",
            ],
        ),
    ],
)
def test_select(sdxl_base_files: List[Path], variant: ModelRepoVariant, expected_list: List[str]) -> None:
    print(f"testing variant {variant}")
    filtered_files = filter_files(sdxl_base_files, variant)
    assert set(filtered_files) == {Path(x) for x in expected_list}


@pytest.fixture
def sd15_test_files() -> list[Path]:
    return [
        Path(f)
        for f in [
            "feature_extractor/preprocessor_config.json",
            "safety_checker/config.json",
            "safety_checker/model.fp16.safetensors",
            "safety_checker/model.safetensors",
            "safety_checker/pytorch_model.bin",
            "safety_checker/pytorch_model.fp16.bin",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "text_encoder/model.safetensors",
            "text_encoder/pytorch_model.bin",
            "text_encoder/pytorch_model.fp16.bin",
            "tokenizer/merges.txt",
            "tokenizer/special_tokens_map.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.bin",
            "unet/diffusion_pytorch_model.fp16.bin",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/diffusion_pytorch_model.non_ema.bin",
            "unet/diffusion_pytorch_model.non_ema.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.fp16.bin",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ]
    ]


@pytest.mark.parametrize(
    "variant,expected_files",
    [
        (
            ModelRepoVariant.FP16,
            [
                "feature_extractor/preprocessor_config.json",
                "safety_checker/config.json",
                "safety_checker/model.fp16.safetensors",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.fp16.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.fp16.safetensors",
            ],
        ),
        (
            ModelRepoVariant.FP32,
            [
                "feature_extractor/preprocessor_config.json",
                "safety_checker/config.json",
                "safety_checker/model.safetensors",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
            ],
        ),
    ],
)
def test_select_multiple_weights(
    sd15_test_files: list[Path], variant: ModelRepoVariant, expected_files: list[str]
) -> None:
    filtered_files = filter_files(sd15_test_files, variant)
    assert set(filtered_files) == {Path(f) for f in expected_files}


@pytest.fixture
def flux_schnell_test_files() -> list[Path]:
    return [
        Path(f)
        for f in [
            "FLUX.1-schnell/.gitattributes",
            "FLUX.1-schnell/README.md",
            "FLUX.1-schnell/ae.safetensors",
            "FLUX.1-schnell/flux1-schnell.safetensors",
            "FLUX.1-schnell/model_index.json",
            "FLUX.1-schnell/scheduler/scheduler_config.json",
            "FLUX.1-schnell/schnell_grid.jpeg",
            "FLUX.1-schnell/text_encoder/config.json",
            "FLUX.1-schnell/text_encoder/model.safetensors",
            "FLUX.1-schnell/text_encoder_2/config.json",
            "FLUX.1-schnell/text_encoder_2/model-00001-of-00002.safetensors",
            "FLUX.1-schnell/text_encoder_2/model-00002-of-00002.safetensors",
            "FLUX.1-schnell/text_encoder_2/model.safetensors.index.json",
            "FLUX.1-schnell/tokenizer/merges.txt",
            "FLUX.1-schnell/tokenizer/special_tokens_map.json",
            "FLUX.1-schnell/tokenizer/tokenizer_config.json",
            "FLUX.1-schnell/tokenizer/vocab.json",
            "FLUX.1-schnell/tokenizer_2/special_tokens_map.json",
            "FLUX.1-schnell/tokenizer_2/spiece.model",
            "FLUX.1-schnell/tokenizer_2/tokenizer.json",
            "FLUX.1-schnell/tokenizer_2/tokenizer_config.json",
            "FLUX.1-schnell/transformer/config.json",
            "FLUX.1-schnell/transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "FLUX.1-schnell/transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "FLUX.1-schnell/transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "FLUX.1-schnell/transformer/diffusion_pytorch_model.safetensors.index.json",
            "FLUX.1-schnell/vae/config.json",
            "FLUX.1-schnell/vae/diffusion_pytorch_model.safetensors",
        ]
    ]


@pytest.mark.parametrize(
    ["variant", "expected_files"],
    [
        (
            ModelRepoVariant.Default,
            [
                "FLUX.1-schnell/model_index.json",
                "FLUX.1-schnell/scheduler/scheduler_config.json",
                "FLUX.1-schnell/text_encoder/config.json",
                "FLUX.1-schnell/text_encoder/model.safetensors",
                "FLUX.1-schnell/text_encoder_2/config.json",
                "FLUX.1-schnell/text_encoder_2/model-00001-of-00002.safetensors",
                "FLUX.1-schnell/text_encoder_2/model-00002-of-00002.safetensors",
                "FLUX.1-schnell/text_encoder_2/model.safetensors.index.json",
                "FLUX.1-schnell/tokenizer/merges.txt",
                "FLUX.1-schnell/tokenizer/special_tokens_map.json",
                "FLUX.1-schnell/tokenizer/tokenizer_config.json",
                "FLUX.1-schnell/tokenizer/vocab.json",
                "FLUX.1-schnell/tokenizer_2/special_tokens_map.json",
                "FLUX.1-schnell/tokenizer_2/spiece.model",
                "FLUX.1-schnell/tokenizer_2/tokenizer.json",
                "FLUX.1-schnell/tokenizer_2/tokenizer_config.json",
                "FLUX.1-schnell/transformer/config.json",
                "FLUX.1-schnell/transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
                "FLUX.1-schnell/transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
                "FLUX.1-schnell/transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
                "FLUX.1-schnell/transformer/diffusion_pytorch_model.safetensors.index.json",
                "FLUX.1-schnell/vae/config.json",
                "FLUX.1-schnell/vae/diffusion_pytorch_model.safetensors",
            ],
        ),
    ],
)
def test_select_flux_schnell_files(
    flux_schnell_test_files: list[Path], variant: ModelRepoVariant, expected_files: list[str]
) -> None:
    filtered_files = filter_files(flux_schnell_test_files, variant)
    assert set(filtered_files) == {Path(f) for f in expected_files}


# A subset of huggingface.co/MiniMaxAI/MiniMax-H3: a root-level Modular Diffusers pipeline whose
# repo also carries a sibling task transformer (transformer_ref) that a slim install must be able
# to skip while still fetching the root pipeline index and the transformer's bare config.json.
@pytest.fixture
def minimax_h3_test_files() -> list[Path]:
    return [
        Path(f)
        for f in [
            "MiniMax-H3/.gitattributes",
            "MiniMax-H3/LICENSE",
            "MiniMax-H3/README.md",
            "MiniMax-H3/model_index.json",
            "MiniMax-H3/modular_model_index.json",
            "MiniMax-H3/audio_scheduler/scheduler_config.json",
            "MiniMax-H3/audio_vae/config.json",
            "MiniMax-H3/audio_vae/diffusion_pytorch_model.safetensors",
            "MiniMax-H3/processor/chat_template.json",
            "MiniMax-H3/processor/preprocessor_config.json",
            "MiniMax-H3/processor/tokenizer.json",
            "MiniMax-H3/scheduler/scheduler_config.json",
            "MiniMax-H3/text_encoder/config.json",
            "MiniMax-H3/text_encoder/model-00001-of-00002.safetensors",
            "MiniMax-H3/text_encoder/model-00002-of-00002.safetensors",
            "MiniMax-H3/text_encoder/model.safetensors.index.json",
            "MiniMax-H3/tokenizer/merges.txt",
            "MiniMax-H3/tokenizer/tokenizer_config.json",
            "MiniMax-H3/tokenizer/vocab.json",
            "MiniMax-H3/transformer/config.json",
            "MiniMax-H3/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
            "MiniMax-H3/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            "MiniMax-H3/transformer/diffusion_pytorch_model.safetensors.index.json",
            "MiniMax-H3/transformer_ref/config.json",
            "MiniMax-H3/transformer_ref/diffusion_pytorch_model-00001-of-00002.safetensors",
            "MiniMax-H3/transformer_ref/diffusion_pytorch_model-00002-of-00002.safetensors",
            "MiniMax-H3/vae/config.json",
            "MiniMax-H3/vae/diffusion_pytorch_model-00001-of-00003.safetensors",
            "MiniMax-H3/vae/diffusion_pytorch_model-00002-of-00003.safetensors",
            "MiniMax-H3/vae/diffusion_pytorch_model-00003-of-00003.safetensors",
            "MiniMax-H3/vae/diffusion_pytorch_model.safetensors.index.json",
        ]
    ]


def test_select_subfolders_with_explicit_files(minimax_h3_test_files: list[Path]) -> None:
    """Explicit file entries ride alongside subfolder entries: the root pipeline index and the
    transformer's bare config.json are included verbatim (the latter would otherwise be dropped by
    the config-only-folder pruning), while the transformer weights stay excluded."""
    filtered_files = filter_files(
        minimax_h3_test_files,
        subfolders=[
            Path("modular_model_index.json"),
            Path("transformer/config.json"),
            Path("tokenizer"),
            Path("processor"),
            Path("vae"),
            Path("audio_vae"),
        ],
    )
    assert set(filtered_files) == {
        Path(f)
        for f in [
            "MiniMax-H3/modular_model_index.json",
            "MiniMax-H3/transformer/config.json",
            "MiniMax-H3/tokenizer/merges.txt",
            "MiniMax-H3/tokenizer/tokenizer_config.json",
            "MiniMax-H3/tokenizer/vocab.json",
            "MiniMax-H3/processor/chat_template.json",
            "MiniMax-H3/processor/preprocessor_config.json",
            "MiniMax-H3/processor/tokenizer.json",
            "MiniMax-H3/vae/config.json",
            "MiniMax-H3/vae/diffusion_pytorch_model-00001-of-00003.safetensors",
            "MiniMax-H3/vae/diffusion_pytorch_model-00002-of-00003.safetensors",
            "MiniMax-H3/vae/diffusion_pytorch_model-00003-of-00003.safetensors",
            "MiniMax-H3/vae/diffusion_pytorch_model.safetensors.index.json",
            "MiniMax-H3/audio_vae/config.json",
            "MiniMax-H3/audio_vae/diffusion_pytorch_model.safetensors",
        ]
    }


def test_select_explicit_files_only(minimax_h3_test_files: list[Path]) -> None:
    """A list made up solely of explicit files selects exactly those files - not the whole repo."""
    filtered_files = filter_files(
        minimax_h3_test_files,
        subfolders=[Path("modular_model_index.json"), Path("transformer/config.json")],
    )
    assert set(filtered_files) == {
        Path("MiniMax-H3/modular_model_index.json"),
        Path("MiniMax-H3/transformer/config.json"),
    }


def test_select_explicit_weights_file_bypasses_name_prefilter() -> None:
    """An explicit weights-file entry is included even when its name would fail the 'model'
    naming-convention prefilter that subfolder contents are subject to."""
    files = [
        Path(f)
        for f in [
            "Repo/README.md",
            "Repo/text_encoders/foo_int8_convrot.safetensors",
            "Repo/vae/config.json",
            "Repo/vae/diffusion_pytorch_model.safetensors",
        ]
    ]
    filtered_files = filter_files(
        files,
        subfolders=[Path("text_encoders/foo_int8_convrot.safetensors"), Path("vae")],
    )
    assert set(filtered_files) == {
        Path("Repo/text_encoders/foo_int8_convrot.safetensors"),
        Path("Repo/vae/config.json"),
        Path("Repo/vae/diffusion_pytorch_model.safetensors"),
    }


def test_select_nonexistent_entry_selects_nothing(minimax_h3_test_files: list[Path]) -> None:
    """An entry matching neither a file nor a folder contributes nothing (and does not disable
    the subfolder filtering for the remaining entries)."""
    filtered_files = filter_files(
        minimax_h3_test_files,
        subfolders=[Path("no_such_entry.json"), Path("audio_vae")],
    )
    assert set(filtered_files) == {
        Path("MiniMax-H3/audio_vae/config.json"),
        Path("MiniMax-H3/audio_vae/diffusion_pytorch_model.safetensors"),
    }


def test_select_multiple_plain_subfolders_unchanged(minimax_h3_test_files: list[Path]) -> None:
    """Regression: a pure-folder multi-subfolder list (the pre-existing '+' syntax) behaves as
    before - explicit-file handling must not alter it."""
    filtered_files = filter_files(
        minimax_h3_test_files,
        subfolders=[Path("text_encoder"), Path("tokenizer")],
    )
    assert set(filtered_files) == {
        Path(f)
        for f in [
            "MiniMax-H3/text_encoder/config.json",
            "MiniMax-H3/text_encoder/model-00001-of-00002.safetensors",
            "MiniMax-H3/text_encoder/model-00002-of-00002.safetensors",
            "MiniMax-H3/text_encoder/model.safetensors.index.json",
            "MiniMax-H3/tokenizer/merges.txt",
            "MiniMax-H3/tokenizer/tokenizer_config.json",
            "MiniMax-H3/tokenizer/vocab.json",
        ]
    }
