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
