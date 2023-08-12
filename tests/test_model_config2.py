"""
Test the refactored model config classes.
"""

from pathlib import Path

from invokeai.backend.model_management2.model_config import (
    ModelConfig,
    InvalidModelConfigException,
    MainCheckpointConfig,
    MainDiffusersConfig,
    LoRAConfig,
    TextualInversionConfig,
    ONNXSD1Config,
    ONNXSD2Config,
    ValidationError,
)


def test_checkpoints():
    raw = dict(
        path="/tmp/foo.ckpt",
        name="foo",
        base_model="sd-1",
        model_type="main",
        config="/tmp/foo.yaml",
        model_variant="normal",
        model_format="checkpoint",
    )
    config = ModelConfig.parse_obj(raw)
    assert isinstance(config, MainCheckpointConfig)
    assert config.model_format == "checkpoint"
    assert config.base_model == "sd-1"
    assert config.vae is None


def test_diffusers():
    raw = dict(
        path="/tmp/foo",
        name="foo",
        base_model="sd-2",
        model_type="main",
        model_variant="inpaint",
        model_format="diffusers",
        vae="/tmp/foobar/vae.pt",
    )
    config = ModelConfig.parse_obj(raw)
    assert isinstance(config, MainDiffusersConfig)
    assert config.model_format == "diffusers"
    assert config.base_model == "sd-2"
    assert config.model_variant == "inpaint"
    assert config.vae == Path("/tmp/foobar/vae.pt")
    assert isinstance(config.vae, Path)


def test_invalid_diffusers():
    raw = dict(
        path="/tmp/foo",
        name="foo",
        base_model="sd-2",
        model_type="main",
        model_variant="inpaint",
        config="/tmp/foo.ckpt",
        model_format="diffusers",
    )
    # This is expected to fail with a validation error, because
    # diffusers format does not have a `config` field
    try:
        ModelConfig.parse_obj(raw)
        assert False, "Validation should have failed"
    except InvalidModelConfigException:
        assert True


def test_lora():
    raw = dict(
        path="/tmp/foo",
        name="foo",
        base_model="sdxl",
        model_type="lora",
        model_format="lycoris",
    )
    config = ModelConfig.parse_obj(raw)
    assert isinstance(config, LoRAConfig)
    assert config.model_format == "lycoris"
    raw["model_format"] = "diffusers"
    config = ModelConfig.parse_obj(raw)
    assert isinstance(config, LoRAConfig)
    assert config.model_format == "diffusers"


def test_embedding():
    raw = dict(
        path="/tmp/foo",
        name="foo",
        base_model="sdxl-refiner",
        model_type="embedding",
        model_format="embedding_file",
    )
    config = ModelConfig.parse_obj(raw)
    assert isinstance(config, TextualInversionConfig)
    assert config.model_format == "embedding_file"


def test_onnx():
    raw = dict(
        path="/tmp/foo.ckpt",
        name="foo",
        base_model="sd-1",
        model_type="onnx",
        model_variant="normal",
        model_format="onnx",
    )
    config = ModelConfig.parse_obj(raw)
    assert isinstance(config, ONNXSD1Config)
    assert config.model_format == "onnx"

    raw["base_model"] = "sd-2"
    # this should not validate without the upcast_attention field
    try:
        ModelConfig.parse_obj(raw)
        assert False, "Config should not have validated without upcast_attention"
    except InvalidModelConfigException:
        assert True

    raw["upcast_attention"] = True
    raw["prediction_type"] = "epsilon"
    config = ModelConfig.parse_obj(raw)
    assert isinstance(config, ONNXSD2Config)
    assert config.upcast_attention


def test_assignment():
    raw = dict(
        path="/tmp/foo.ckpt",
        name="foo",
        base_model="sd-2",
        model_type="onnx",
        model_variant="normal",
        model_format="onnx",
        upcast_attention=True,
        prediction_type="epsilon",
    )
    config = ModelConfig.parse_obj(raw)
    config.upcast_attention = False
    assert not config.upcast_attention
    try:
        config.prediction_type = "not valid"
        assert False, "Config should not have accepted invalid assignment"
    except ValidationError:
        assert True


def test_invalid_combination():
    raw = dict(
        path="/tmp/foo.ckpt",
        name="foo",
        base_model="sd-2",
        model_type="main",
        model_variant="normal",
        model_format="onnx",
        upcast_attention=True,
        prediction_type="epsilon",
    )
    try:
        ModelConfig.parse_obj(raw)
        assert False, "This should have raised an InvalidModelConfigException"
    except InvalidModelConfigException:
        assert True
