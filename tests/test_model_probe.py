import abc
import json
from pathlib import Path
from typing import Any

import pydantic
import pytest
import torch
from polyfactory.factories.pydantic_factory import ModelFactory
from sympy.testing.pytest import slow
from torch import tensor

from invokeai.backend.model_manager.config import (
    BaseModelType,
    CLIPGEmbedDiffusersConfig,
    CLIPLEmbedDiffusersConfig,
    CLIPVisionDiffusersConfig,
    ControlLoRADiffusersConfig,
    ControlLoRALyCORISConfig,
    ControlNetCheckpointConfig,
    ControlNetDiffusersConfig,
    FluxReduxConfig,
    InvalidModelConfigException,
    IPAdapterCheckpointConfig,
    IPAdapterInvokeAIConfig,
    LoRADiffusersConfig,
    LoRALyCORISConfig,
    MainBnbQuantized4bCheckpointConfig,
    MainCheckpointConfig,
    MainDiffusersConfig,
    MainGGUFCheckpointConfig,
    ModelConfigBase,
    ModelConfigFactory,
    ModelFormat,
    ModelOnDisk,
    ModelRepoVariant,
    ModelType,
    ModelVariantType,
    SigLIPConfig,
    SpandrelImageToImageConfig,
    T2IAdapterConfig,
    T5EncoderBnbQuantizedLlmInt8bConfig,
    T5EncoderConfig,
    TextualInversionFileConfig,
    TextualInversionFolderConfig,
    VAECheckpointConfig,
    VAEDiffusersConfig,
    concrete_subclasses,
)
from invokeai.backend.model_manager.legacy_probe import (
    CkptType,
    ModelProbe,
    VaeFolderProbe,
    get_default_settings_control_adapters,
    get_default_settings_main,
)
from invokeai.backend.model_manager.search import ModelSearch


@pytest.mark.parametrize(
    "vae_path,expected_type",
    [
        ("sd-vae-ft-mse", BaseModelType.StableDiffusion1),
        ("sdxl-vae", BaseModelType.StableDiffusionXL),
        ("taesd", BaseModelType.StableDiffusion1),
        ("taesdxl", BaseModelType.StableDiffusionXL),
    ],
)
def test_get_base_type(vae_path: str, expected_type: BaseModelType, datadir: Path):
    sd1_vae_path = datadir / "vae" / vae_path
    probe = VaeFolderProbe(sd1_vae_path)
    base_type = probe.get_base_type()
    assert base_type == expected_type
    repo_variant = probe.get_repo_variant()
    assert repo_variant == ModelRepoVariant.Default


def test_repo_variant(datadir: Path):
    probe = VaeFolderProbe(datadir / "vae" / "taesdxl-fp16")
    repo_variant = probe.get_repo_variant()
    assert repo_variant == ModelRepoVariant.FP16


def test_controlnet_t2i_default_settings():
    assert get_default_settings_control_adapters("some_canny_model").preprocessor == "canny_image_processor"
    assert get_default_settings_control_adapters("some_depth_model").preprocessor == "depth_anything_image_processor"
    assert get_default_settings_control_adapters("some_pose_model").preprocessor == "dw_openpose_image_processor"
    assert get_default_settings_control_adapters("i like turtles") is None


def test_default_settings_main():
    assert get_default_settings_main(BaseModelType.StableDiffusion1).width == 512
    assert get_default_settings_main(BaseModelType.StableDiffusion1).height == 512
    assert get_default_settings_main(BaseModelType.StableDiffusion2).width == 512
    assert get_default_settings_main(BaseModelType.StableDiffusion2).height == 512
    assert get_default_settings_main(BaseModelType.StableDiffusionXL).width == 1024
    assert get_default_settings_main(BaseModelType.StableDiffusionXL).height == 1024
    assert get_default_settings_main(BaseModelType.StableDiffusionXLRefiner) is None
    assert get_default_settings_main(BaseModelType.Any) is None


def test_probe_handles_state_dict_with_integer_keys(tmp_path: Path):
    # This structure isn't supported by invoke, but we still need to handle it gracefully.
    # See https://github.com/invoke-ai/InvokeAI/issues/6044
    state_dict_with_integer_keys: CkptType = {
        320: (
            {
                "linear1.weight": tensor([1.0]),
                "linear1.bias": tensor([1.0]),
                "linear2.weight": tensor([1.0]),
                "linear2.bias": tensor([1.0]),
            },
            {
                "linear1.weight": tensor([1.0]),
                "linear1.bias": tensor([1.0]),
                "linear2.weight": tensor([1.0]),
                "linear2.bias": tensor([1.0]),
            },
        ),
    }
    sd_path = tmp_path / "sd.pt"
    torch.save(state_dict_with_integer_keys, sd_path)
    with pytest.raises(InvalidModelConfigException):
        ModelProbe.get_model_type_from_checkpoint(sd_path, state_dict_with_integer_keys)


def test_probe_sd1_diffusers_inpainting(datadir: Path):
    config = ModelProbe.probe(datadir / "sd-1/main/dreamshaper-8-inpainting")
    assert isinstance(config, MainDiffusersConfig)
    assert config.base is BaseModelType.StableDiffusion1
    assert config.variant is ModelVariantType.Inpaint
    assert config.repo_variant is ModelRepoVariant.FP16


class MinimalConfigExample(ModelConfigBase):
    type: ModelType = ModelType.Main
    format: ModelFormat = ModelFormat.Checkpoint
    fun_quote: str

    @classmethod
    def matches(cls, mod: ModelOnDisk) -> bool:
        return mod.path.suffix == ".json"

    @classmethod
    def parse(cls, mod: ModelOnDisk) -> dict[str, Any]:
        with open(mod.path, "r") as f:
            contents = json.load(f)

        return {
            "fun_quote": contents["quote"],
            "base": BaseModelType.Any,
        }


def test_minimal_working_example(datadir: Path):
    model_path = datadir / "minimal_config_model.json"
    overrides = {"base": BaseModelType.StableDiffusion1}
    config = ModelConfigBase.classify(model_path, **overrides)

    assert isinstance(config, MinimalConfigExample)
    assert config.base == BaseModelType.StableDiffusion1
    assert config.path == model_path.as_posix()
    assert config.fun_quote == "Minimal working example of a ModelConfigBase subclass"


def test_regression_against_model_probe(datadir: Path):
    """Ensure ModelConfigBase.classify returns consistent results as ModelProbe.probe"""
    model_paths = ModelSearch().search(datadir)  # TODO: add more 'stripped' models to test_model_probe directory
    for path in model_paths:
        legacy_config = new_config = None
        probe_success = classify_success = True

        try:
            legacy_config = ModelProbe.probe(path)
        except InvalidModelConfigException:
            probe_success = False

        try:
            new_config = ModelConfigBase.classify(path)
        except InvalidModelConfigException:
            classify_success = False

        if probe_success and classify_success:
            assert legacy_config == new_config

        elif probe_success:
            assert type(legacy_config) in ModelConfigBase._USING_LEGACY_PROBE

        elif classify_success:
            assert type(new_config) not in ModelConfigBase._USING_LEGACY_PROBE

        else:
            raise ValueError(f"Both probe and classify failed to classify model at path {path}.")


class ConfigMocker:
    """Utility class to create config instances with random data"""

    def __init__(self):
        self._factories = {}

    def mock(self, config_cls, count):
        if config_cls not in self._factories:

            class Factory(ModelFactory[config_cls]):
                __use_defaults__ = True
                __random_seed__ = 1234
                __check_model__ = True

            f = Factory()
            self._factories[config_cls] = f
        factory = self._factories[config_cls]
        return [factory.build() for _ in range(count)]


@slow
def test_serialisation_roundtrip():
    """After classification, models are serialised to json and stored in the database.
    We need to ensure they are de-serialised into the original config with all relevant fields restored.
    """
    mocker = ConfigMocker()

    excluded = {MinimalConfigExample}
    for config_cls in concrete_subclasses(ModelConfigBase) - excluded:
        trials_per_class = 50
        configs_with_random_data = mocker.mock(config_cls, trials_per_class)

        for config in configs_with_random_data:
            as_json = config.model_dump_json()
            as_dict = json.loads(as_json)
            reconstructed = ModelConfigFactory.make_config(as_dict)
            assert isinstance(reconstructed, config_cls)
            assert config.model_dump_json() == reconstructed.model_dump_json()


def test_inheritance_order():
    """
    Safeguard test to warn against incorrect inheritance order.

    Config classes using multiple inheritance should inherit from ModelConfigBase last
    to ensure that more specific fields take precedence over the generic defaults.

    It may be worth rethinking our config taxonomy in the future, but in the meantime,
    this test can help prevent the debugging effort I went through discovering this.
    """
    for config_cls in concrete_subclasses(ModelConfigBase):
        excluded = {abc.ABC, pydantic.BaseModel, object}
        inheritance_list = [cls for cls in config_cls.mro() if cls not in excluded]
        assert inheritance_list[-1] is ModelConfigBase


def test_concrete_subclasses():
    excluded = {MinimalConfigExample}
    config_classes = concrete_subclasses(ModelConfigBase) - excluded
    expected = {
        CLIPGEmbedDiffusersConfig,
        MainGGUFCheckpointConfig,
        T2IAdapterConfig,
        TextualInversionFolderConfig,
        IPAdapterInvokeAIConfig,
        ControlNetDiffusersConfig,
        ControlLoRALyCORISConfig,
        MainDiffusersConfig,
        LoRALyCORISConfig,
        CLIPVisionDiffusersConfig,
        MainCheckpointConfig,
        T5EncoderConfig,
        IPAdapterCheckpointConfig,
        VAEDiffusersConfig,
        LoRADiffusersConfig,
        ControlNetCheckpointConfig,
        FluxReduxConfig,
        T5EncoderBnbQuantizedLlmInt8bConfig,
        SpandrelImageToImageConfig,
        MainBnbQuantized4bCheckpointConfig,
        TextualInversionFileConfig,
        CLIPLEmbedDiffusersConfig,
        VAECheckpointConfig,
        ControlLoRADiffusersConfig,
        SigLIPConfig,
    }
    assert config_classes == expected
