import abc
import json
from pathlib import Path
from typing import Any, get_args

import pydantic
import pytest
from polyfactory.factories.pydantic_factory import ModelFactory
from sympy.testing.pytest import slow

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.controlnet import ControlAdapterDefaultSettings
from invokeai.backend.model_manager.configs.factory import (
    AnyModelConfig,
    ModelConfigFactory,
)
from invokeai.backend.model_manager.configs.main import Main_Diffusers_SD1_Config, MainModelDefaultSettings
from invokeai.backend.model_manager.configs.vae import VAE_Diffusers_Config_Base
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.search import ModelSearch
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelType,
    ModelVariantType,
)
from invokeai.backend.util.logging import InvokeAILogger
from scripts.strip_models import StrippedModelOnDisk

logger = InvokeAILogger.get_logger(__file__)


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
    mod = ModelOnDisk(datadir / "vae" / vae_path)
    config = ModelConfigFactory.from_model_on_disk(mod)
    assert isinstance(config, VAE_Diffusers_Config_Base)
    assert config.base == expected_type
    assert config.repo_variant == ModelRepoVariant.Default


def test_repo_variant(datadir: Path):
    mod = ModelOnDisk(datadir / "vae" / "taesdxl-fp16")
    config = ModelConfigFactory.from_model_on_disk(mod)
    assert isinstance(config, VAE_Diffusers_Config_Base)
    assert config.repo_variant == ModelRepoVariant.FP16


def test_controlnet_t2i_default_settings():
    assert ControlAdapterDefaultSettings.from_model_name("some_canny_model").preprocessor == "canny_image_processor"
    assert (
        ControlAdapterDefaultSettings.from_model_name("some_depth_model").preprocessor
        == "depth_anything_image_processor"
    )
    assert (
        ControlAdapterDefaultSettings.from_model_name("some_pose_model").preprocessor == "dw_openpose_image_processor"
    )
    assert ControlAdapterDefaultSettings.from_model_name("i like turtles") is None


def test_default_settings_main():
    assert MainModelDefaultSettings.from_base(BaseModelType.StableDiffusion1).width == 512
    assert MainModelDefaultSettings.from_base(BaseModelType.StableDiffusion1).height == 512
    assert MainModelDefaultSettings.from_base(BaseModelType.StableDiffusion2).width == 512
    assert MainModelDefaultSettings.from_base(BaseModelType.StableDiffusion2).height == 512
    assert MainModelDefaultSettings.from_base(BaseModelType.StableDiffusionXL).width == 1024
    assert MainModelDefaultSettings.from_base(BaseModelType.StableDiffusionXL).height == 1024
    assert MainModelDefaultSettings.from_base(BaseModelType.StableDiffusionXLRefiner) is None
    assert MainModelDefaultSettings.from_base(BaseModelType.Any) is None


def test_probe_sd1_diffusers_inpainting(datadir: Path):
    mod = ModelOnDisk(datadir / "sd-1/main/dreamshaper-8-inpainting")
    config = ModelConfigFactory.from_model_on_disk(mod)
    assert isinstance(config, Main_Diffusers_SD1_Config)
    assert config.base is BaseModelType.StableDiffusion1
    assert config.variant is ModelVariantType.Inpaint
    assert config.repo_variant is ModelRepoVariant.FP16


class MinimalConfigExample(Config_Base):
    type: ModelType = ModelType.Main
    format: ModelFormat = ModelFormat.Checkpoint
    fun_quote: str

    @classmethod
    def matches(cls, mod: ModelOnDisk, **overrides) -> bool:
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
    config = ModelConfigFactory.from_model_on_disk(
        mod=model_path,
        overrides=overrides,
    )

    assert isinstance(config, MinimalConfigExample)
    assert config.base == BaseModelType.StableDiffusion1
    assert config.path == model_path.as_posix()
    assert config.fun_quote == "Minimal working example of a ModelConfigBase subclass"


@pytest.mark.xfail(reason="Known issue with 'helloyoung25d_V15j.safetensors'.", strict=True)
def test_regression_against_model_probe(datadir: Path, override_model_loading):
    """Verifies results from ModelConfigBase.classify are consistent with those from ModelProbe.probe.
    The test paths are gathered from the 'test_model_probe' directory.
    """
    configs_with_tests = set()
    model_paths = ModelSearch().search(datadir / "stripped_models")
    fake_hash = "abcdefgh"  # skip hashing to make test quicker
    fake_key = "123"  # fixed uuid for comparison

    for path in model_paths:
        legacy_config = new_config = None

        try:
            legacy_config = ModelProbe.probe(path, {"hash": fake_hash, "key": fake_key})
        except InvalidModelConfigException:
            pass

        try:
            stripped_mod = StrippedModelOnDisk(path)
            new_config = ModelConfigFactory.from_model_on_disk(
                mod=stripped_mod,
                overrides={"hash": fake_hash, "key": fake_key},
            )
        except InvalidModelConfigException:
            pass

        if legacy_config and new_config:
            assert type(legacy_config) is type(new_config)
            assert legacy_config.model_dump_json() == new_config.model_dump_json()

        elif legacy_config:
            assert type(legacy_config) in Config_Base.USING_LEGACY_PROBE

        elif new_config:
            assert type(new_config) in Config_Base.USING_CLASSIFY_API

        else:
            raise ValueError(f"Both probe and classify failed to classify model at path {path}.")

        config_type = type(legacy_config or new_config)
        configs_with_tests.add(config_type)

    untested_configs = Config_Base.all_config_classes() - configs_with_tests - {MinimalConfigExample}
    logger.warning(f"Function test_regression_against_model_probe missing test case for: {untested_configs}")


def create_fake_configs(config_cls, n):
    factory_args = {
        "__use_defaults__": True,
        "__random_seed__": 1234,
        "__check_model__": True,
    }
    factory = ModelFactory.create_factory(config_cls, **factory_args)
    return [factory.build() for _ in range(n)]


@slow
def test_serialisation_roundtrip():
    """After classification, models are serialised to json and stored in the database.
    We need to ensure they are de-serialised into the original config with all relevant fields restored.
    """
    excluded = {MinimalConfigExample}
    for config_cls in Config_Base.all_config_classes() - excluded:
        trials_per_class = 50
        configs_with_random_data = create_fake_configs(config_cls, trials_per_class)

        for config in configs_with_random_data:
            as_json = config.model_dump_json()
            as_dict = json.loads(as_json)
            reconstructed = ModelConfigFactory.make_config(as_dict)
            assert isinstance(reconstructed, config_cls)
            assert config.model_dump_json() == reconstructed.model_dump_json()


def test_discriminator_tagging_for_config_instances():
    """Verify that each ModelConfig instance is assigned the correct, unique Pydantic discriminator tag."""
    excluded = {MinimalConfigExample}
    config_classes = Config_Base.all_config_classes() - excluded

    tags = {c.get_tag() for c in config_classes}
    assert len(tags) == len(config_classes), "Each config should have its own unique tag"

    for config_cls in config_classes:
        expected_tag = config_cls.get_tag().tag

        trials_per_class = 3
        configs_with_random_data = create_fake_configs(config_cls, trials_per_class)

        for config in configs_with_random_data:
            assert get_model_discriminator_value(config) == expected_tag


def test_inheritance_order():
    """
    Safeguard test to warn against incorrect inheritance order.

    Config classes using multiple inheritance should inherit from ModelConfigBase last
    to ensure that more specific fields take precedence over the generic defaults.

    It may be worth rethinking our config taxonomy in the future, but in the meantime
    this test can help prevent debugging effort.
    """
    for config_cls in Config_Base.all_config_classes():
        excluded = {abc.ABC, pydantic.BaseModel, object}
        inheritance_list = [cls for cls in config_cls.mro() if cls not in excluded]
        assert inheritance_list[-1] is Config_Base


def test_any_model_config_includes_all_config_classes():
    """Safeguard test to ensure that AnyModelConfig includes all ModelConfigBase subclasses."""

    union_type = get_args(AnyModelConfig)[0]

    extracted = set()
    for annotated_pair in get_args(union_type):
        config_class, _ = get_args(annotated_pair)
        extracted.add(config_class)

    expected = set(Config_Base.all_config_classes()) - {MinimalConfigExample}
    assert extracted == expected


def test_config_uniquely_matches_model(datadir: Path):
    model_paths = ModelSearch().search(datadir / "stripped_models")
    for path in model_paths:
        mod = StrippedModelOnDisk(path)
        matches = {cls for cls in Config_Base.USING_CLASSIFY_API if cls.matches(mod)}
        assert len(matches) <= 1, f"Model at path {path} matches multiple config classes: {matches}"
        if not matches:
            logger.warning(f"Model at path {path} does not match any config classes using classify API.")
