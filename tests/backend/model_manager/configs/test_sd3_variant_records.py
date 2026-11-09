"""SD 3 records written before the size variant existed have to keep loading.

The variant is optional on SD 3 mains and LoRAs and set only when a model is installed or re-identified; no migration
fills it in. A stored record without it must therefore still deserialize -- as the same config class, with no
variant -- and the UI and the denoise node then treat it as "size unknown".
"""

import pytest

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.configs.lora import LoRA_Diffusers_SD3_Config, LoRA_LyCORIS_SD3_Config
from invokeai.backend.model_manager.configs.main import Main_Diffusers_SD3_Config
from invokeai.backend.model_manager.taxonomy import ModelRepoVariant, SD3VariantType

_COMMON = {
    "key": "record",
    "hash": "blake3:0",
    "path": "models/sd3",
    "file_size": 1,
    "name": "SD 3",
    "source": "models/sd3",
    "source_type": "path",
}


@pytest.mark.parametrize(
    "config",
    [
        Main_Diffusers_SD3_Config(**_COMMON, repo_variant=ModelRepoVariant.Default, variant=SD3VariantType.Large),
        LoRA_LyCORIS_SD3_Config(**_COMMON, variant=SD3VariantType.Medium),
        LoRA_Diffusers_SD3_Config(**_COMMON, variant=SD3VariantType.Medium),
    ],
    ids=lambda config: type(config).__name__,
)
def test_a_stored_record_with_or_without_its_size_loads_as_the_same_config(config) -> None:
    stored = config.model_dump(mode="json")
    legacy = {k: v for k, v in stored.items() if k != "variant"}

    loaded = ModelConfigFactory.from_dict(stored)
    loaded_legacy = ModelConfigFactory.from_dict(legacy)

    assert type(loaded) is type(config) and loaded.variant is config.variant
    assert type(loaded_legacy) is type(config) and loaded_legacy.variant is None
