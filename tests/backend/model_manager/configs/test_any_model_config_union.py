"""Every config identification can return is one that a stored record can be read back as."""

from typing import get_args

import pytest

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig


@pytest.mark.platform_independent
def test_every_identifiable_config_reads_back_from_its_record() -> None:
    """Identification tries every class in `Config_Base.CONFIG_CLASSES`; a record is read back through
    `AnyModelConfig`. A class missing from the union identifies and installs, and then its own record cannot be
    read, so the model fails to load from the database it was just written to."""
    union = get_args(AnyModelConfig)[0]
    readable = {get_args(member)[0] for member in get_args(union)}

    assert sorted(config_class.__name__ for config_class in Config_Base.CONFIG_CLASSES - readable) == []
