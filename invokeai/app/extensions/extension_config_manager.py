import pathlib
from typing import List, Optional

from omegaconf import OmegaConf
from pydantic import BaseModel


class ExtensionConfigSchema(BaseModel):
    repo_id: Optional[str]
    name: Optional[str]
    version: Optional[float]
    last_updated: Optional[int]


class ExtensionConfigManager:
    def __init__(self, config: pathlib.Path) -> None:
        self.config_file = config
        loaded_config = ExtensionConfigSchema(**OmegaConf.load(self.config_file)).dict()
        self.config: ExtensionConfigSchema = OmegaConf.structured(loaded_config)

    def get(self, keys: str | List[str]) -> dict | str:
        """
        Takes in a key or a list of keys and retrieves them from the extension config
        """
        if isinstance(keys, List):
            retrieved_keys = {}
            for key in keys:
                retrieved_keys[key] = OmegaConf.select(self.config, key)
            return retrieved_keys
        else:
            return OmegaConf.select(self.config, keys)

    def update(self, key, value):
        """
        Updates the key with a new value in the extension config
        """
        OmegaConf.update(self.config, key, value)
        OmegaConf.save(self.config, self.config_file)
