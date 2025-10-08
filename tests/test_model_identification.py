import json
from pathlib import Path
from typing import Optional

import torch

from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk, StateDict

METADATA_KEY = "metadata_key_for_stripped_models"


STR_TO_DTYPE = {str(dtype): dtype for dtype in torch.__dict__.values() if isinstance(dtype, torch.dtype)}


def dress(v):
    match v:
        case {"shape": shape, "dtype": dtype_str, "fakeTensor": True}:
            dtype = STR_TO_DTYPE[dtype_str]
            return torch.empty(shape, dtype=dtype)
        case dict():
            return {k: dress(v) for k, v in v.items()}
        case list() | tuple():
            return [dress(x) for x in v]
        case _:
            return v


def load_stripped_model(path: Path, *args, **kwargs):
    with open(path, "r") as f:
        contents = json.load(f)
        contents.pop(METADATA_KEY, None)
    return dress(contents)


class StrippedModelOnDisk(ModelOnDisk):
    def load_state_dict(self, path: Optional[Path] = None) -> StateDict:
        path = self.resolve_weight_file(path)
        return load_stripped_model(path)

    def metadata(self, path: Optional[Path] = None) -> dict[str, str]:
        path = self.resolve_weight_file(path)
        with open(path, "r") as f:
            contents = json.load(f)
        return contents.get(METADATA_KEY, {})


base_path = Path("/home/bat/git/InvokeAI/tests/test_model_probe/stripped_models/")

EXPECTED_SUFFIX = "_expected_config.json"


def test_model_probe():
    for p in base_path.iterdir():
        if EXPECTED_SUFFIX in p.name:
            continue
        if "flat_colour_anime_style_schnell_v3.4" in p.name:
            # this one is broken for some reason
            pass
        mod = StrippedModelOnDisk(p)
        config = ModelConfigFactory.from_model_on_disk(mod)
        file_name = f"{p.name}_{EXPECTED_SUFFIX}"
        with open(base_path / file_name, "w") as f:
            f.write(config.model_dump_json(indent=2))


if __name__ == "__main__":
    test_model_probe()
