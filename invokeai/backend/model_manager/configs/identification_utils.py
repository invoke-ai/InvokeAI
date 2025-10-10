import json
from functools import cache
from pathlib import Path

from pydantic import BaseModel, ValidationError
from pydantic_core import CoreSchema, SchemaValidator
from typing_extensions import Any

from invokeai.backend.model_manager.model_on_disk import ModelOnDisk


class NotAMatchError(Exception):
    """Exception for when a model does not match a config class.

    Args:
        reason: The reason why the model did not match.
    """

    def __init__(self, reason: str):
        super().__init__(reason)


def get_config_dict_or_raise(config_path: Path | set[Path]) -> dict[str, Any]:
    """Load the diffusers/transformers model config file and return it as a dictionary. The config file is expected
    to be in JSON format.

    Args:
        config_path: The path to the config file, or a set of paths to try.

    Returns:
        The config file as a dictionary.

    Raises:
        NotAMatch if the config file is missing or cannot be loaded.
    """
    paths_to_check = config_path if isinstance(config_path, set) else {config_path}

    problems: dict[Path, str] = {}

    for p in paths_to_check:
        if not p.exists():
            problems[p] = "file does not exist"
            continue

        try:
            with open(p, "r") as file:
                config = json.load(file)

            return config
        except Exception as e:
            problems[p] = str(e)
            continue

    raise NotAMatchError(f"unable to load config file(s): {problems}")


def get_class_name_from_config_dict_or_raise(config: Path | set[Path] | dict[str, Any]) -> str:
    """Load the diffusers/transformers model config file and return the class name.

    Args:
        config_path: The path to the config file, or a set of paths to try.

    Returns:
        The class name from the config file.

    Raises:
        NotAMatch if the config file is missing or does not contain a valid class name.
    """

    if not isinstance(config, dict):
        config = get_config_dict_or_raise(config)

    try:
        if "_class_name" in config:
            # This is a diffusers-style config
            config_class_name = config["_class_name"]
        elif "architectures" in config:
            # This is a transformers-style config
            config_class_name = config["architectures"][0]
        else:
            raise ValueError("missing _class_name or architectures field")
    except Exception as e:
        raise NotAMatchError(f"unable to determine class name from config file: {config}") from e

    if not isinstance(config_class_name, str):
        raise NotAMatchError(f"_class_name or architectures field is not a string: {config_class_name}")

    return config_class_name


def raise_for_class_name(config: Path | set[Path] | dict[str, Any], class_name: str | set[str]) -> None:
    """Get the class name from the config file and raise NotAMatch if it is not in the expected set.

    Args:
        config_path: The path to the config file, or a set of paths to try.
        class_name: The expected class name, or a set of expected class names.

    Raises:
        NotAMatch if the class name is not in the expected set.
    """

    class_name = {class_name} if isinstance(class_name, str) else class_name

    actual_class_name = get_class_name_from_config_dict_or_raise(config)
    if actual_class_name not in class_name:
        raise NotAMatchError(f"invalid class name from config: {actual_class_name}")


def raise_for_override_fields(candidate_config_class: type[BaseModel], override_fields: dict[str, Any]) -> None:
    """Check if the provided override fields are valid for the config class using pydantic.

    For example, if the candidate config class has a field "base" of type Literal[BaseModelType.StableDiffusion1], and
    the override fields contain "base": BaseModelType.Flux, this function will raise NotAMatch.

    Internally, this function extracts the pydantic schema for each individual override field from the candidate config
    class and validates the override value against that schema. Post-instantiation validators are not run.

    Args:
        candidate_config_class: The config class that is being tested.
        override_fields: The override fields provided by the user.

    Raises:
        NotAMatch if any override field is invalid for the config class.
    """
    for field_name, override_value in override_fields.items():
        if field_name not in candidate_config_class.model_fields:
            raise NotAMatchError(f"unknown override field: {field_name}")
        try:
            PydanticFieldValidator.validate_field(candidate_config_class, field_name, override_value)
        except ValidationError as e:
            raise NotAMatchError(f"invalid override for field '{field_name}': {e}") from e


def raise_if_not_file(mod: ModelOnDisk) -> None:
    """Raise NotAMatch if the model path is not a file."""
    if not mod.path.is_file():
        raise NotAMatchError("model path is not a file")


def raise_if_not_dir(mod: ModelOnDisk) -> None:
    """Raise NotAMatch if the model path is not a directory."""
    if not mod.path.is_dir():
        raise NotAMatchError("model path is not a directory")


def state_dict_has_any_keys_exact(state_dict: dict[str | int, Any], keys: str | set[str]) -> bool:
    """Returns true if the state dict has any of the specified keys."""
    _keys = {keys} if isinstance(keys, str) else keys
    return any(key in state_dict for key in _keys)


def state_dict_has_any_keys_starting_with(state_dict: dict[str | int, Any], prefixes: str | set[str]) -> bool:
    """Returns true if the state dict has any keys starting with any of the specified prefixes."""
    _prefixes = {prefixes} if isinstance(prefixes, str) else prefixes
    return any(any(key.startswith(prefix) for prefix in _prefixes) for key in state_dict.keys() if isinstance(key, str))


def state_dict_has_any_keys_ending_with(state_dict: dict[str | int, Any], suffixes: str | set[str]) -> bool:
    """Returns true if the state dict has any keys ending with any of the specified suffixes."""
    _suffixes = {suffixes} if isinstance(suffixes, str) else suffixes
    return any(any(key.endswith(suffix) for suffix in _suffixes) for key in state_dict.keys() if isinstance(key, str))


def common_config_paths(path: Path) -> set[Path]:
    """Returns common config file paths for models stored in directories."""
    return {path / "config.json", path / "model_index.json"}


class PydanticFieldValidator:
    """Utility class for validating individual fields of a Pydantic model without instantiating the whole model.

    See: https://github.com/pydantic/pydantic/discussions/7367#discussioncomment-14213144
    """

    @staticmethod
    def find_field_schema(model: type[BaseModel], field_name: str) -> CoreSchema:
        """Find the Pydantic core schema for a specific field in a model."""
        schema: CoreSchema = model.__pydantic_core_schema__.copy()
        # we shallow copied, be careful not to mutate the original schema!

        assert schema["type"] in ["definitions", "model"]

        # find the field schema
        field_schema = schema["schema"]  # type: ignore
        while "fields" not in field_schema:
            field_schema = field_schema["schema"]  # type: ignore

        field_schema = field_schema["fields"][field_name]["schema"]  # type: ignore

        # if the original schema is a definition schema, replace the model schema with the field schema
        if schema["type"] == "definitions":
            schema["schema"] = field_schema
            return schema
        else:
            return field_schema

    @cache
    @staticmethod
    def get_validator(model: type[BaseModel], field_name: str) -> SchemaValidator:
        """Get a SchemaValidator for a specific field in a model."""
        return SchemaValidator(PydanticFieldValidator.find_field_schema(model, field_name))

    @staticmethod
    def validate_field(model: type[BaseModel], field_name: str, value: Any) -> Any:
        """Validate a value for a specific field in a model."""
        return PydanticFieldValidator.get_validator(model, field_name).validate_python(value)
