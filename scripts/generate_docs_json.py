from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Literal, cast, get_args, get_origin, get_type_hints

from pydantic.fields import FieldInfo

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared import invocation_context as invocation_context_module

OUTPUT_DIR = Path("docs-new/src/generated")
EXCLUDED_SETTINGS = {"schema_version", "legacy_models_yaml_path"}
INTERFACE_NAMES = (
    "ImagesInterface",
    "TensorsInterface",
    "ConditioningInterface",
    "ModelsInterface",
    "LoggerInterface",
    "ConfigInterface",
    "UtilInterface",
    "BoardsInterface",
)


def build_docs_bundle() -> dict[str, Any]:
    return {
        "invocation_context": build_invocation_context_export(),
        "settings": build_settings_export(),
    }


def _simplify_signature(sig: str) -> str:
    """Simplify a Python signature string for documentation display.

    - Strips 'self' parameter
    - Removes fully-qualified module paths (e.g. invokeai.backend.foo.Bar -> Bar)
    - Collapses large Annotated[Union[...]] type blocks to AnyModelConfig
    - Strips typing. prefixes
    - Strips ForwardRef() wrappers
    - Removes Discriminator(...) noise
    """
    # Remove 'self' parameter
    sig = re.sub(r"\(self(?:,\s*)?", "(", sig)

    # Strip typing. prefix early so bracket-balancing can find patterns
    sig = re.sub(r"\btyping\.", "", sig)

    # Collapse any Annotated[Union[Annotated[..._Config, Tag(...)], ...], Discriminator(...)]
    # These massive blocks are the AnyModelConfig discriminated union.
    # Match from "Annotated[Union[Annotated[" through to the closing "]]" including Discriminator.
    # We use a greedy approach: find the pattern start and then balance brackets.
    result = sig
    while True:
        # Find the start of an AnyModelConfig union block
        marker = "Annotated[Union[Annotated["
        start = result.find(marker)
        if start == -1:
            break
        # Find the balanced end - count brackets from 'Annotated[Union[...'
        depth = 0
        i = start
        while i < len(result):
            if result[i] == "[":
                depth += 1
            elif result[i] == "]":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        # Replace the entire block
        result = result[:start] + "AnyModelConfig" + result[i + 1 :]

    sig = result

    # Strip ForwardRef('...') -> just the name
    sig = re.sub(r"ForwardRef\('([^']+)'\)", r"\1", sig)

    # Strip fully-qualified module paths: some.module.path.ClassName -> ClassName
    sig = re.sub(r"[a-z_][a-z0-9_.]*\.([A-Z][A-Za-z0-9_]*)", r"\1", sig)

    # Clean up any remaining pathlib.Path -> Path
    sig = sig.replace("pathlib.Path", "Path")

    # Clean up PIL.Image.Image -> Image
    sig = re.sub(r"PIL\.I[a-zA-Z.]*", "Image", sig)

    # Collapse the AnyModel union type (used in loader callables)
    sig = sig.replace(
        "Union[ModelMixin, RawModel, Module, Dict[str, Tensor], DiffusionPipeline, InferenceSession]",
        "AnyModel",
    )

    # Clean up raw enum reprs like <ImageCGENERAL: 'general'> -> ImageCategory.GENERAL
    sig = re.sub(r"<ImageC([A-Z_]+):\s*'[^']*'>", r"ImageCategory.\1", sig)

    return sig


def build_invocation_context_export() -> dict[str, Any]:
    context_class = invocation_context_module.InvocationContext
    interfaces: list[dict[str, Any]] = []

    for interface_name in INTERFACE_NAMES:
        interface_class = getattr(invocation_context_module, interface_name)
        methods: list[dict[str, Any]] = []
        for method_name, method in inspect.getmembers(interface_class, predicate=inspect.isfunction):
            if method_name.startswith("_"):
                continue
            description, doc_parameters, returns = _parse_docstring(inspect.getdoc(method) or "")

            sig = inspect.signature(method)
            simplified_sig = _simplify_signature(str(sig))

            # Build a lookup from docstring param descriptions
            doc_param_map: dict[str, str] = {p["name"]: p["description"] for p in doc_parameters}

            # Extract type and default from the actual signature, merge with docstring descriptions
            parameters: list[dict[str, str]] = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                # Format the type annotation
                if param.annotation is not inspect.Parameter.empty:
                    param_type = _simplify_signature(inspect.formatannotation(param.annotation))
                else:
                    param_type = ""

                # Format the default value
                if param.default is not inspect.Parameter.empty:
                    default_str = _simplify_signature(repr(param.default))
                else:
                    default_str = ""

                parameters.append(
                    {
                        "name": param_name,
                        "type": param_type,
                        "default": default_str,
                        "description": doc_param_map.get(param_name, ""),
                    }
                )

            # Format the return type
            if sig.return_annotation is not inspect.Signature.empty:
                return_type = _simplify_signature(inspect.formatannotation(sig.return_annotation))
            else:
                return_type = ""

            methods.append(
                {
                    "name": method_name,
                    "signature": simplified_sig,
                    "description": description,
                    "parameters": parameters,
                    "returns": returns,
                    "return_type": return_type,
                }
            )

        interfaces.append(
            {
                "name": interface_name,
                "description": inspect.getdoc(interface_class) or "",
                "methods": methods,
            }
        )

    return {
        "name": context_class.__name__,
        "description": inspect.getdoc(context_class) or "",
        "interfaces": interfaces,
    }


def build_settings_export() -> dict[str, Any]:
    type_hints = get_type_hints(InvokeAIAppConfig)
    categories = _extract_settings_categories()
    settings: list[dict[str, Any]] = []

    fields = cast(dict[str, FieldInfo], InvokeAIAppConfig.model_fields)  # pyright: ignore[reportUnknownMemberType]

    for field_name, field_info in fields.items():
        if field_name in EXCLUDED_SETTINGS or field_info.exclude:
            continue

        field_type = type_hints.get(field_name)
        literal_values: list[Any] = []
        if get_origin(field_type) is Literal:
            literal_values = list(get_args(field_type))

        settings.append(
            {
                "name": field_name,
                "description": field_info.description or "",
                "type": str(field_info.annotation),
                "default": _normalize_value(field_info.default),
                "required": field_info.is_required(),
                "literal_values": literal_values,
                "env_var": f"INVOKEAI_{field_name.upper()}",
                "category": categories.get(field_name, "OTHER"),
                "validation": _extract_validation(field_info),
            }
        )

    return {"settings": settings}


def write_docs_bundle(bundle: dict[str, Any], output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "invocation-context.json": bundle["invocation_context"],
        "settings.json": bundle["settings"],
    }

    for name, payload in files.items():
        with open(output_dir / name, "w") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")


def _normalize_value(value: Any) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        items = cast(list[Any], value)
        normalized_list: list[Any] = [_normalize_value(item) for item in items]
        return normalized_list
    if isinstance(value, dict):
        entries = cast(dict[Any, Any], value)
        normalized_dict: dict[str, Any] = {str(key): _normalize_value(val) for key, val in entries.items()}
        return normalized_dict
    return str(value)


def _parse_docstring(docstring: str) -> tuple[str, list[dict[str, str]], str]:
    if not docstring:
        return "", [], ""

    lines = docstring.splitlines()
    description_lines: list[str] = []
    parameter_lines: list[str] = []
    return_lines: list[str] = []
    section = "description"

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == "Args:":
            section = "args"
            continue
        if stripped == "Returns:":
            section = "returns"
            continue
        if section == "description":
            description_lines.append(stripped)
        elif section == "args":
            parameter_lines.append(stripped)
        elif section == "returns":
            return_lines.append(stripped)

    parameters: list[dict[str, str]] = []
    current_name: str | None = None
    current_description: list[str] = []
    for line in parameter_lines:
        if not line:
            continue
        if ":" in line and not line.startswith("```"):
            if current_name is not None:
                parameters.append({"name": current_name, "description": " ".join(current_description).strip()})
            current_name, remainder = line.split(":", 1)
            current_name = current_name.strip()
            current_description = [remainder.strip()]
        elif current_name is not None:
            current_description.append(line)
    if current_name is not None:
        parameters.append({"name": current_name, "description": " ".join(current_description).strip()})

    description = "\n".join(line for line in description_lines if line).strip()
    returns = " ".join(line for line in return_lines if line).strip()
    return description, parameters, returns


def _extract_settings_categories() -> dict[str, str]:
    categories: dict[str, str] = {}
    current_category = "OTHER"
    config_path = Path(__file__).resolve().parent.parent / "invokeai/app/services/config/config_default.py"

    with open(config_path) as config_file:
        for raw_line in config_file:
            stripped = raw_line.strip()
            if stripped.startswith("# ") and stripped[2:].isupper() and "fmt:" not in stripped:
                current_category = stripped[2:]
                continue
            match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*:", stripped)
            if match:
                categories[match.group(1)] = current_category

    return categories


def _extract_validation(field_info: FieldInfo) -> dict[str, Any]:
    validation: dict[str, Any] = {}
    for attribute in ("gt", "ge", "lt", "le", "pattern"):
        value = getattr(field_info, attribute, None)
        if value is not None:
            validation[attribute] = value
    return validation


def main() -> None:
    os.chdir(Path(__file__).resolve().parent.parent)
    write_docs_bundle(build_docs_bundle())


if __name__ == "__main__":
    main()
