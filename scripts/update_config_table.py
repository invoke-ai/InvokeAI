import json
import os
from typing import Literal, get_args, get_type_hints

from invokeai.app.services.config.config_default import InvokeAIAppConfig

# Get the repo root directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_excluded = {"schema_version", "legacy_models_yaml_path"}


def get_type_name(field_type) -> str:
    """Return a human-readable string for a type hint."""
    origin = getattr(field_type, "__origin__", None)

    if origin is Literal:
        options = get_args(field_type)
        return " | ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in options)

    if origin is not None:
        args = get_args(field_type)
        base = getattr(origin, "__name__", str(origin))
        if args:
            inner = ", ".join(get_type_name(a) for a in args)
            return f"{base}[{inner}]"
        return base

    if hasattr(field_type, "__name__"):
        return field_type.__name__

    return str(field_type)


def main():
    # Change working directory to the repo root
    os.chdir(repo_root)

    type_hints = get_type_hints(InvokeAIAppConfig)
    rows = []

    for k, v in InvokeAIAppConfig.model_fields.items():
        if v.exclude or k in _excluded:
            continue

        field_type = type_hints.get(k)
        type_str = get_type_name(field_type) if field_type else "unknown"

        default = v.default
        if default is None:
            default_str = "None"
        elif isinstance(default, bool):
            default_str = str(default).lower()
        elif isinstance(default, str):
            default_str = f'"{default}"'
        else:
            default_str = str(default)

        description = v.description or ""

        if getattr(field_type, "__origin__", None) is Literal:
            options = get_args(field_type)
            formatted = ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in options)
            description += f" Valid values: {formatted}"

        rows.append(
            {
                "name": k,
                "type": type_str,
                "default": default_str,
                "description": description,
            }
        )

    # Output to JSON
    out_dir = os.path.join(repo_root, "docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "invoke-config.json")

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    rows_len = len(rows) if isinstance(rows, list) else 0
    print(f"Wrote {rows_len} config fields to {out_path}")


if __name__ == "__main__":
    main()
