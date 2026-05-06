import fnmatch
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

WILDCARDS_DIR_NAME = "wildcards"
SUPPORTED_WILDCARD_EXTENSIONS = {
    ".txt": "txt",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}
WILDCARD_REFERENCE_PATTERN = re.compile(r"__([^\r\n]+?)__")
VISIBLE_BOM_PREFIXES = ("\ufeff", "\u00ef\u00bb\u00bf", "\u00c3\u00af\u00c2\u00bb\u00c2\u00bf")


class WildcardIndexItem(BaseModel):
    token: str
    path: str
    label: str
    file_type: Literal["txt", "json", "yaml"]
    value_count: int = Field(ge=0)
    samples: list[str]


class WildcardIndexError(BaseModel):
    path: str
    message: str


class WildcardsResponse(BaseModel):
    wildcard_dir: str = WILDCARDS_DIR_NAME
    wildcards: list[WildcardIndexItem]
    errors: list[WildcardIndexError]


class WildcardValuesResponse(BaseModel):
    token: str
    path: str
    label: str
    file_type: Literal["txt", "json", "yaml"]
    value_count: int = Field(ge=0)
    values: list[str]
    truncated: bool


def get_wildcards_path(root_path: Path) -> Path:
    return root_path / WILDCARDS_DIR_NAME


def index_wildcards(wildcards_path: Path, max_samples: int = 5) -> WildcardsResponse:
    wildcards: list[WildcardIndexItem] = []
    errors: list[WildcardIndexError] = []

    if not wildcards_path.exists():
        return WildcardsResponse(wildcards=wildcards, errors=errors)
    if not wildcards_path.is_dir():
        return WildcardsResponse(
            wildcards=wildcards,
            errors=[
                WildcardIndexError(path=WILDCARDS_DIR_NAME, message="Wildcards path exists but is not a folder"),
            ],
        )

    root = wildcards_path.resolve()
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue

        file_type = SUPPORTED_WILDCARD_EXTENSIONS.get(path.suffix.lower())
        if file_type is None:
            continue

        rel_path = _get_safe_relative_path(path, root)
        if rel_path is None:
            errors.append(WildcardIndexError(path=path.name, message="Wildcard file is outside the wildcards folder"))
            continue

        if path.is_symlink():
            errors.append(WildcardIndexError(path=_as_posix(rel_path), message="Symlinks are not supported"))
            continue

        try:
            if file_type == "txt":
                items = _index_txt_wildcard(path, rel_path, max_samples)
            else:
                items = _index_structured_wildcard(path, rel_path, file_type, max_samples)
            wildcards.extend(items)
        except Exception as e:
            errors.append(WildcardIndexError(path=_as_posix(rel_path), message=str(e)))

    wildcards.sort(key=lambda item: item.path)
    return WildcardsResponse(wildcards=wildcards, errors=errors)


def get_wildcard_references(prompt: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for match in WILDCARD_REFERENCE_PATTERN.finditer(prompt):
        path = normalize_wildcard_reference(match.group(1))
        if path and path not in seen:
            seen.add(path)
            refs.append(path)
    return refs


def find_missing_wildcard_references(prompt: str, wildcards: list[WildcardIndexItem]) -> list[str]:
    available_paths = {wildcard.path for wildcard in wildcards}
    missing: list[str] = []
    for ref in get_wildcard_references(prompt):
        if _reference_exists(ref, available_paths):
            continue
        missing.append(ref)
    return missing


def clean_dynamic_prompt_outputs(prompts: list[str]) -> list[str]:
    return [_clean_wildcard_value(prompt) for prompt in prompts]


def get_wildcard_values(wildcards_path: Path, reference: str, limit: int = 200) -> WildcardValuesResponse | None:
    if _has_absolute_wildcard_reference(reference):
        return None
    wildcard_path = normalize_wildcard_reference(reference)
    if not _is_safe_wildcard_path(wildcard_path) or not wildcards_path.exists() or not wildcards_path.is_dir():
        return None

    root = wildcards_path.resolve()
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue

        file_type = SUPPORTED_WILDCARD_EXTENSIONS.get(path.suffix.lower())
        if file_type is None:
            continue

        rel_path = _get_safe_relative_path(path, root)
        if rel_path is None or path.is_symlink():
            continue

        if file_type == "txt":
            item = _get_txt_wildcard_values(path, rel_path, wildcard_path, limit)
        else:
            item = _get_structured_wildcard_values(path, rel_path, file_type, wildcard_path, limit)

        if item is not None:
            return item

    return None


def normalize_wildcard_reference(reference: str) -> str:
    path = reference.strip()
    if path.startswith("~") or path.startswith("@"):
        path = path[1:]
    if "(" in path:
        path = path.split("(", 1)[0]
    path = path.replace("\\", "/").strip("/")
    return path


def _is_safe_wildcard_path(path: str) -> bool:
    if not path or "*" in path:
        return False
    candidate = Path(path)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _has_absolute_wildcard_reference(reference: str) -> bool:
    normalized = reference.strip().replace("\\", "/")
    if normalized.startswith("~") or normalized.startswith("@"):
        normalized = normalized[1:]
    return normalized.startswith("/")


def _reference_exists(reference: str, available_paths: set[str]) -> bool:
    if "*" in reference:
        return any(fnmatch.fnmatch(path, reference) for path in available_paths)
    return reference in available_paths


def _get_safe_relative_path(path: Path, root: Path) -> Path | None:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        return None
    return resolved.relative_to(root)


def _index_txt_wildcard(path: Path, rel_path: Path, max_samples: int) -> list[WildcardIndexItem]:
    values = _read_txt_values(path)
    wildcard_path = _as_posix(rel_path.with_suffix(""))
    return [
        WildcardIndexItem(
            token=f"__{wildcard_path}__",
            path=wildcard_path,
            label=Path(wildcard_path).name,
            file_type="txt",
            value_count=len(values),
            samples=values[:max_samples],
        )
    ]


def _get_txt_wildcard_values(
    path: Path, rel_path: Path, wildcard_path: str, limit: int
) -> WildcardValuesResponse | None:
    indexed_path = _as_posix(rel_path.with_suffix(""))
    if indexed_path != wildcard_path:
        return None
    values = _read_txt_values(path)
    return _make_values_response(indexed_path, "txt", values, limit)


def _read_txt_values(path: Path) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = _clean_wildcard_value(line)
        if value and not value.startswith("#"):
            values.append(value)
    return values


def _index_structured_wildcard(
    path: Path, rel_path: Path, file_type: Literal["json", "yaml"], max_samples: int
) -> list[WildcardIndexItem]:
    data = _read_structured_data(path, file_type)

    folder_prefix = _as_posix(rel_path.parent) if rel_path.parent != Path(".") else ""
    file_stem_path = _as_posix(rel_path.with_suffix(""))
    leaves = _collect_wildcard_leaves(data)
    items: list[WildcardIndexItem] = []

    if isinstance(data, list):
        values = _stringify_values(data)
        items.append(_make_index_item(file_stem_path, file_type, values, max_samples))
        return items

    for key_path, values in leaves:
        wildcard_path = "/".join(part for part in [folder_prefix, key_path] if part)
        items.append(_make_index_item(wildcard_path, file_type, values, max_samples))

    return items


def _get_structured_wildcard_values(
    path: Path,
    rel_path: Path,
    file_type: Literal["json", "yaml"],
    wildcard_path: str,
    limit: int,
) -> WildcardValuesResponse | None:
    data = _read_structured_data(path, file_type)
    folder_prefix = _as_posix(rel_path.parent) if rel_path.parent != Path(".") else ""
    file_stem_path = _as_posix(rel_path.with_suffix(""))

    if isinstance(data, list):
        if file_stem_path != wildcard_path:
            return None
        return _make_values_response(file_stem_path, file_type, _stringify_values(data), limit)

    for key_path, values in _collect_wildcard_leaves(data):
        indexed_path = "/".join(part for part in [folder_prefix, key_path] if part)
        if indexed_path == wildcard_path:
            return _make_values_response(indexed_path, file_type, values, limit)

    return None


def _read_structured_data(path: Path, file_type: Literal["json", "yaml"]) -> Any:
    text = path.read_text(encoding="utf-8")
    if file_type == "json":
        return json.loads(text)
    return yaml.safe_load(text)


def _make_index_item(
    wildcard_path: str, file_type: Literal["txt", "json", "yaml"], values: list[str], max_samples: int
) -> WildcardIndexItem:
    return WildcardIndexItem(
        token=f"__{wildcard_path}__",
        path=wildcard_path,
        label=Path(wildcard_path).name,
        file_type=file_type,
        value_count=len(values),
        samples=values[:max_samples],
    )


def _make_values_response(
    wildcard_path: str, file_type: Literal["txt", "json", "yaml"], values: list[str], limit: int
) -> WildcardValuesResponse:
    return WildcardValuesResponse(
        token=f"__{wildcard_path}__",
        path=wildcard_path,
        label=Path(wildcard_path).name,
        file_type=file_type,
        value_count=len(values),
        values=values[:limit],
        truncated=len(values) > limit,
    )


def _collect_wildcard_leaves(data: Any, prefix: str = "") -> list[tuple[str, list[str]]]:
    if isinstance(data, list):
        return [(prefix, _stringify_values(data))] if prefix else []

    if not isinstance(data, dict):
        return []

    leaves: list[tuple[str, list[str]]] = []
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        key_path = f"{prefix}/{key}" if prefix else key
        leaves.extend(_collect_wildcard_leaves(value, key_path))
    return leaves


def _stringify_values(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str):
            result.append(_clean_wildcard_value(value))
        elif isinstance(value, (int, float, bool)):
            result.append(str(value))
        elif isinstance(value, dict):
            result.extend(_clean_wildcard_value(key) for key in value.keys() if isinstance(key, str))
        if result and result[-1] == "":
            result.pop()
    return result


def _clean_wildcard_value(value: str) -> str:
    cleaned = value.strip()
    for prefix in VISIBLE_BOM_PREFIXES:
        cleaned = cleaned.replace(prefix, "")
    return cleaned.strip()


def _as_posix(path: Path) -> str:
    return path.as_posix()
