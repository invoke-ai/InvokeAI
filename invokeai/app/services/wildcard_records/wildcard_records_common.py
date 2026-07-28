import json
import re
from typing import Any

from dynamicprompts.wildcards import WildcardManager
from pydantic import BaseModel, Field, TypeAdapter, field_validator


class WildcardNotFoundError(Exception):
    """Raised when a wildcard is not found"""


class WildcardNameConflictError(Exception):
    """Raised when a user already owns a wildcard with the requested name"""


# A wildcard name is spliced into a prompt as `__name__`, and `/` is dynamicprompts' nesting
# separator (`__animals/dogs__`). Each segment must begin and end alphanumerically: a leading or
# trailing underscore would run into the `__` delimiters and make the reference ambiguous
# (`__trailing___` could be read either way). Underscores and hyphens are fine inside.
_WILDCARD_NAME_SEGMENT = r"[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?"
WILDCARD_NAME_RE = re.compile(rf"^{_WILDCARD_NAME_SEGMENT}(?:/{_WILDCARD_NAME_SEGMENT})*$")

MAX_WILDCARD_NAME_LENGTH = 128


def validate_wildcard_name(name: str) -> str:
    """Normalizes and validates a wildcard name, or raises ValueError."""
    normalized = name.strip()

    if not normalized:
        raise ValueError("Wildcard name must not be empty")
    if len(normalized) > MAX_WILDCARD_NAME_LENGTH:
        raise ValueError(f"Wildcard name must be at most {MAX_WILDCARD_NAME_LENGTH} characters")
    if not WILDCARD_NAME_RE.match(normalized):
        raise ValueError(
            "Wildcard name may contain only letters, numbers, underscores and hyphens, must start "
            "and end with a letter or number, and may be nested with '/' (for example 'animals/dogs')"
        )

    return normalized


def _clean_values(values: list[str]) -> list[str]:
    """Drops blank entries and surrounding whitespace; a wildcard of empty strings expands to nothing."""
    return [stripped for stripped in (value.strip() for value in values) if stripped]


class WildcardWithoutId(BaseModel):
    name: str = Field(description="The wildcard's name, referenced in a prompt as `__name__`.")
    values: list[str] = Field(description="The values this wildcard expands to.")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return validate_wildcard_name(value)

    @field_validator("values")
    @classmethod
    def _validate_values(cls, value: list[str]) -> list[str]:
        return _clean_values(value)


class WildcardChanges(BaseModel, extra="forbid"):
    name: str | None = Field(default=None, description="The wildcard's new name.")
    values: list[str] | None = Field(default=None, description="The wildcard's new values.")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return None if value is None else validate_wildcard_name(value)

    @field_validator("values")
    @classmethod
    def _validate_values(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else _clean_values(value)


class WildcardRecordDTO(WildcardWithoutId):
    id: str = Field(description="The wildcard ID.")
    user_id: str = Field(description="The user who owns this wildcard.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WildcardRecordDTO":
        data["values"] = json.loads(data.get("values") or "[]")
        return WildcardRecordDTOValidator.validate_python(data)


WildcardRecordDTOValidator = TypeAdapter(WildcardRecordDTO)


def build_wildcard_manager(wildcards: list[WildcardRecordDTO]) -> WildcardManager:
    """Builds the manager that resolves `__name__` against a user's wildcards.

    `root_map` accepts in-memory value lists, so wildcards resolve without ever touching the
    filesystem. Wildcards with no values are omitted: an empty collection would register the name as
    "known" while expanding to nothing, which reads as a silently dropped prompt.

    Constructed per request. The tree is built lazily from a handful of small lists, so there is
    nothing here worth caching and invalidating against edits.
    """
    root_map = {"": [{wildcard.name: wildcard.values for wildcard in wildcards if wildcard.values}]}
    return WildcardManager(root_map=root_map)
