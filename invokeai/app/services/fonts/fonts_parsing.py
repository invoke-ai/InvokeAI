"""Byte-level fontTools work run by the font worker subprocess.

This module imports neither the database layer nor the application config, so a worker
spawned for each parse or instance pays only for fontTools.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

from invokeai.app.services.fonts.fonts_common import (
    SUPPORTED_FONT_EXTENSIONS,
    FontAxis,
    FontInstance,
    FontValidationError,
    FontValidationResult,
)


def _normalize_variant_text(value: str) -> str:
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def _infer_font_weight(style_name: str, filename: str, weight_class: int | None) -> int:
    if isinstance(weight_class, int) and 1 <= weight_class <= 1000:
        return weight_class

    combined = _normalize_variant_text(f"{style_name} {filename}")
    weight_keywords = [
        (("thin", "hairline"), 100),
        (("extra light", "ultra light", "extralight", "ultralight"), 200),
        (("light",), 300),
        (("normal", "regular", "roman", "book"), 400),
        (("medium",), 500),
        (("semi bold", "semibold", "demi bold", "demibold"), 600),
        (("extra bold", "ultra bold", "extrabold", "ultrabold"), 800),
        (("black", "heavy"), 900),
        (("bold",), 700),
    ]
    for keywords, weight in weight_keywords:
        if any(keyword in combined for keyword in keywords):
            return weight
    return 400


def _infer_font_style(style_name: str, filename: str, italic_flag: bool) -> str:
    if italic_flag:
        return "italic"
    combined = _normalize_variant_text(f"{style_name} {filename}")
    return "italic" if "italic" in combined or "oblique" in combined else "normal"


def _get_name_table_value(font: TTFont, name_ids: tuple[int, ...]) -> str | None:
    """Read a localized OpenType name, preferring English Windows records."""
    if "name" not in font:
        return None

    def sort_key(record: Any) -> tuple[int, int]:
        platform_id = getattr(record, "platformID", -1)
        language_id = getattr(record, "langID", -1)
        if platform_id == 3 and language_id in (0x0409, 0):
            return (0, 0)
        if platform_id == 3:
            return (1, 0)
        if platform_id == 0:
            return (2, 0)
        return (3, language_id)

    records = font["name"].names
    for name_id in name_ids:
        for record in sorted((record for record in records if record.nameID == name_id), key=sort_key):
            try:
                value = record.toUnicode().strip()
            except Exception:
                continue
            if value:
                return value
    return None


def parse_font_bytes(filename: str, data: bytes, *, max_bytes: int) -> FontValidationResult:
    """Validate and extract metadata without persisting the supplied bytes."""
    if (
        not filename
        or Path(filename).name != filename
        or any(separator in filename for separator in ("/", "\\"))
        or any(ord(character) < 32 for character in filename)
        or filename in {".", ".."}
    ):
        raise FontValidationError("Font filename is invalid")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_FONT_EXTENSIONS:
        raise FontValidationError("Unsupported font format; use TTF, OTF, WOFF, or WOFF2")
    if len(data) == 0:
        raise FontValidationError("Font file is empty")
    if len(data) > max_bytes:
        raise FontValidationError(f"Font file exceeds the {max_bytes // (1024 * 1024)} MiB limit")

    try:
        with TTFont(file=io.BytesIO(data), lazy=False) as font:
            family = (_get_name_table_value(font, (16, 1, 4)) or "").strip()
            style_name = (_get_name_table_value(font, (17, 2)) or "").strip()
            label = (_get_name_table_value(font, (4,)) or family).strip()
            if not family:
                raise FontValidationError("Font is missing a family name")

            os2_table = font["OS/2"] if "OS/2" in font else None
            head_table = font["head"] if "head" in font else None
            post_table = font["post"] if "post" in font else None
            weight_class = getattr(os2_table, "usWeightClass", None)
            italic_flag = bool(getattr(os2_table, "fsSelection", 0) & 0x01) or bool(
                getattr(head_table, "macStyle", 0) & 0x02
            )
            if post_table is not None:
                italic_flag = italic_flag or bool(getattr(post_table, "italicAngle", 0))
            weight = _infer_font_weight(style_name, filename, weight_class)
            style = _infer_font_style(style_name, filename, italic_flag)

            axes: list[FontAxis] = []
            instances: list[FontInstance] = []
            if "fvar" in font:
                for axis in font["fvar"].axes:
                    axis_tag = str(axis.axisTag)
                    axis_label = _get_name_table_value(font, (int(axis.axisNameID),)) or axis_tag
                    axes.append(
                        FontAxis(
                            tag=axis_tag,
                            label=axis_label,
                            minimum=float(axis.minValue),
                            default=float(axis.defaultValue),
                            maximum=float(axis.maxValue),
                            hidden=bool(getattr(axis, "flags", 0) & 0x0001),
                        )
                    )
                for index, instance in enumerate(font["fvar"].instances):
                    instance_name = _get_name_table_value(font, (int(instance.subfamilyNameID),)) or (
                        f"Instance {index + 1}"
                    )
                    coordinates = {
                        axis.axisTag: float(instance.coordinates.get(axis.axisTag, axis.defaultValue))
                        for axis in font["fvar"].axes
                    }
                    instances.append(FontInstance(name=instance_name, coordinates=coordinates))
    except FontValidationError:
        raise
    except Exception as e:
        raise FontValidationError("The uploaded file is not a valid font") from e

    return FontValidationResult(
        filename=filename,
        family=family,
        label=label,
        style=style,
        weight=weight,
        content_hash=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        axes=tuple(axes),
        instances=tuple(instances),
    )


def instantiate_font_bytes(data: bytes, filename: str, coordinates: dict[str, float]) -> bytes:
    """Worker function for variable-font instancing."""
    source = io.BytesIO(data)
    with TTFont(file=source, lazy=False) as font:
        if "fvar" not in font:
            return data
        flavor = font.flavor
        # fontTools calls the coordinate mapping ``axisLimits``: values outside the
        # declared ranges are rejected by the service before this worker is submitted.
        instantiated = instantiateVariableFont(font, axisLimits=coordinates, inplace=False)
        instantiated.flavor = flavor
        output = io.BytesIO()
        try:
            instantiated.save(output)
            return output.getvalue()
        finally:
            instantiated.close()
