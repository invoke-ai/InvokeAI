"""Compile the local desktop palette into a bounded, value-only stylesheet for frontend HTML."""

import os
import re
import stat
import tomllib
from pathlib import Path

DESKTOP_THEME_MARKER = '<meta name="invokeai-desktop-theme" content="" />'
_MAX_PALETTE_BYTES = 16_384
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}\Z")


def _luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
    return sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))


def _contrast(first: str, second: str) -> float:
    light, dark = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def _blend(background: str, foreground: str, foreground_weight: float) -> str:
    channels = []
    for offset in (1, 3, 5):
        back = int(background[offset : offset + 2], 16)
        front = int(foreground[offset : offset + 2], 16)
        channels.append(round(back + (front - back) * foreground_weight))
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def read_desktop_theme_style() -> str:
    """Read Omarchy's active palette afresh. Unavailable or invalid palettes leave browser colors in charge."""
    try:
        state_home = os.environ.get("XDG_STATE_HOME")
        root = Path(state_home) if state_home else Path.home() / ".local" / "state"
        if not root.is_absolute():
            return ""
        path = root / "omarchy" / "current" / "theme" / "colors.toml"
        # The fixed path may contain Omarchy's theme symlink. Nonblocking open prevents a FIFO from hanging a request.
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_PALETTE_BYTES:
                return ""
            data = source.read(_MAX_PALETTE_BYTES + 1)
        if len(data) > _MAX_PALETTE_BYTES:
            return ""
        palette = tomllib.loads(data.decode("utf-8"))
        mode = palette.get("mode")
        if mode not in ("light", "dark"):
            return ""
        colors = [palette.get(key) for key in ("background", "foreground", "accent")]
        if any(not isinstance(color, str) or not _HEX_COLOR.fullmatch(color) for color in colors):
            return ""
        background, foreground, accent = (color.lower() for color in colors)
        if _contrast(background, foreground) < 4.5:
            return ""
        accent_contrast = next(
            (color for color in (foreground, background) if _contrast(accent, color) >= 4.5),
            max(("#000000", "#ffffff"), key=lambda color: _contrast(accent, color)),
        )
        values = {
            "background": background,
            "foreground": foreground,
            "accent": accent,
            "accent-contrast": accent_contrast,
            "color-scheme": mode,
        }
        for role, source_key, fallback in (
            ("surface", "lighter_background", _blend(background, foreground, 0.06)),
            ("inset", "dark_background", background),
            ("selection", "selection", _blend(background, foreground, 0.10)),
        ):
            supplied = palette.get(source_key)
            if supplied is not None and (not isinstance(supplied, str) or not _HEX_COLOR.fullmatch(supplied)):
                return ""
            color = supplied.lower() if supplied is not None else fallback
            if _contrast(color, foreground) < 4.5:
                color = fallback if _contrast(fallback, foreground) >= 4.5 else background
            values[role] = color
        declarations = ";".join(f"--invokeai-desktop-{key}:{value}" for key, value in values.items())
        # A valid native palette already handles dark/light mode; extension recoloring would corrupt both.
        return f'<meta name="darkreader-lock"><style data-invokeai-desktop-theme>:root{{{declarations}}}</style>'
    except (OSError, RuntimeError, UnicodeError, tomllib.TOMLDecodeError):
        return ""


if __name__ == "__main__":
    # Vite's optional local preview invokes this file directly, without importing the Python application or models.
    print(read_desktop_theme_style(), end="")
