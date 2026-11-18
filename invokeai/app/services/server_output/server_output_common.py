from typing import Literal

from pydantic import BaseModel, Field

# A cell style: (fg, bg, bold, dim, italic, underline, inverse). `None` stands for the default style, so unstyled text
# carries no run. Colors are a palette index "0".."255" (0-15 are the named ANSI colors) or "#rrggbb".
Style = tuple[str | None, str | None, bool, bool, bool, bool, bool]
ResetReason = Literal["initial", "restarted", "expired"]


class ServerOutputStyleRun(BaseModel):
    """A styled span of a row's text. Text outside every run has the default style."""

    offset: int = Field(description="Start of the span, in UTF-16 code units of the row's text")
    length: int = Field(description="Length of the span, in UTF-16 code units")
    fg: str | None = Field(
        default=None, description='Foreground color: a palette index "0".."255" (0-15 are ANSI colors) or "#rrggbb"'
    )
    bg: str | None = Field(default=None, description="Background color, in the same form as fg")
    bold: bool = Field(default=False)
    dim: bool = Field(default=False)
    italic: bool = Field(default=False)
    underline: bool = Field(default=False)
    inverse: bool = Field(default=False, description="Swap foreground and background")


class ServerOutputRow(BaseModel):
    """One display row of server console output, at its latest value."""

    id: int = Field(description="Stable row ID; IDs increase in display order")
    text: str = Field(description="The row's text, without control sequences")
    styles: list[ServerOutputStyleRun] = Field(description="Styled spans in text order, non-overlapping")
    truncated: bool = Field(description="Text past the server's row length limit was dropped")


class ServerOutputRead(BaseModel):
    """One read of the server console transcript.

    When `reset` is set, the client replaces everything it holds with `rows`. Otherwise it applies `rows` as inserts
    or replacements by ID and drops held rows with an ID below `floor_id`. Either way it sends `cursor` with its next
    read, at once while `has_more` is true.
    """

    run_id: str = Field(description="Identifies this server process run")
    cursor: str = Field(description="Opaque position to pass to the next read")
    reset: ResetReason | None = Field(
        description=(
            "Why `rows` is a replacement snapshot: the read had no cursor (initial), the cursor came from an earlier "
            "server run (restarted), or the cursor's history is gone or too far behind to replay (expired). "
            "Null when `rows` are changes to apply."
        )
    )
    rows: list[ServerOutputRow] = Field(description="Rows in display order")
    floor_id: int = Field(description="Rows with a lower ID are no longer part of the transcript")
    has_more: bool = Field(description="More changes are pending; read again with `cursor`")
    snapshot_truncated: bool = Field(
        description="Earlier output of this run exists or existed but is not in this snapshot"
    )


class InvalidServerOutputCursorError(ValueError):
    """The cursor is not one this server could have issued. The message never echoes the cursor."""

    def __init__(self) -> None:
        super().__init__("Invalid server output cursor")
