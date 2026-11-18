"""Incremental decoding of terminal output into screen operations.

The decoder keeps only a small, bounded state between writes (the parse state, at most `MAX_CSI_LENGTH` characters
of a pending control sequence, and the current pen), so escape sequences may be split across writes. It understands
what console output in this process uses: newline, carriage return, backspace, tab, cursor up/down/left/right and
column (`CSI A/B/C/D/G`, `ESC M`), erase in line (`CSI K`), and SGR colors and attributes. Everything else is
consumed and ignored, including OSC/DCS/SOS/PM/APC strings (window titles, OSC 52 clipboard writes, OSC 8 links), so
none of it can reach a reader. The 8-bit introducers (U+009B CSI, U+009D OSC, U+0090 DCS, U+0098 SOS, U+009E PM,
U+009F APC) and terminator (U+009C ST) act like their `ESC` forms; other C1 controls are ignored.

Deliberate differences from a terminal:
- `\\n` also returns to column 0, as a terminal does under the usual `onlcr` setting.
- A control string also ends at a newline, so a stray unterminated `ESC ]` cannot hide the rest of the transcript.
- A C0 control inside a CSI sequence aborts the sequence instead of executing within it.
- SGR colon subparameters (`38:2::r:g:b`) are ignored as a whole.
- Every code point occupies one cell; wide and combining characters are not measured.
"""

import re
from typing import Protocol

from invokeai.app.services.server_output.server_output_common import Style

MAX_CSI_LENGTH = 64
# Larger movement and column parameters are clipped to this; the screen clips further to its own bounds.
MAX_PARAMETER = 65535

# Everything the ground state does not print: C0, DEL, C1, and lone surrogates (printed as U+FFFD so the text stays
# encodable as UTF-8).
_CONTROL = re.compile("[\x00-\x1f\x7f-\x9f\ud800-\udfff]")
# CSI parameter and intermediate bytes.
_CSI_BODY = re.compile("[\x20-\x3f]*")
# BEL, LF, CAN, SUB, ESC, ST.
_STRING_END = re.compile("[\x07\x0a\x18\x1a\x1b\x9c]")
# 8-bit string introducers: DCS, SOS, OSC, PM, APC.
_C1_STRINGS = "\x90\x98\x9d\x9e\x9f"
_CSI_PARAMETERS = re.compile("[0-9;]*")

_GROUND, _ESC, _ESC_INTERMEDIATE, _CSI, _STRING, _STRING_ESC = range(6)


class TerminalScreen(Protocol):
    """Screen operations the decoder drives. Counts are at least 1."""

    def put_text(self, text: str, start: int, end: int, style: Style | None) -> None:
        """Write `text[start:end]` (no control characters) at the cursor, overwriting cells, and advance."""

    def line_feed(self) -> None: ...

    def carriage_return(self) -> None: ...

    def cursor_up(self, count: int) -> None: ...

    def cursor_down(self, count: int) -> None: ...

    def cursor_forward(self, count: int) -> None: ...

    def cursor_back(self, count: int) -> None: ...

    def cursor_column(self, column: int) -> None:
        """Move to a 0-based column of the current row."""

    def tab(self) -> None: ...

    def erase_line(self, mode: int) -> None:
        """0: cursor to end; 1: start through cursor; 2: whole row."""


def _palette_color(index: int) -> str:
    return str(min(index, 255))


class TerminalDecoder:
    """Decodes one output stream's text. Not thread-safe: the owner serializes `feed` calls."""

    __slots__ = (
        "_screen",
        "_state",
        "_csi",
        "_csi_overflow",
        "_fg",
        "_bg",
        "_bold",
        "_dim",
        "_italic",
        "_underline",
        "_inverse",
        "_style",
    )

    def __init__(self, screen: TerminalScreen) -> None:
        self._screen = screen
        self._state = _GROUND
        self._csi = ""
        self._csi_overflow = False
        self._reset_pen()

    def _reset_pen(self) -> None:
        self._fg: str | None = None
        self._bg: str | None = None
        self._bold = False
        self._dim = False
        self._italic = False
        self._underline = False
        self._inverse = False
        self._style: Style | None = None

    def feed(self, text: str) -> None:
        screen = self._screen
        i = 0
        n = len(text)
        while i < n:
            state = self._state
            if state == _GROUND:
                match = _CONTROL.search(text, i)
                j = match.start() if match else n
                if j > i:
                    screen.put_text(text, i, j, self._style)
                if match is None:
                    return
                i = j + 1
                self._control(text[j])
            elif state == _CSI:
                body = _CSI_BODY.match(text, i)
                assert body is not None  # the pattern matches the empty string
                if body.end() > i:
                    if self._csi_overflow or len(self._csi) + body.end() - i > MAX_CSI_LENGTH:
                        self._csi_overflow = True
                    else:
                        self._csi += body.group()
                    i = body.end()
                if i == n:
                    return
                final = text[i]
                self._state = _GROUND
                if "\x40" <= final <= "\x7e":
                    i += 1
                    if not self._csi_overflow:
                        self._dispatch_csi(final)
                # Anything else aborts the sequence and is processed in the ground state.
            elif state == _STRING:
                match = _STRING_END.search(text, i)
                if match is None:
                    return
                end = match.group()
                if end == "\n":
                    self._state = _GROUND
                    i = match.start()
                else:
                    self._state = _STRING_ESC if end == "\x1b" else _GROUND
                    i = match.end()
            elif state == _STRING_ESC:
                if text[i] == "\\":
                    self._state = _GROUND
                    i += 1
                else:
                    self._state = _ESC
            elif state == _ESC:
                self._escape(text[i])
                if self._state != _GROUND or text[i] >= "\x20":
                    i += 1
            else:  # _ESC_INTERMEDIATE
                c = text[i]
                if "\x20" <= c <= "\x2f":
                    i += 1
                elif "\x30" <= c <= "\x7e":
                    self._state = _GROUND
                    i += 1
                else:
                    self._state = _GROUND

    def _control(self, c: str) -> None:
        screen = self._screen
        if c == "\n":
            screen.line_feed()
        elif c == "\r":
            screen.carriage_return()
        elif c == "\x1b":
            self._state = _ESC
        elif c == "\x9b":
            self._start_csi()
        elif c in _C1_STRINGS:
            self._state = _STRING
        elif c == "\b":
            screen.cursor_back(1)
        elif c == "\t":
            screen.tab()
        elif "\ud800" <= c <= "\udfff":
            screen.put_text("�", 0, 1, self._style)
        # Other C0, DEL and C1 controls (a stray ST among them) are ignored.

    def _start_csi(self) -> None:
        self._state = _CSI
        self._csi = ""
        self._csi_overflow = False

    def _escape(self, c: str) -> None:
        """Handle the character after ESC. The caller consumes it unless it is a control that aborted the escape."""
        if c == "[":
            self._start_csi()
        elif c in "]PX^_":
            self._state = _STRING
        elif c == "\x1b":
            pass
        elif c < "\x20":
            self._state = _GROUND
        elif c <= "\x2f":
            self._state = _ESC_INTERMEDIATE
        else:
            self._state = _GROUND
            if c == "M":
                self._screen.cursor_up(1)
            # Other single-character escapes (save/restore cursor, keypad modes, reset) are ignored.

    def _dispatch_csi(self, final: str) -> None:
        body = self._csi
        if _CSI_PARAMETERS.fullmatch(body) is None:
            # Private (`?25l`), intermediate, or colon-subparameter sequences.
            return
        params = [min(int(p), MAX_PARAMETER) if p else 0 for p in body.split(";")]
        screen = self._screen
        if final == "m":
            self._sgr(params)
            return
        count = max(params[0], 1)
        if final == "A":
            screen.cursor_up(count)
        elif final == "B":
            screen.cursor_down(count)
        elif final == "C":
            screen.cursor_forward(count)
        elif final == "D":
            screen.cursor_back(count)
        elif final == "G":
            screen.cursor_column(count - 1)
        elif final == "K":
            if params[0] <= 2:
                screen.erase_line(params[0])

    def _sgr(self, params: list[int]) -> None:
        i = 0
        n = len(params)
        while i < n:
            p = params[i]
            i += 1
            if p == 0:
                self._reset_pen()
            elif p == 1:
                self._bold = True
            elif p == 2:
                self._dim = True
            elif p == 3:
                self._italic = True
            elif p == 4:
                self._underline = True
            elif p == 7:
                self._inverse = True
            elif p == 22:
                self._bold = self._dim = False
            elif p == 23:
                self._italic = False
            elif p == 24:
                self._underline = False
            elif p == 27:
                self._inverse = False
            elif 30 <= p <= 37:
                self._fg = _palette_color(p - 30)
            elif 90 <= p <= 97:
                self._fg = _palette_color(p - 90 + 8)
            elif p == 39:
                self._fg = None
            elif 40 <= p <= 47:
                self._bg = _palette_color(p - 40)
            elif 100 <= p <= 107:
                self._bg = _palette_color(p - 100 + 8)
            elif p == 49:
                self._bg = None
            elif p == 38 or p == 48:
                color, i = _extended_color(params, i)
                if color is not None:
                    if p == 38:
                        self._fg = color
                    else:
                        self._bg = color
            # Other attributes (blink, conceal, strike, fonts) are ignored.
        style: Style = (self._fg, self._bg, self._bold, self._dim, self._italic, self._underline, self._inverse)
        self._style = None if style == (None, None, False, False, False, False, False) else style


def _extended_color(params: list[int], i: int) -> tuple[str | None, int]:
    """Parse the parameters after 38/48 starting at `i`. An unknown form consumes nothing more, as terminals do with
    InvokeAI's own `38;20` (rendered as the default color)."""
    if i < len(params):
        kind = params[i]
        if kind == 5 and i + 1 < len(params):
            return _palette_color(params[i + 1]), i + 2
        if kind == 2 and i + 3 < len(params):
            r, g, b = (min(v, 255) for v in params[i + 1 : i + 4])
            return f"#{r:02x}{g:02x}{b:02x}", i + 4
    return None, i
