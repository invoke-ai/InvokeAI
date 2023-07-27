"""
Widget class definitions used by model_select.py, merge_diffusers.py and textual_inversion.py
"""
import curses
import math
import os
import platform
import pyperclip
import struct
import subprocess
import sys
import npyscreen
import textwrap
import npyscreen.wgmultiline as wgmultiline
from npyscreen import fmPopup
from shutil import get_terminal_size
from curses import BUTTON2_CLICKED, BUTTON3_CLICKED

# minimum size for UIs
MIN_COLS = 130
MIN_LINES = 38


# -------------------------------------
def set_terminal_size(columns: int, lines: int):
    ts = get_terminal_size()
    width = max(columns, ts.columns)
    height = max(lines, ts.lines)

    OS = platform.uname().system
    if OS == "Windows":
        pass
        # not working reliably - ask user to adjust the window
        # _set_terminal_size_powershell(width,height)
    elif OS in ["Darwin", "Linux"]:
        _set_terminal_size_unix(width, height)

    # check whether it worked....
    ts = get_terminal_size()
    pause = False
    if ts.columns < columns:
        print("\033[1mThis window is too narrow for the user interface.\033[0m")
        pause = True
    if ts.lines < lines:
        print("\033[1mThis window is too short for the user interface.\033[0m")
        pause = True
    if pause:
        input("Maximize the window then press any key to continue..")


def _set_terminal_size_powershell(width: int, height: int):
    script = f"""
$pshost = get-host
$pswindow = $pshost.ui.rawui
$newsize = $pswindow.buffersize
$newsize.height = 3000
$newsize.width = {width}
$pswindow.buffersize = $newsize
$newsize = $pswindow.windowsize
$newsize.height = {height}
$newsize.width = {width}
$pswindow.windowsize = $newsize
"""
    subprocess.run(["powershell", "-Command", "-"], input=script, text=True)


def _set_terminal_size_unix(width: int, height: int):
    import fcntl
    import termios

    # These terminals accept the size command and report that the
    # size changed, but they lie!!!
    for bad_terminal in ["TERMINATOR_UUID", "ALACRITTY_WINDOW_ID"]:
        if os.environ.get(bad_terminal):
            return

    winsize = struct.pack("HHHH", height, width, 0, 0)
    fcntl.ioctl(sys.stdout.fileno(), termios.TIOCSWINSZ, winsize)
    sys.stdout.write("\x1b[8;{height};{width}t".format(height=height, width=width))
    sys.stdout.flush()


def set_min_terminal_size(min_cols: int, min_lines: int):
    # make sure there's enough room for the ui
    term_cols, term_lines = get_terminal_size()
    if term_cols >= min_cols and term_lines >= min_lines:
        return
    cols = max(term_cols, min_cols)
    lines = max(term_lines, min_lines)
    set_terminal_size(cols, lines)


class IntSlider(npyscreen.Slider):
    def translate_value(self):
        stri = "%2d / %2d" % (self.value, self.out_of)
        l = (len(str(self.out_of))) * 2 + 4
        stri = stri.rjust(l)
        return stri


# -------------------------------------
# fix npyscreen form so that cursor wraps both forward and backward
class CyclingForm(object):
    def find_previous_editable(self, *args):
        done = False
        n = self.editw - 1
        while not done:
            if self._widgets__[n].editable and not self._widgets__[n].hidden:
                self.editw = n
                done = True
            n -= 1
            if n < 0:
                if self.cycle_widgets:
                    n = len(self._widgets__) - 1
                else:
                    done = True


# -------------------------------------
class CenteredTitleText(npyscreen.TitleText):
    def __init__(self, *args, **keywords):
        super().__init__(*args, **keywords)
        self.resize()

    def resize(self):
        super().resize()
        maxy, maxx = self.parent.curses_pad.getmaxyx()
        label = self.name
        self.relx = (maxx - len(label)) // 2


# -------------------------------------
class CenteredButtonPress(npyscreen.ButtonPress):
    def resize(self):
        super().resize()
        maxy, maxx = self.parent.curses_pad.getmaxyx()
        label = self.name
        self.relx = (maxx - len(label)) // 2


# -------------------------------------
class OffsetButtonPress(npyscreen.ButtonPress):
    def __init__(self, screen, offset=0, *args, **keywords):
        super().__init__(screen, *args, **keywords)
        self.offset = offset

    def resize(self):
        maxy, maxx = self.parent.curses_pad.getmaxyx()
        width = len(self.name)
        self.relx = self.offset + (maxx - width) // 2


class IntTitleSlider(npyscreen.TitleText):
    _entry_type = IntSlider


class FloatSlider(npyscreen.Slider):
    # this is supposed to adjust display precision, but doesn't
    def translate_value(self):
        stri = "%3.2f / %3.2f" % (self.value, self.out_of)
        l = (len(str(self.out_of))) * 2 + 4
        stri = stri.rjust(l)
        return stri


class FloatTitleSlider(npyscreen.TitleText):
    _entry_type = FloatSlider


class SelectColumnBase:
    def make_contained_widgets(self):
        self._my_widgets = []
        column_width = self.width // self.columns
        for h in range(self.value_cnt):
            self._my_widgets.append(
                self._contained_widgets(
                    self.parent,
                    rely=self.rely + (h % self.rows) * self._contained_widget_height,
                    relx=self.relx + (h // self.rows) * column_width,
                    max_width=column_width,
                    max_height=self.__class__._contained_widget_height,
                )
            )

    def set_up_handlers(self):
        super().set_up_handlers()
        self.handlers.update(
            {
                curses.KEY_UP: self.h_cursor_line_left,
                curses.KEY_DOWN: self.h_cursor_line_right,
            }
        )

    def h_cursor_line_down(self, ch):
        self.cursor_line += self.rows
        if self.cursor_line >= len(self.values):
            if self.scroll_exit:
                self.cursor_line = len(self.values) - self.rows
                self.h_exit_down(ch)
                return True
            else:
                self.cursor_line -= self.rows
                return True

    def h_cursor_line_up(self, ch):
        self.cursor_line -= self.rows
        if self.cursor_line < 0:
            if self.scroll_exit:
                self.cursor_line = 0
                self.h_exit_up(ch)
            else:
                self.cursor_line = 0

    def h_cursor_line_left(self, ch):
        super().h_cursor_line_up(ch)

    def h_cursor_line_right(self, ch):
        super().h_cursor_line_down(ch)

    def handle_mouse_event(self, mouse_event):
        mouse_id, rel_x, rel_y, z, bstate = self.interpret_mouse_event(mouse_event)
        column_width = self.width // self.columns
        column_height = math.ceil(self.value_cnt / self.columns)
        column_no = rel_x // column_width
        row_no = rel_y // self._contained_widget_height
        self.cursor_line = column_no * column_height + row_no
        if bstate & curses.BUTTON1_DOUBLE_CLICKED:
            if hasattr(self, "on_mouse_double_click"):
                self.on_mouse_double_click(self.cursor_line)
        self.display()


class MultiSelectColumns(SelectColumnBase, npyscreen.MultiSelect):
    def __init__(self, screen, columns: int = 1, values: list = [], **keywords):
        self.columns = columns
        self.value_cnt = len(values)
        self.rows = math.ceil(self.value_cnt / self.columns)
        super().__init__(screen, values=values, **keywords)

    def on_mouse_double_click(self, cursor_line):
        self.h_select_toggle(cursor_line)


class SingleSelectWithChanged(npyscreen.SelectOne):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def h_select(self, ch):
        super().h_select(ch)
        if self.on_changed:
            self.on_changed(self.value)


class SingleSelectColumns(SelectColumnBase, SingleSelectWithChanged):
    def __init__(self, screen, columns: int = 1, values: list = [], **keywords):
        self.columns = columns
        self.value_cnt = len(values)
        self.rows = math.ceil(self.value_cnt / self.columns)
        self.on_changed = None
        super().__init__(screen, values=values, **keywords)

    def when_value_edited(self):
        self.h_select(self.cursor_line)

    def when_cursor_moved(self):
        self.h_select(self.cursor_line)

    def h_cursor_line_right(self, ch):
        self.h_exit_down("bye bye")


class TextBoxInner(npyscreen.MultiLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yank = None
        self.handlers.update(
            {
                "^A": self.h_cursor_to_start,
                "^E": self.h_cursor_to_end,
                "^K": self.h_kill,
                "^F": self.h_cursor_right,
                "^B": self.h_cursor_left,
                "^Y": self.h_yank,
                "^V": self.h_paste,
            }
        )

    def h_cursor_to_start(self, input):
        self.cursor_position = 0

    def h_cursor_to_end(self, input):
        self.cursor_position = len(self.value)

    def h_kill(self, input):
        self.yank = self.value[self.cursor_position :]
        self.value = self.value[: self.cursor_position]

    def h_yank(self, input):
        if self.yank:
            self.paste(self.yank)

    def paste(self, text: str):
        self.value = self.value[: self.cursor_position] + text + self.value[self.cursor_position :]
        self.cursor_position += len(text)

    def h_paste(self, input: int = 0):
        try:
            text = pyperclip.paste()
        except ModuleNotFoundError:
            text = "To paste with the mouse on Linux, please install the 'xclip' program."
        self.paste(text)

    def handle_mouse_event(self, mouse_event):
        mouse_id, rel_x, rel_y, z, bstate = self.interpret_mouse_event(mouse_event)
        if bstate & (BUTTON2_CLICKED | BUTTON3_CLICKED):
            self.h_paste()

    # def update(self, clear=True):
    #     if clear:
    #         self.clear()

    #     HEIGHT = self.height
    #     WIDTH = self.width
    #     # draw box.
    #     self.parent.curses_pad.hline(self.rely, self.relx, curses.ACS_HLINE, WIDTH)
    #     self.parent.curses_pad.hline(
    #         self.rely + HEIGHT, self.relx, curses.ACS_HLINE, WIDTH
    #     )
    #     self.parent.curses_pad.vline(
    #         self.rely, self.relx, curses.ACS_VLINE, self.height
    #     )
    #     self.parent.curses_pad.vline(
    #         self.rely, self.relx + WIDTH, curses.ACS_VLINE, HEIGHT
    #     )

    # # draw corners
    # self.parent.curses_pad.addch(
    #     self.rely,
    #     self.relx,
    #     curses.ACS_ULCORNER,
    # )
    # self.parent.curses_pad.addch(
    #     self.rely,
    #     self.relx + WIDTH,
    #     curses.ACS_URCORNER,
    # )
    # self.parent.curses_pad.addch(
    #     self.rely + HEIGHT,
    #     self.relx,
    #     curses.ACS_LLCORNER,
    # )
    # self.parent.curses_pad.addch(
    #     self.rely + HEIGHT,
    #     self.relx + WIDTH,
    #     curses.ACS_LRCORNER,
    # )

    # # fool our superclass into thinking drawing area is smaller - this is really hacky but it seems to work
    # (relx, rely, height, width) = (self.relx, self.rely, self.height, self.width)
    # self.relx += 1
    # self.rely += 1
    # self.height -= 1
    # self.width -= 1
    # super().update(clear=False)
    # (self.relx, self.rely, self.height, self.width) = (relx, rely, height, width)


class TextBox(npyscreen.BoxTitle):
    _contained_widget = TextBoxInner


class BufferBox(npyscreen.BoxTitle):
    _contained_widget = npyscreen.BufferPager


class ConfirmCancelPopup(fmPopup.ActionPopup):
    DEFAULT_COLUMNS = 100

    def on_ok(self):
        self.value = True

    def on_cancel(self):
        self.value = False


class FileBox(npyscreen.BoxTitle):
    _contained_widget = npyscreen.Filename


class PrettyTextBox(npyscreen.BoxTitle):
    _contained_widget = TextBox


def _wrap_message_lines(message, line_length):
    lines = []
    for line in message.split("\n"):
        lines.extend(textwrap.wrap(line.rstrip(), line_length))
    return lines


def _prepare_message(message):
    if isinstance(message, list) or isinstance(message, tuple):
        return "\n".join([s.rstrip() for s in message])
        # return "\n".join(message)
    else:
        return message


def select_stable_diffusion_config_file(
    form_color: str = "DANGER",
    wrap: bool = True,
    model_name: str = "Unknown",
):
    message = f"Please select the correct base model for the V2 checkpoint named '{model_name}'. Press <CANCEL> to skip installation."
    title = "CONFIG FILE SELECTION"
    options = [
        "An SD v2.x base model (512 pixels; no 'parameterization:' line in its yaml file)",
        "An SD v2.x v-predictive model (768 pixels; 'parameterization: \"v\"' line in its yaml file)",
        "Skip installation for now and come back later",
    ]

    F = ConfirmCancelPopup(
        name=title,
        color=form_color,
        cycle_widgets=True,
        lines=16,
    )
    F.preserve_selected_widget = True

    mlw = F.add(
        wgmultiline.Pager,
        max_height=4,
        editable=False,
    )
    mlw_width = mlw.width - 1
    if wrap:
        message = _wrap_message_lines(message, mlw_width)
    mlw.values = message

    choice = F.add(
        npyscreen.SelectOne,
        values=options,
        value=[0],
        max_height=len(options) + 1,
        scroll_exit=True,
    )

    F.editw = 1
    F.edit()
    if not F.value:
        return None
    assert choice.value[0] in range(0, 3), "invalid choice"
    choices = ["epsilon", "v", "abort"]
    return choices[choice.value[0]]
