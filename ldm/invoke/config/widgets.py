'''
Widget class definitions used by model_select.py, merge_diffusers.py and textual_inversion.py
'''
import math
import platform
import npyscreen
import os
import sys
import curses
import struct

from shutil import get_terminal_size

# -------------------------------------
def set_terminal_size(columns: int, lines: int):
    OS = platform.uname().system
    if OS=="Windows":
        os.system(f'mode con: cols={columns} lines={lines}')
    elif OS in ['Darwin', 'Linux']:
        import termios
        import fcntl
        winsize = struct.pack("HHHH", lines, columns, 0, 0)
        fcntl.ioctl(sys.stdout.fileno(), termios.TIOCSWINSZ, winsize)
        sys.stdout.write("\x1b[8;{rows};{cols}t".format(rows=lines, cols=columns))
        sys.stdout.flush()

def set_min_terminal_size(min_cols: int, min_lines: int):
    # make sure there's enough room for the ui
    term_cols, term_lines = get_terminal_size()
    cols  = max(term_cols, min_cols)
    lines = max(term_lines, min_lines)
    set_terminal_size(cols,lines)

class IntSlider(npyscreen.Slider):
    def translate_value(self):
        stri = "%2d / %2d" % (self.value, self.out_of)
        l = (len(str(self.out_of))) * 2 + 4
        stri = stri.rjust(l)
        return stri

# -------------------------------------
class CenteredTitleText(npyscreen.TitleText):
    def __init__(self,*args,**keywords):
        super().__init__(*args,**keywords)
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
    def __init__(self, screen, offset=0, *args,  **keywords):
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

class MultiSelectColumns(npyscreen.MultiSelect):
    def __init__(self, screen, columns: int=1, values: list=[], **keywords):
        self.columns = columns
        self.value_cnt = len(values)
        self.rows = math.ceil(self.value_cnt / self.columns)
        super().__init__(screen,values=values, **keywords)

    def make_contained_widgets(self):
        self._my_widgets = []
        column_width = self.width // self.columns
        for h in range(self.value_cnt):
            self._my_widgets.append(
                self._contained_widgets(self.parent, 
                                        rely=self.rely + (h % self.rows) * self._contained_widget_height,
                                        relx=self.relx + (h // self.rows) * column_width,
                                        max_width=column_width,
                                        max_height=self.__class__._contained_widget_height,
                                        )
            )

    def set_up_handlers(self):
        super().set_up_handlers()
        self.handlers.update({
            curses.KEY_UP:    self.h_cursor_line_left,
            curses.KEY_DOWN:  self.h_cursor_line_right,
        }
                             )
    def h_cursor_line_down(self, ch):
        self.cursor_line += self.rows
        if self.cursor_line >= len(self.values):
            if self.scroll_exit: 
                self.cursor_line = len(self.values)-self.rows
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

    def h_cursor_line_left(self,ch):
        super().h_cursor_line_up(ch)
        
    def h_cursor_line_right(self,ch):
        super().h_cursor_line_down(ch)

class TextBox(npyscreen.MultiLineEdit):
    def update(self, clear=True):
        if clear: self.clear()

        HEIGHT = self.height
        WIDTH  = self.width
        # draw box.
        self.parent.curses_pad.hline(self.rely, self.relx, curses.ACS_HLINE, WIDTH)
        self.parent.curses_pad.hline(self.rely + HEIGHT, self.relx, curses.ACS_HLINE, WIDTH)
        self.parent.curses_pad.vline(self.rely, self.relx, curses.ACS_VLINE, self.height)
        self.parent.curses_pad.vline(self.rely, self.relx+WIDTH, curses.ACS_VLINE, HEIGHT)
        
        # draw corners
        self.parent.curses_pad.addch(self.rely, self.relx, curses.ACS_ULCORNER, )
        self.parent.curses_pad.addch(self.rely, self.relx+WIDTH, curses.ACS_URCORNER, )
        self.parent.curses_pad.addch(self.rely+HEIGHT, self.relx, curses.ACS_LLCORNER, )
        self.parent.curses_pad.addch(self.rely+HEIGHT, self.relx+WIDTH, curses.ACS_LRCORNER, )
        
        # fool our superclass into thinking drawing area is smaller - this is really hacky but it seems to work
        (relx,rely,height,width) = (self.relx, self.rely, self.height, self.width)
        self.relx += 1
        self.rely += 1
        self.height -= 1
        self.width -= 1
        super().update(clear=False)
        (self.relx,self.rely,self.height,self.width) = (relx, rely, height, width)
