'''
Widget class definitions used by model_select.py, merge_diffusers.py and textual_inversion.py
'''
import math
import npyscreen
import curses

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
