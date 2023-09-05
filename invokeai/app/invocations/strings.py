# 2023 skunkworxdark (https://github.com/skunkworxdark)

from typing import Literal, Union, Optional
from os.path import exists
import re
import numpy as np

from pydantic import validator

from invokeai.app.invocations.primitives import (
    StringOutput,
    StringCollectionOutput,
)

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    Input,
    OutputField,
    InvocationContext,
    UIComponent,
    UIType,
    invocation,
    invocation_output,
)

@invocation("strings_from_file", title="Strings from File", tags=["string", "file"], category="string")
class StringsFromFileInvocation(BaseInvocation):
    """Loads strings from a text file"""

    file_path: str = InputField(description="Path to string text file")
    pre_string: Optional[str] = InputField(
        default=None, description="String to prepend to each string", ui_component=UIComponent.Textarea
    )
    post_string: Optional[str] = InputField(
        default=None, description="String to append to each string", ui_component=UIComponent.Textarea
    )
    start_line: int = InputField(default=1, ge=1, description="Line in the file to start start from")
    max_lines: int = InputField(default=1, ge=0, description="Max lines to read from file (0=all)")

    @validator("file_path")
    def file_path_exists(cls, v):
        if not exists(v):
            raise ValueError(FileNotFoundError)
        return v

    def stringsFromFile(
        self,
        file_path: str,
        pre_string: Union[str, None],
        post_string: Union[str, None],
        start_line: int,
        max_lines: int,
    ):
        strings = []
        start_line -= 1
        end_line = start_line + max_lines
        if max_lines <= 0:
            end_line = np.iinfo(np.int32).max
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= start_line and i < end_line:
                    strings.append((pre_string or "") + line.strip() + (post_string or ""))
                if i >= end_line:
                    break
        return strings

    def invoke(self, context: InvocationContext) -> StringCollectionOutput:
        strings = self.stringsFromFile(
            self.file_path, self.pre_string, self.post_string, self.start_line, self.max_lines
        )
        return StringCollectionOutput(collection=strings)


@invocation_output("strings_to_file_output")
class StringsToFileInvocationOutput(BaseInvocationOutput):
    """Base class for invocation that writes to a file and returns nothing of use"""

@invocation("string_to_file", title="Strings to File", tags=["string", "file"], category="string")
class StringsToFileInvocation(BaseInvocation):
    '''Save strings to a text file'''

    file_path: str = InputField(description="Path to text file")
    strings: Union[str, list[str], None] = InputField(default=None, description="String or collection of strings to write", ui_type=UIType.Collection)
    append: bool = InputField(default=True, description="Append or overwrite file")

    def invoke(self, context: InvocationContext) -> StringsToFileInvocationOutput:
        with open(self.file_path, 'a' if self.append else 'w') as f:
            if isinstance(self.strings, list):
                for line in (self.strings):
                    f.write ( line + '\n' )
            else:
                f.write((self.strings or '') + '\n')

        return StringsToFileInvocationOutput()

@invocation_output("string_pos_neg_output")
class StringPosNegOutput(BaseInvocationOutput):
    """Base class for invocations that output a positive and negative string"""

    positive_string: str = OutputField(description="Positive string")
    negative_string: str = OutputField(description="Negative string")

@invocation("string_split_neg", title="String Spilt Negative", tags=["string", "split", "negative"], category="string")
class StringSplitNegInvocation(BaseInvocation):
    """Splits string into two strings, inside [] goes into negative string everthing else goes into positive string. Each [ and ] character is replaced with a space"""

    string: str = InputField(default='', description="String to split", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringPosNegOutput:
        p_string = ""
        n_string = ""
        brackets_depth = 0
        escaped = False

        for char in (self.string or ''):
            if char == "[" and not escaped:
                n_string += ' '
                brackets_depth += 1 
            elif char == "]" and not escaped:
                brackets_depth -= 1 
                char = ' ' 
            elif brackets_depth > 0:
                n_string += char
            else:
                p_string += char

            #keep track of the escape char but only if it isn't escaped already
            if char == "\\" and not escaped:
                escaped = True
            else:
                escaped = False

        return StringPosNegOutput(positive_string=p_string, negative_string=n_string)


@invocation_output("string_2_output")
class String2Output(BaseInvocationOutput):
    """Base class for invocations that output two strings"""

    string_1: str = OutputField(description="string 1")
    string_2: str = OutputField(description="string 2")

@invocation("string_split", title="String Spilt", tags=["string", "split"], category="string")
class StringSplitInvocation(BaseInvocation):
    """Splits string into two strings, based on the first occurance of the delimiter. The delimiter will be removed from the string"""

    string: str = InputField(default='', description="String to split", ui_component=UIComponent.Textarea)
    delimiter: str = InputField(default='', description="Delimiter to spilt with. blank will split on the first whitespace")

    def invoke(self, context: InvocationContext) -> String2Output:
        result = self.string.split(self.delimiter, 1)
        if len(result = 2):
            part1, part2 = result
        else:
            part1=result[0]
            part2=''

        return String2Output(string_1=part1, string_2=part2)


@invocation("string_join", title="String Join", tags=["string", "join"], category="string")
class StringJoinInvocation(BaseInvocation):
    """Joins string left to string right"""

    string_left: str = InputField(default='', description="String Left", ui_component=UIComponent.Textarea)
    string_right: str = InputField(default='', description="String Right", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=((self.string_left or '') + (self.string_right or '')))  


@invocation("string_join_three", title="String Join Three", tags=["string", "join"], category="string")
class StringJoinThreeInvocation(BaseInvocation):
    """Joins string left to string middle to string right"""

    string_left: str = InputField(default='', description="String Left", ui_component=UIComponent.Textarea)
    string_middle: str = InputField(default='', description="String Middle", ui_component=UIComponent.Textarea)
    string_right: str = InputField(default='', description="String Right", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=((self.string_left or '') + (self.string_middle or '') + (self.string_right or '')))  


@invocation("string_replace", title="String Replace", tags=["string", "replace", "regex"], category="string")
class StringReplaceInvocation(BaseInvocation):
    """Replaces the search string with the replace string"""

    string: str = InputField(default='', description="String to work on", ui_component=UIComponent.Textarea)
    search_string : str = InputField(default='', description="String to search for", ui_component=UIComponent.Textarea)
    replace_string : str = InputField(default='', description="String to replace the search", ui_component=UIComponent.Textarea)
    use_regex: bool = InputField(default=False, description="Use search string as a regex expression (non regex is case insensitive)")

    def invoke(self, context: InvocationContext) -> StringOutput:
        pattern = (self.search_string or '')
        new_string = (self.string or '')
        if len(pattern) > 0: 
            if not self.use_regex:
                #None regex so make case insensitve
                pattern = "(?i)" + re.escape(pattern)
            new_string = re.sub(pattern, (self.replace_string or ''), new_string)
        return StringOutput(value=new_string)  


@invocation("string_weight", title="String Weight", tags=["string", "weight"], category="string")
class StringWeightInvocation(BaseInvocation):
    """Takes a string and weight and outputs a new string in the format of a compel style (string)weight"""

    string: str = InputField(default='', description="String to work on", ui_component=UIComponent.Textarea)
    weight : float = InputField(default=1, gt=0, description="weight of the string")

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=f"({self.string}){self.weight}")


COMBINE_TYPE = Literal[".and", ".blend"]

@invocation("string_weights_to_combine", title="String Weights to Combine", tags=["string", "combine", "and", "blend"], category="string")
class StringWeightsToCombineInvocation(BaseInvocation):
    """Takes a collection of (string)weight strings and converts it into a combined .and() or .blend() structure. Blank strings are ignored"""

    string_weights: list[str] = InputField(default=[''], description="String weights to combine", ui_type=UIType.Collection)
    combine_type: COMBINE_TYPE = InputField(default=".and", description="Combine type .and() or .blend()", input=Input.Direct)

    def invoke(self, context: InvocationContext) -> StringOutput:
        strings = []
        numbers = []
        for item in self.string_weights:
            string, number = item.rsplit(')', 1)
            string = string[1:].strip()
            number = float(number)
            if len(string)>0:
                strings.append(f'"{string}"')
                numbers.append(number)
        return StringOutput(value=f'({",".join(strings)}){self.combine_type}({",".join(map(str, numbers))})')
