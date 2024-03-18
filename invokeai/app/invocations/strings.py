# 2023 skunkworxdark (https://github.com/skunkworxdark)

import re

from invokeai.app.services.shared.invocation_context import InvocationContext

from .baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    invocation,
    invocation_output,
)
from .fields import InputField, OutputField, UIComponent
from .primitives import StringOutput


@invocation_output("string_pos_neg_output")
class StringPosNegOutput(BaseInvocationOutput):
    """Base class for invocations that output a positive and negative string"""

    positive_string: str = OutputField(description="Positive string")
    negative_string: str = OutputField(description="Negative string")


@invocation(
    "string_split_neg",
    title="String Split Negative",
    tags=["string", "split", "negative"],
    category="string",
    version="1.0.0",
)
class StringSplitNegInvocation(BaseInvocation):
    """Splits string into two strings, inside [] goes into negative string everthing else goes into positive string. Each [ and ] character is replaced with a space"""

    string: str = InputField(default="", description="String to split", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringPosNegOutput:
        p_string = ""
        n_string = ""
        brackets_depth = 0
        escaped = False

        for char in self.string or "":
            if char == "[" and not escaped:
                n_string += " "
                brackets_depth += 1
            elif char == "]" and not escaped:
                brackets_depth -= 1
                char = " "
            elif brackets_depth > 0:
                n_string += char
            else:
                p_string += char

            # keep track of the escape char but only if it isn't escaped already
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


@invocation("string_split", title="String Split", tags=["string", "split"], category="string", version="1.0.0")
class StringSplitInvocation(BaseInvocation):
    """Splits string into two strings, based on the first occurance of the delimiter. The delimiter will be removed from the string"""

    string: str = InputField(default="", description="String to split", ui_component=UIComponent.Textarea)
    delimiter: str = InputField(
        default="", description="Delimiter to spilt with. blank will split on the first whitespace"
    )

    def invoke(self, context: InvocationContext) -> String2Output:
        result = self.string.split(self.delimiter, 1)
        if len(result) == 2:
            part1, part2 = result
        else:
            part1 = result[0]
            part2 = ""

        return String2Output(string_1=part1, string_2=part2)


@invocation("string_join", title="String Join", tags=["string", "join"], category="string", version="1.0.0")
class StringJoinInvocation(BaseInvocation):
    """Joins string left to string right"""

    string_left: str = InputField(default="", description="String Left", ui_component=UIComponent.Textarea)
    string_right: str = InputField(default="", description="String Right", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=((self.string_left or "") + (self.string_right or "")))


@invocation("string_join_three", title="String Join Three", tags=["string", "join"], category="string", version="1.0.0")
class StringJoinThreeInvocation(BaseInvocation):
    """Joins string left to string middle to string right"""

    string_left: str = InputField(default="", description="String Left", ui_component=UIComponent.Textarea)
    string_middle: str = InputField(default="", description="String Middle", ui_component=UIComponent.Textarea)
    string_right: str = InputField(default="", description="String Right", ui_component=UIComponent.Textarea)

    def invoke(self, context: InvocationContext) -> StringOutput:
        return StringOutput(value=((self.string_left or "") + (self.string_middle or "") + (self.string_right or "")))


@invocation(
    "string_replace", title="String Replace", tags=["string", "replace", "regex"], category="string", version="1.0.0"
)
class StringReplaceInvocation(BaseInvocation):
    """Replaces the search string with the replace string"""

    string: str = InputField(default="", description="String to work on", ui_component=UIComponent.Textarea)
    search_string: str = InputField(default="", description="String to search for", ui_component=UIComponent.Textarea)
    replace_string: str = InputField(
        default="", description="String to replace the search", ui_component=UIComponent.Textarea
    )
    use_regex: bool = InputField(
        default=False, description="Use search string as a regex expression (non regex is case insensitive)"
    )

    def invoke(self, context: InvocationContext) -> StringOutput:
        pattern = self.search_string or ""
        new_string = self.string or ""
        if len(pattern) > 0:
            if not self.use_regex:
                # None regex so make case insensitve
                pattern = "(?i)" + re.escape(pattern)
            new_string = re.sub(pattern, (self.replace_string or ""), new_string)
        return StringOutput(value=new_string)
