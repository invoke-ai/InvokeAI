from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.strings import StringSplitInvocation


def test_string_split_blank_delimiter_splits_on_first_whitespace() -> None:
    output = StringSplitInvocation(string="a cat\tsitting on a mat").invoke(MagicMock())

    assert output.string_1 == "a"
    assert output.string_2 == "cat\tsitting on a mat"


def test_string_split_blank_delimiter_without_whitespace_returns_whole_string() -> None:
    output = StringSplitInvocation(string="cat", delimiter="").invoke(MagicMock())

    assert output.string_1 == "cat"
    assert output.string_2 == ""


@pytest.mark.parametrize("string", ["", " \t\n"])
def test_string_split_blank_delimiter_with_empty_or_whitespace_string_returns_empty_strings(string: str) -> None:
    output = StringSplitInvocation(string=string, delimiter="").invoke(MagicMock())

    assert output.string_1 == ""
    assert output.string_2 == ""


def test_string_split_splits_on_first_occurrence_of_delimiter() -> None:
    output = StringSplitInvocation(string="a, b, c", delimiter=", ").invoke(MagicMock())

    assert output.string_1 == "a"
    assert output.string_2 == "b, c"
