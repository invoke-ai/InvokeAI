from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.strings import StringReplaceInvocation, StringSplitInvocation


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


@pytest.mark.parametrize(
    ("string", "search", "replace", "expected"),
    [
        ("path is DIR", "dir", r"C:\new\folder", r"path is C:\new\folder"),
        ("cost: X", "x", r"\1 dollars", r"cost: \1 dollars"),
        ("a+b", "+", "\\\\", "a\\\\b"),
        ("Cat cat CAT", "cat", "dog", "dog dog dog"),
    ],
)
def test_string_replace_plain_text_inserts_the_replacement_literally(
    string: str, search: str, replace: str, expected: str
) -> None:
    output = StringReplaceInvocation(string=string, search_string=search, replace_string=replace).invoke(MagicMock())

    assert output.value == expected


def test_string_replace_regex_mode_still_expands_group_references() -> None:
    output = StringReplaceInvocation(
        string="a1b2", search_string=r"(\d)", replace_string=r"<\1>", use_regex=True
    ).invoke(MagicMock())

    assert output.value == "a<1>b<2>"
