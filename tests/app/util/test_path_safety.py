import pytest

from invokeai.app.util.misc import uuid_string
from invokeai.app.util.path_safety import is_plain_filename

# Every one of these, joined onto a storage directory, resolves outside it - or, for the empty/dot cases,
# resolves to the directory itself rather than a file in it. The backslash and drive-letter shapes only escape on
# Windows, but they must be rejected on posix too: otherwise a posix-only test run reports all clear on a payload
# that works against Windows deployments.
UNSAFE_NAMES = [
    "..",
    ".",
    "",
    "../escaped",
    "../../../../etc/passwd",
    "/etc/passwd",
    "//srv/share/file",
    "sub/nested",
    "foo/../../bar",
    "trailing/",
    "..\\escaped",
    "sub\\nested",
    "C:\\Windows\\win.ini",
    "C:relative",
    "\\\\?\\C:\\Windows\\win.ini",
    "nul\x00byte",
]

SAFE_NAMES = [
    "Tensor_ecd3b3a5-6c4f-4a5f-9a0e-4b1c2d3e4f50",
    "ConditioningFieldData_ecd3b3a5-6c4f-4a5f-9a0e-4b1c2d3e4f50",
    "openai-dall-e-3",
    "image.png",
    "..leading.dots",
    "trailing.dots..",
    "~tilde",
    "spaces are fine",
]


@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_is_plain_filename_rejects_traversal(name: str):
    assert is_plain_filename(name) is False


@pytest.mark.parametrize("name", SAFE_NAMES)
def test_is_plain_filename_accepts_plain_names(name: str):
    assert is_plain_filename(name) is True


def test_is_plain_filename_accepts_generated_names():
    """The predicate must never reject a name the server itself hands out."""
    assert is_plain_filename(uuid_string()) is True
    assert is_plain_filename(f"Tensor_{uuid_string()}") is True


@pytest.mark.parametrize("name", [".. ", "..  ", "...", ".. .", "  ", ". "])
def test_is_plain_filename_rejects_windows_normalised_dot_names(name: str):
    """`pathlib` keeps trailing dots and spaces, but the Win32 normaliser strips them from the final component at
    open time - so on Windows `dir / ".. "` is the parent directory."""
    assert is_plain_filename(name) is False


@pytest.mark.parametrize("name", ["a\rb", "a\nb", "a\tb", "a\x00b", "a\x7fb", "a\x1bb"])
def test_is_plain_filename_rejects_control_characters(name: str):
    assert is_plain_filename(name) is False


@pytest.mark.parametrize("name", ["foo..", "foo.", "a.b.c", "..leading"])
def test_is_plain_filename_keeps_names_that_only_look_like_dot_names(name: str):
    """The trailing-dot rule must not swallow ordinary names that happen to end in a dot."""
    assert is_plain_filename(name) is True
