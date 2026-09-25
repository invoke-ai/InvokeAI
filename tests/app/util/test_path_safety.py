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
    # NTFS alternate data streams reaching the PARENT: `<name>:<stream>` names a stream on `<name>`, so a `..`
    # before the colon addresses the store's parent directory. `PureWindowsPath` only reads a single *letter*
    # followed by a colon as a drive, so it hands these back as one bare component and cannot catch them.
    "..:stream",
    "..:stream.webp",
    "..:$DATA",
    "..::$INDEX_ALLOCATION",
    "..:",
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


@pytest.mark.parametrize(
    "name",
    ["a\rb", "a\nb", "a\tb", "a\x00b", "a\x7fb", "a\x1bb", "a\x85b", "a\u2028b", "a\u2029b", "a\x80b", "a\x9fb"],
)
def test_is_plain_filename_rejects_control_characters(name: str):
    """`str.splitlines()` splits on U+0085, U+2028 and U+2029 as well as the C0 set, and routes interpolate these
    names into log records raw, so a name carrying one forges a second log line."""
    assert is_plain_filename(name) is False


@pytest.mark.parametrize("name", ["foo:bar", "foo::$DATA", ":stream", "...:s"])
def test_is_plain_filename_rejects_contained_alternate_data_streams(name: str):
    """These name a stream on a file inside the store, or on the store directory itself, rather than escaping it.
    They are rejected because a stream is not the file the caller asked for - not because they traverse."""
    assert is_plain_filename(name) is False


@pytest.mark.parametrize("name", ["foo<bar", "foo>bar", 'foo"bar', "foo|bar", "foo?bar", "foo*bar"])
def test_is_plain_filename_rejects_characters_windows_forbids(name: str):
    """These cannot escape a directory on their own, but a name that is not a valid filename on every platform we
    support is not one we should be joining onto a directory."""
    assert is_plain_filename(name) is False


@pytest.mark.parametrize(
    "name",
    [
        "NUL",
        "nul",
        "CON",
        "PRN",
        "AUX",
        "COM1",
        "LPT9",
        "NUL.webp",
        "CON.txt",
        "NUL.",
        "NUL ",
        "CONIN$",
        # Win32 trims trailing spaces from the base name before the device lookup, so the space between the stem
        # and the extension does not save these.
        "NUL .webp",
        "CON  .txt",
        # `COM`/`LPT` are reserved with the superscript digits and with 0, not just ASCII 1-9.
        "COM¹",
        "COM²",
        "COM³",
        "LPT¹",
        "COM0",
        "LPT0",
        # Legacy DOS devices, still reserved on some builds.
        "CLOCK$",
        "CONFIG$",
        "KEYBD$",
        "SCREEN$",
    ],
)
def test_is_plain_filename_rejects_windows_device_names(name: str):
    """Win32 resolves these to the device namespace from any directory, so the join does not address a file in the
    store at all - a write to `<store>/NUL.webp` is silently discarded and `.exists()` still reports True."""
    assert is_plain_filename(name) is False


@pytest.mark.parametrize("name", ["console", "context", "nullish", "comic", "con-model", "NULL", "com10", "aux-enc"])
def test_is_plain_filename_keeps_names_that_merely_start_like_a_device(name: str):
    """The device rule matches the whole stem, so it must not swallow ordinary names with the same prefix."""
    assert is_plain_filename(name) is True


@pytest.mark.parametrize("name", ["foo..", "foo.", "a.b.c", "..leading"])
def test_is_plain_filename_keeps_names_that_only_look_like_dot_names(name: str):
    """The trailing-dot rule must not swallow ordinary names that happen to end in a dot."""
    assert is_plain_filename(name) is True
