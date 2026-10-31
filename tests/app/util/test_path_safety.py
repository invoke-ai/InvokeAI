import pytest

from invokeai.app.util import path_safety
from invokeai.app.util.misc import uuid_string
from invokeai.app.util.path_safety import is_contained_filename, is_plain_filename

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


# Legal posix filenames that `is_plain_filename` refuses. Older versions stored model keys unchecked, so data can
# already exist under these names on a posix install.
LEGACY_POSIX_NAMES = [
    "legacy:key",
    "legacy?key",
    "legacy*key",
    'a"b',
    "a|b",
    "<a>",
    "a\\b",
    "NUL",
    "COM1",
    "...",
    "... ",
    "..:stream",
    "C:relative",
]
POSIX_ESCAPING_NAMES = ["", ".", "..", "../escaped", "/etc/passwd", "sub/nested", "trailing/", "nul\x00byte", "a\nb"]


@pytest.mark.parametrize("name", LEGACY_POSIX_NAMES)
def test_is_contained_filename_keeps_legacy_posix_names_on_posix(monkeypatch: pytest.MonkeyPatch, name: str):
    monkeypatch.setattr(path_safety, "_ON_WINDOWS", False)
    assert not is_plain_filename(name)
    assert is_contained_filename(name)


@pytest.mark.parametrize("name", POSIX_ESCAPING_NAMES)
def test_is_contained_filename_rejects_posix_escapes_on_posix(monkeypatch: pytest.MonkeyPatch, name: str):
    monkeypatch.setattr(path_safety, "_ON_WINDOWS", False)
    assert not is_contained_filename(name)


@pytest.mark.parametrize("name", UNSAFE_NAMES + LEGACY_POSIX_NAMES + SAFE_NAMES)
def test_is_contained_filename_is_is_plain_filename_on_windows(monkeypatch: pytest.MonkeyPatch, name: str):
    monkeypatch.setattr(path_safety, "_ON_WINDOWS", True)
    assert is_contained_filename(name) == is_plain_filename(name)


# Relative paths that must never resolve inside the root. Separately from the single-name cases above, these are
# the shapes a *path*-taking input receives: traversal, absolute and drive-qualified payloads, and the backslash
# spellings that only escape on Windows but must be refused on posix too so a posix test run cannot report all
# clear on a payload that works against Windows deployments.
UNCONTAINED_PATHS = [
    "..",
    "../secret.txt",
    "../../../../etc/passwd",
    "sub/../../secret.txt",
    "/etc/passwd",
    "/proc/self/environ",
    "//srv/share/file",
    "..\\secret.txt",
    "sub\\..\\..\\secret.txt",
    "C:\\Windows\\win.ini",
    "C:relative",
    "\\\\?\\C:\\Windows\\win.ini",
    "\\\\server\\share\\f.txt",
    "",
    "nul\x00byte.txt",
    "sub/nul\x00byte.txt",
]

CONTAINED_PATHS = [
    "prompts.txt",
    "./prompts.txt",
    "lists/fantasy.txt",
    "lists/sub/deep.txt",
    # Backslashes are read as separators, so this addresses the same file as the forward-slash spelling.
    "lists\\fantasy.txt",
    "..leading.dots.txt",
]


class TestResolveContainedPath:
    @pytest.mark.parametrize("candidate", UNCONTAINED_PATHS)
    def test_rejects_uncontained(self, tmp_path, candidate):
        root = tmp_path / "root"
        root.mkdir()
        with pytest.raises(ValueError):
            path_safety.resolve_contained_path(root, candidate)

    @pytest.mark.parametrize("candidate", CONTAINED_PATHS)
    def test_accepts_contained(self, tmp_path, candidate):
        root = tmp_path / "root"
        root.mkdir()
        resolved = path_safety.resolve_contained_path(root, candidate)
        assert resolved == root.resolve() or root.resolve() in resolved.parents

    # Legal posix filenames an operator may already have on disk; the portable is_plain_filename would refuse them.
    @pytest.mark.parametrize("candidate", ["aux.txt", "notes:1.txt", "lists/aux.txt"])
    def test_accepts_existing_posix_names_on_posix(self, tmp_path, monkeypatch: pytest.MonkeyPatch, candidate):
        monkeypatch.setattr(path_safety, "_ON_WINDOWS", False)
        root = tmp_path / "root"
        root.mkdir()
        assert path_safety.resolve_contained_path(root, candidate) == (root / candidate).resolve()

    def test_does_not_require_existence(self, tmp_path):
        """Callers that create the directory (save_image_to_file) resolve before it exists."""
        root = tmp_path / "root"
        root.mkdir()
        assert path_safety.resolve_contained_path(root, "not/yet/there") == (root / "not/yet/there").resolve()

    def test_rejects_symlink_escaping_root(self, tmp_path):
        """The check no amount of string inspection can make: a symlink inside the root pointing out of it."""
        root = tmp_path / "root"
        root.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("SECRET")
        (root / "escape.txt").symlink_to(secret)

        with pytest.raises(ValueError, match="stay within"):
            path_safety.resolve_contained_path(root, "escape.txt")

    def test_rejects_symlinked_directory_escaping_root(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("SECRET")
        (root / "link").symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match="stay within"):
            path_safety.resolve_contained_path(root, "link/secret.txt")

    def test_allows_symlink_staying_inside_root(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        (root / "real.txt").write_text("ok")
        (root / "alias.txt").symlink_to(root / "real.txt")

        assert path_safety.resolve_contained_path(root, "alias.txt") == (root / "real.txt").resolve()

    def test_root_itself_is_contained(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        assert path_safety.resolve_contained_path(root, ".") == root.resolve()

    def test_sibling_directory_sharing_a_prefix_is_rejected(self, tmp_path):
        """`<tmp>/root-evil` must not pass as contained by `<tmp>/root` on a string prefix match."""
        root = tmp_path / "root"
        root.mkdir()
        (tmp_path / "root-evil").mkdir()
        with pytest.raises(ValueError):
            path_safety.resolve_contained_path(root, "../root-evil/f.txt")
