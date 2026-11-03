"""Tests for PromptsFromFileInvocation.

Regression coverage for GHSA-xwjh-4cmx-mqjh: `prompt_from_file` took an unconstrained `file_path`,
opened it in the queue worker and returned every line to whoever enqueued the graph, so any account
with a session could read any UTF-8 file the service account could read.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.prompt import PromptsFromFileInvocation
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache


def _make_context(
    prompt_files_path: Path,
    multiuser: bool = False,
    user_id: str = "system",
    is_admin: bool = True,
    user_exists: bool = True,
) -> MagicMock:
    context = MagicMock()
    config = MagicMock()
    config.multiuser = multiuser
    config.prompt_files_path = prompt_files_path
    context._services.configuration = config
    context._data.queue_item.user_id = user_id
    if user_exists:
        user = MagicMock()
        user.is_admin = is_admin
        context._services.users.get.return_value = user
    else:
        context._services.users.get.return_value = None
    return context


def _build_node(**overrides) -> PromptsFromFileInvocation:
    defaults = {"id": "test", "file_path": "prompts.txt"}
    defaults.update(overrides)
    return PromptsFromFileInvocation(**defaults)


@pytest.fixture
def prompt_files(tmp_path: Path) -> Path:
    root = tmp_path / "prompt_files"
    root.mkdir()
    (root / "prompts.txt").write_text("a red cat\na blue dog\na green bird\n", encoding="utf-8")
    (root / "lists").mkdir()
    (root / "lists" / "fantasy.txt").write_text("a dragon\na wizard\n", encoding="utf-8")
    return root


class TestPromptsFromFileReadsPermittedFiles:
    def test_reads_file_in_root(self, prompt_files):
        ctx = _make_context(prompt_files)
        out = _build_node(max_prompts=0).invoke(ctx)
        assert out.collection == ["a red cat", "a blue dog", "a green bird"]

    def test_reads_file_in_subdirectory(self, prompt_files):
        ctx = _make_context(prompt_files)
        out = _build_node(file_path="lists/fantasy.txt", max_prompts=0).invoke(ctx)
        assert out.collection == ["a dragon", "a wizard"]

    def test_respects_start_line_and_max_prompts(self, prompt_files):
        ctx = _make_context(prompt_files)
        out = _build_node(start_line=2, max_prompts=1).invoke(ctx)
        assert out.collection == ["a blue dog"]

    def test_applies_pre_and_post_prompt(self, prompt_files):
        ctx = _make_context(prompt_files)
        out = _build_node(max_prompts=1, pre_prompt="masterpiece, ", post_prompt=", 4k").invoke(ctx)
        assert out.collection == ["masterpiece, a red cat, 4k"]


class TestPromptsFromFileRefusesEscape:
    # The advisory's payloads, plus the spellings that only escape on Windows.
    @pytest.mark.parametrize(
        "file_path",
        [
            "/etc/passwd",
            "/proc/self/environ",
            "../../../../etc/passwd",
            "../outside.txt",
            "lists/../../outside.txt",
            "C:\\Windows\\win.ini",
            "..\\outside.txt",
            "\\\\server\\share\\f.txt",
            "",
        ],
    )
    def test_rejects_path_outside_root(self, prompt_files, file_path):
        (prompt_files.parent / "outside.txt").write_text("SECRET\n", encoding="utf-8")
        ctx = _make_context(prompt_files)
        with pytest.raises(ValueError) as exc:
            _build_node(file_path=file_path, max_prompts=0).invoke(ctx)
        # Names the setting so an upgrading operator knows where files now live, without the resolved path.
        assert "prompt_files_dir" in str(exc.value)
        assert str(prompt_files) not in str(exc.value)

    def test_rejects_symlink_escaping_root(self, prompt_files):
        secret = prompt_files.parent / "secret.txt"
        secret.write_text("SECRET\n", encoding="utf-8")
        (prompt_files / "escape.txt").symlink_to(secret)

        ctx = _make_context(prompt_files)
        with pytest.raises(ValueError, match="stay within"):
            _build_node(file_path="escape.txt", max_prompts=0).invoke(ctx)

    def test_etc_passwd_contents_never_returned(self, prompt_files):
        """The advisory's exact primitive: the response must not carry the file's contents."""
        ctx = _make_context(prompt_files)
        with pytest.raises(ValueError) as exc:
            _build_node(file_path="/etc/passwd", max_prompts=0).invoke(ctx)
        assert "root:x:0:0" not in str(exc.value)


class TestPromptsFromFileExistenceIsNotAnOracle:
    def test_no_field_validator_rejects_at_request_validation(self):
        """Constructing the node must not touch the filesystem.

        The original `file_path_exists` validator ran while the enqueue body was parsed, so a
        missing path was answered as a 422 before anything was queued - a filesystem probe that
        left no queue item behind. Construction must now succeed regardless of the path.
        """
        node = _build_node(file_path="definitely/does/not/exist.txt")
        assert node.file_path == "definitely/does/not/exist.txt"
        # An absolute path must also survive construction - it is refused at invoke time, as a
        # failure on the caller's own queue item.
        assert _build_node(file_path="/etc/passwd").file_path == "/etc/passwd"

    def test_missing_file_fails_at_invoke(self, prompt_files):
        ctx = _make_context(prompt_files)
        with pytest.raises(ValueError, match=r"No prompt file 'nope\.txt' in .*prompt_files_dir"):
            _build_node(file_path="nope.txt").invoke(ctx)

    def test_missing_file_error_does_not_leak_absolute_path(self, prompt_files):
        ctx = _make_context(prompt_files)
        with pytest.raises(ValueError) as exc:
            _build_node(file_path="nope.txt").invoke(ctx)
        assert str(prompt_files) not in str(exc.value)

    def test_directory_is_not_readable_as_a_file(self, prompt_files):
        ctx = _make_context(prompt_files)
        with pytest.raises(ValueError, match="No prompt file"):
            _build_node(file_path="lists").invoke(ctx)


class TestPromptsFromFileMultiuserAuthorization:
    def test_non_admin_is_refused(self, prompt_files):
        ctx = _make_context(prompt_files, multiuser=True, user_id="attacker", is_admin=False)
        with pytest.raises(ValueError, match="restricted to administrators"):
            _build_node(max_prompts=0).invoke(ctx)

    def test_unknown_user_is_refused(self, prompt_files):
        """Fail closed when the queueing account cannot be resolved."""
        ctx = _make_context(prompt_files, multiuser=True, user_id="ghost", user_exists=False)
        with pytest.raises(ValueError, match="restricted to administrators"):
            _build_node(max_prompts=0).invoke(ctx)

    def test_admin_is_allowed(self, prompt_files):
        ctx = _make_context(prompt_files, multiuser=True, user_id="admin", is_admin=True)
        out = _build_node(max_prompts=0).invoke(ctx)
        assert out.collection == ["a red cat", "a blue dog", "a green bird"]

    def test_authorization_is_checked_against_the_queueing_user(self, prompt_files):
        ctx = _make_context(prompt_files, multiuser=True, user_id="user-123", is_admin=True)
        _build_node(max_prompts=0).invoke(ctx)
        ctx._services.users.get.assert_called_once_with("user-123")

    def test_single_user_mode_does_not_require_a_user_lookup(self, prompt_files):
        ctx = _make_context(prompt_files, multiuser=False)
        _build_node(max_prompts=0).invoke(ctx)
        ctx._services.users.get.assert_not_called()

    @pytest.mark.parametrize("file_path", ["nope.txt", "escape.txt", "../outside.txt"])
    def test_non_admin_refused_before_the_file_is_opened(self, prompt_files, file_path):
        """The gate must precede path resolution, so a non-admin cannot even probe for existence or containment."""
        outside = prompt_files.parent / "outside.txt"
        outside.write_text("SECRET\n", encoding="utf-8")
        (prompt_files / "escape.txt").symlink_to(outside)
        ctx = _make_context(prompt_files, multiuser=True, user_id="attacker", is_admin=False)
        with pytest.raises(ValueError, match="restricted to administrators"):
            _build_node(file_path=file_path).invoke(ctx)


class TestPromptsFromFileBypassesInvocationCache:
    def test_admin_warmed_cache_entry_is_not_served_to_non_admin(self, prompt_files):
        """Cache keys carry no user identity, so a cache hit would skip the admin gate in `invoke()`."""
        ctx = _make_context(prompt_files, multiuser=True, user_id="admin", is_admin=True)
        ctx._skip_invocation_cache = False
        services = ctx._services
        services.configuration.node_cache_size = 10
        services.invocation_cache = MemoryInvocationCache(max_cache_size=10)

        # Clients choose `use_cache` per node, whatever the decorator default.
        node = _build_node(max_prompts=0, use_cache=True)
        admin_out = node.invoke_internal_with_effects(ctx, services).output
        assert admin_out.collection == ["a red cat", "a blue dog", "a green bird"]

        services.users.get.return_value.is_admin = False
        with pytest.raises(ValueError, match="restricted to administrators"):
            _build_node(max_prompts=0, use_cache=True).invoke_internal_with_effects(ctx, services)
