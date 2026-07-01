"""Tests for the custom nodes router."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from invokeai.app.api.routers.custom_nodes import (
    PACK_MANIFEST_FILENAME,
    _get_installed_packs,
    _import_workflows_from_pack,
    _load_node_pack,
    _purge_pack_modules,
    _read_pack_manifest,
    _remove_workflows_by_ids,
    _write_pack_manifest,
)
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InvocationRegistry,
)


class TestGetInstalledPacks:
    """Tests for _get_installed_packs()."""

    def test_returns_empty_when_dir_not_exists(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "nonexistent"
        with patch("invokeai.app.api.routers.custom_nodes._get_custom_nodes_path", return_value=nonexistent):
            packs = _get_installed_packs()
        assert packs == []

    def test_returns_empty_when_dir_empty(self, tmp_path: Path) -> None:
        with patch("invokeai.app.api.routers.custom_nodes._get_custom_nodes_path", return_value=tmp_path):
            packs = _get_installed_packs()
        assert packs == []

    def test_skips_files(self, tmp_path: Path) -> None:
        (tmp_path / "some_file.py").touch()
        with patch("invokeai.app.api.routers.custom_nodes._get_custom_nodes_path", return_value=tmp_path):
            packs = _get_installed_packs()
        assert packs == []

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden_pack"
        hidden.mkdir()
        (hidden / "__init__.py").touch()
        with patch("invokeai.app.api.routers.custom_nodes._get_custom_nodes_path", return_value=tmp_path):
            packs = _get_installed_packs()
        assert packs == []

    def test_skips_dirs_without_init(self, tmp_path: Path) -> None:
        no_init = tmp_path / "no_init_pack"
        no_init.mkdir()
        with patch("invokeai.app.api.routers.custom_nodes._get_custom_nodes_path", return_value=tmp_path):
            packs = _get_installed_packs()
        assert packs == []

    def test_finds_valid_pack(self, tmp_path: Path) -> None:
        pack = tmp_path / "my_pack"
        pack.mkdir()
        (pack / "__init__.py").touch()
        with patch("invokeai.app.api.routers.custom_nodes._get_custom_nodes_path", return_value=tmp_path):
            packs = _get_installed_packs()
        assert len(packs) == 1
        assert packs[0].name == "my_pack"
        assert packs[0].path == str(pack)

    def test_finds_multiple_packs_sorted(self, tmp_path: Path) -> None:
        for name in ["zebra_pack", "alpha_pack", "middle_pack"]:
            d = tmp_path / name
            d.mkdir()
            (d / "__init__.py").touch()
        with patch("invokeai.app.api.routers.custom_nodes._get_custom_nodes_path", return_value=tmp_path):
            packs = _get_installed_packs()
        assert len(packs) == 3
        assert [p.name for p in packs] == ["alpha_pack", "middle_pack", "zebra_pack"]


class TestImportWorkflowsFromPack:
    """Tests for _import_workflows_from_pack()."""

    @staticmethod
    def _mock_service_with_id(workflow_id: str = "new-id") -> MagicMock:
        """Returns a mock workflow_records service whose create() yields a DTO with the given id."""
        mock_service = MagicMock()
        mock_service.create.return_value = MagicMock(workflow_id=workflow_id)
        return mock_service

    def test_no_json_files(self, tmp_path: Path) -> None:
        (tmp_path / "__init__.py").touch()
        (tmp_path / "node.py").write_text("# node code")
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies"):
            ids = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")
        assert ids == []

    def test_skips_non_workflow_json(self, tmp_path: Path) -> None:
        # JSON without nodes/edges should be skipped
        config = {"setting": "value"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies"):
            ids = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")
        assert ids == []

    def test_imports_valid_workflow(self, tmp_path: Path) -> None:
        workflow = {
            "name": "Test Workflow",
            "author": "Test",
            "description": "A test workflow",
            "version": "1.0.0",
            "contact": "",
            "tags": "test",
            "notes": "",
            "exposedFields": [],
            "meta": {"version": "3.0.0", "category": "user"},
            "nodes": [{"id": "1", "type": "test_node"}],
            "edges": [],
        }
        workflows_dir = tmp_path / "workflows"
        workflows_dir.mkdir()
        (workflows_dir / "test_workflow.json").write_text(json.dumps(workflow))

        mock_service = self._mock_service_with_id("wf-new-1")
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            ids = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")

        assert ids == ["wf-new-1"]
        mock_service.create.assert_called_once()
        # Verify the workflow was tagged
        create_kwargs = mock_service.create.call_args.kwargs
        assert "node-pack:test_pack" in create_kwargs["workflow"].tags
        assert create_kwargs["user_id"] == "admin"
        assert create_kwargs["is_public"] is True

    def test_adds_pack_tag_to_existing_tags(self, tmp_path: Path) -> None:
        workflow = {
            "name": "Tagged Workflow",
            "author": "Test",
            "description": "",
            "version": "1.0.0",
            "contact": "",
            "tags": "existing, tags",
            "notes": "",
            "exposedFields": [],
            "meta": {"version": "3.0.0", "category": "user"},
            "nodes": [{"id": "1"}],
            "edges": [],
        }
        (tmp_path / "workflow.json").write_text(json.dumps(workflow))

        mock_service = self._mock_service_with_id()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            ids = _import_workflows_from_pack(tmp_path, "my_pack", owner_user_id="admin")

        assert len(ids) == 1
        created_workflow = mock_service.create.call_args.kwargs["workflow"]
        assert "existing, tags" in created_workflow.tags
        assert "node-pack:my_pack" in created_workflow.tags

    def test_removes_id_before_import(self, tmp_path: Path) -> None:
        workflow = {
            "id": "should-be-removed",
            "name": "Workflow With ID",
            "author": "Test",
            "description": "",
            "version": "1.0.0",
            "contact": "",
            "tags": "",
            "notes": "",
            "exposedFields": [],
            "meta": {"version": "3.0.0", "category": "user"},
            "nodes": [],
            "edges": [],
        }
        (tmp_path / "workflow.json").write_text(json.dumps(workflow))

        mock_service = self._mock_service_with_id()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            ids = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")

        assert len(ids) == 1

    def test_sets_category_to_user(self, tmp_path: Path) -> None:
        workflow = {
            "name": "Default-like Workflow",
            "author": "Test",
            "description": "",
            "version": "1.0.0",
            "contact": "",
            "tags": "",
            "notes": "",
            "exposedFields": [],
            "meta": {"version": "3.0.0", "category": "default"},
            "nodes": [],
            "edges": [],
        }
        (tmp_path / "workflow.json").write_text(json.dumps(workflow))

        mock_service = self._mock_service_with_id()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            ids = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")

        assert len(ids) == 1
        created_workflow = mock_service.create.call_args.kwargs["workflow"]
        assert created_workflow.meta.category.value == "user"

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "broken.json").write_text("{invalid json")
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies"):
            ids = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")
        assert ids == []

    def test_finds_workflows_recursively(self, tmp_path: Path) -> None:
        workflow = {
            "name": "Nested Workflow",
            "author": "Test",
            "description": "",
            "version": "1.0.0",
            "contact": "",
            "tags": "",
            "notes": "",
            "exposedFields": [],
            "meta": {"version": "3.0.0", "category": "user"},
            "nodes": [{"id": "1"}],
            "edges": [],
        }
        nested = tmp_path / "sub" / "dir"
        nested.mkdir(parents=True)
        (nested / "deep_workflow.json").write_text(json.dumps(workflow))

        mock_service = self._mock_service_with_id()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            ids = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")

        assert len(ids) == 1

    def test_skips_manifest_file(self, tmp_path: Path) -> None:
        # A manifest inside the pack must not be mistaken for a workflow during import
        (tmp_path / PACK_MANIFEST_FILENAME).write_text(json.dumps({"workflow_ids": ["wf-old"]}))
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies"):
            ids = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")
        assert ids == []


class TestPackManifest:
    """Tests for _write_pack_manifest() and _read_pack_manifest()."""

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        _write_pack_manifest(tmp_path, ["wf-1", "wf-2"])
        assert _read_pack_manifest(tmp_path) == ["wf-1", "wf-2"]

    def test_read_returns_empty_when_manifest_missing(self, tmp_path: Path) -> None:
        assert _read_pack_manifest(tmp_path) == []

    def test_read_returns_empty_when_manifest_malformed(self, tmp_path: Path) -> None:
        (tmp_path / PACK_MANIFEST_FILENAME).write_text("{not valid json")
        assert _read_pack_manifest(tmp_path) == []

    def test_read_returns_empty_when_workflow_ids_not_a_list(self, tmp_path: Path) -> None:
        (tmp_path / PACK_MANIFEST_FILENAME).write_text(json.dumps({"workflow_ids": "oops"}))
        assert _read_pack_manifest(tmp_path) == []


class TestRemoveWorkflowsByIds:
    """Tests for _remove_workflows_by_ids()."""

    def test_deletes_only_given_ids(self) -> None:
        mock_service = MagicMock()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _remove_workflows_by_ids(["wf-1", "wf-2"], "test_pack")

        assert count == 2
        assert mock_service.delete.call_count == 2
        deleted_ids = [
            call.args[0] if call.args else call.kwargs.get("workflow_id") for call in mock_service.delete.call_args_list
        ]
        assert deleted_ids == ["wf-1", "wf-2"]

    def test_returns_zero_when_no_ids(self) -> None:
        mock_service = MagicMock()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _remove_workflows_by_ids([], "empty_pack")

        assert count == 0
        mock_service.delete.assert_not_called()

    def test_continues_on_individual_delete_error(self) -> None:
        # One workflow is already gone; the helper still removes the others
        mock_service = MagicMock()
        mock_service.delete.side_effect = [Exception("not found"), None]

        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _remove_workflows_by_ids(["wf-gone", "wf-still-here"], "test_pack")

        assert count == 1

    def test_preserves_user_workflow_with_colliding_tag(self, tmp_path: Path) -> None:
        # Regression test for the data-destruction risk the reviewer raised:
        # If a user-authored workflow reuses the 'node-pack:<name>' tag, uninstall
        # must NOT delete it. The full flow is exercised here: a manifest records
        # only the pack's own workflow IDs, and _remove_workflows_by_ids operates
        # only on those — so the user's workflow (whose id is NOT in the manifest)
        # is never touched.
        pack_wf_id = "pack-wf-1"
        user_wf_id = "user-owned-wf-with-same-tag"

        _write_pack_manifest(tmp_path, [pack_wf_id])
        manifest_ids = _read_pack_manifest(tmp_path)

        mock_service = MagicMock()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            _remove_workflows_by_ids(manifest_ids, "test_pack")

        assert mock_service.delete.call_count == 1
        deleted_id = (
            mock_service.delete.call_args.args[0]
            if mock_service.delete.call_args.args
            else mock_service.delete.call_args.kwargs.get("workflow_id")
        )
        assert deleted_id == pack_wf_id
        # The user-owned workflow id is never passed to delete()
        all_delete_args = [
            (call.args[0] if call.args else call.kwargs.get("workflow_id"))
            for call in mock_service.delete.call_args_list
        ]
        assert user_wf_id not in all_delete_args


class TestUnregisterPack:
    """Tests for InvocationRegistry.unregister_pack()."""

    def test_unregister_removes_invocations(self) -> None:
        # Save original state
        original_invocations = InvocationRegistry._invocation_classes.copy()
        original_outputs = InvocationRegistry._output_classes.copy()

        try:
            # Create a mock invocation class
            mock_inv = MagicMock(spec=BaseInvocation)
            mock_inv.UIConfig = MagicMock()
            mock_inv.UIConfig.node_pack = "test_removable_pack"
            mock_inv.get_type.return_value = "test_removable_node"

            InvocationRegistry._invocation_classes.add(mock_inv)

            # Verify it's registered
            assert mock_inv in InvocationRegistry._invocation_classes

            # Unregister
            removed = InvocationRegistry.unregister_pack("test_removable_pack")

            assert "test_removable_node" in removed
            assert mock_inv not in InvocationRegistry._invocation_classes
        finally:
            # Restore original state
            InvocationRegistry._invocation_classes = original_invocations
            InvocationRegistry._output_classes = original_outputs

    def test_unregister_returns_empty_for_unknown_pack(self) -> None:
        removed = InvocationRegistry.unregister_pack("nonexistent_pack_xyz")
        assert removed == []

    def test_unregister_removes_multiple_invocations(self) -> None:
        original_invocations = InvocationRegistry._invocation_classes.copy()
        original_outputs = InvocationRegistry._output_classes.copy()

        try:
            mock_inv_1 = MagicMock(spec=BaseInvocation)
            mock_inv_1.UIConfig = MagicMock()
            mock_inv_1.UIConfig.node_pack = "multi_pack"
            mock_inv_1.get_type.return_value = "multi_node_1"

            mock_inv_2 = MagicMock(spec=BaseInvocation)
            mock_inv_2.UIConfig = MagicMock()
            mock_inv_2.UIConfig.node_pack = "multi_pack"
            mock_inv_2.get_type.return_value = "multi_node_2"

            mock_inv_other = MagicMock(spec=BaseInvocation)
            mock_inv_other.UIConfig = MagicMock()
            mock_inv_other.UIConfig.node_pack = "other_pack"
            mock_inv_other.get_type.return_value = "other_node"

            InvocationRegistry._invocation_classes.update({mock_inv_1, mock_inv_2, mock_inv_other})

            removed = InvocationRegistry.unregister_pack("multi_pack")

            assert len(removed) == 2
            assert "multi_node_1" in removed
            assert "multi_node_2" in removed
            # Other pack's node should remain
            assert mock_inv_other in InvocationRegistry._invocation_classes
        finally:
            InvocationRegistry._invocation_classes = original_invocations
            InvocationRegistry._output_classes = original_outputs


class TestPurgePackModules:
    """Tests for _purge_pack_modules() — clears the pack subtree from sys.modules."""

    def test_removes_root_module(self) -> None:
        sys.modules["purge_test_root"] = MagicMock()
        try:
            removed = _purge_pack_modules("purge_test_root")
            assert "purge_test_root" in removed
            assert "purge_test_root" not in sys.modules
        finally:
            sys.modules.pop("purge_test_root", None)

    def test_removes_submodules(self) -> None:
        sys.modules["purge_test_pack"] = MagicMock()
        sys.modules["purge_test_pack.nodes"] = MagicMock()
        sys.modules["purge_test_pack.utils.helpers"] = MagicMock()
        try:
            removed = _purge_pack_modules("purge_test_pack")
            assert set(removed) == {
                "purge_test_pack",
                "purge_test_pack.nodes",
                "purge_test_pack.utils.helpers",
            }
            assert "purge_test_pack" not in sys.modules
            assert "purge_test_pack.nodes" not in sys.modules
            assert "purge_test_pack.utils.helpers" not in sys.modules
        finally:
            for key in ("purge_test_pack", "purge_test_pack.nodes", "purge_test_pack.utils.helpers"):
                sys.modules.pop(key, None)

    def test_does_not_remove_unrelated_modules_with_prefix_collision(self) -> None:
        # "foo_pack_extra" must NOT be removed when purging "foo_pack"
        sys.modules["foo_pack"] = MagicMock()
        sys.modules["foo_pack_extra"] = MagicMock()
        sys.modules["foo_pack.sub"] = MagicMock()
        try:
            removed = _purge_pack_modules("foo_pack")
            assert set(removed) == {"foo_pack", "foo_pack.sub"}
            assert "foo_pack_extra" in sys.modules
        finally:
            for key in ("foo_pack", "foo_pack_extra", "foo_pack.sub"):
                sys.modules.pop(key, None)

    def test_noop_when_pack_not_loaded(self) -> None:
        removed = _purge_pack_modules("never_loaded_pack_xyz")
        assert removed == []


class TestUninstallReinstallReloadsSubmodules:
    """Regression test for the uninstall -> reinstall cache bug.

    Before the fix, uninstall only cleared sys.modules[pack_name] and left
    submodules cached. On reinstall, Python reused the cached submodules,
    their @invocation decorators never re-ran, and the pack loaded with
    zero registered nodes until a full process restart.
    """

    def test_reinstall_re_executes_submodule(self, tmp_path: Path) -> None:
        pack_name = "reinstall_regression_pack"
        pack_dir = tmp_path / pack_name
        pack_dir.mkdir()

        # __init__.py imports from a submodule — this is the shape that triggered the bug
        (pack_dir / "__init__.py").write_text("from .nodes import *  # noqa: F401,F403\n")
        submodule = pack_dir / "nodes.py"

        # Each import of the submodule must append a marker to this file.
        # If the submodule gets reused from sys.modules instead of re-executed,
        # the second install won't produce a second marker.
        marker_file = tmp_path / "exec_markers.txt"
        submodule.write_text(
            f"from pathlib import Path\nPath(r'{marker_file.as_posix()}').open('a').write('exec\\n')\n"
        )

        try:
            # First install
            _load_node_pack(pack_name, pack_dir)
            assert pack_name in sys.modules
            assert f"{pack_name}.nodes" in sys.modules
            assert marker_file.read_text().count("exec") == 1

            # Simulate uninstall's module cleanup
            _purge_pack_modules(pack_name)
            assert pack_name not in sys.modules
            assert f"{pack_name}.nodes" not in sys.modules

            # Reinstall — submodule MUST re-execute
            _load_node_pack(pack_name, pack_dir)
            assert marker_file.read_text().count("exec") == 2, (
                "Submodule was not re-executed on reinstall — the @invocation "
                "decorators would not have re-registered the pack's nodes."
            )
        finally:
            _purge_pack_modules(pack_name)
