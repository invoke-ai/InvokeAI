"""Tests for the custom nodes router."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from invokeai.app.api.routers.custom_nodes import (
    _get_installed_packs,
    _import_workflows_from_pack,
    _remove_workflows_for_pack,
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

    def test_no_json_files(self, tmp_path: Path) -> None:
        (tmp_path / "__init__.py").touch()
        (tmp_path / "node.py").write_text("# node code")
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies"):
            count = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")
        assert count == 0

    def test_skips_non_workflow_json(self, tmp_path: Path) -> None:
        # JSON without nodes/edges should be skipped
        config = {"setting": "value"}
        (tmp_path / "config.json").write_text(json.dumps(config))
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies"):
            count = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")
        assert count == 0

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

        mock_service = MagicMock()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")

        assert count == 1
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

        mock_service = MagicMock()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _import_workflows_from_pack(tmp_path, "my_pack", owner_user_id="admin")

        assert count == 1
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

        mock_service = MagicMock()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")

        assert count == 1

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

        mock_service = MagicMock()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")

        assert count == 1
        created_workflow = mock_service.create.call_args.kwargs["workflow"]
        assert created_workflow.meta.category.value == "user"

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "broken.json").write_text("{invalid json")
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies"):
            count = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")
        assert count == 0

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

        mock_service = MagicMock()
        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _import_workflows_from_pack(tmp_path, "test_pack", owner_user_id="admin")

        assert count == 1


class TestRemoveWorkflowsForPack:
    """Tests for _remove_workflows_for_pack()."""

    def test_removes_matching_workflows(self) -> None:
        mock_item_1 = MagicMock()
        mock_item_1.workflow_id = "wf-1"
        mock_item_1.name = "Workflow 1"
        mock_item_2 = MagicMock()
        mock_item_2.workflow_id = "wf-2"
        mock_item_2.name = "Workflow 2"

        mock_result = MagicMock()
        mock_result.items = [mock_item_1, mock_item_2]

        mock_service = MagicMock()
        mock_service.get_many.return_value = mock_result

        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _remove_workflows_for_pack("test_pack")

        assert count == 2
        assert mock_service.delete.call_count == 2
        deleted_ids = [
            call.args[0] if call.args else call.kwargs.get("workflow_id") for call in mock_service.delete.call_args_list
        ]
        assert "wf-1" in deleted_ids
        assert "wf-2" in deleted_ids

    def test_returns_zero_when_no_workflows(self) -> None:
        mock_result = MagicMock()
        mock_result.items = []

        mock_service = MagicMock()
        mock_service.get_many.return_value = mock_result

        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _remove_workflows_for_pack("empty_pack")

        assert count == 0

    def test_handles_query_error(self) -> None:
        mock_service = MagicMock()
        mock_service.get_many.side_effect = Exception("DB error")

        with patch("invokeai.app.api.routers.custom_nodes.ApiDependencies") as mock_deps:
            mock_deps.invoker.services.workflow_records = mock_service
            count = _remove_workflows_for_pack("error_pack")

        assert count == 0


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
