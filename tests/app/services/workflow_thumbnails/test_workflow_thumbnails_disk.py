"""Where a workflow's thumbnail is: bundled workflows ship theirs, the accounts' own are stored."""

from pathlib import Path
from unittest import mock

from invokeai.app.services.workflow_records.workflow_records_common import WorkflowCategory
from invokeai.app.services.workflow_thumbnails.workflow_thumbnails_disk import WorkflowThumbnailFileStorageDisk


def test_a_caller_that_knows_the_category_spares_reading_the_workflow(tmp_path: Path) -> None:
    thumbnails = WorkflowThumbnailFileStorageDisk(tmp_path)
    invoker = mock.Mock()
    invoker.services.workflow_records.get.side_effect = AssertionError("the workflow was read")
    thumbnails.start(invoker)

    assert thumbnails.get_path("mine", category=WorkflowCategory.User) == tmp_path / "mine.webp"
    assert thumbnails.get_path("default_x", category=WorkflowCategory.Default).name == "default_x.png"
    assert thumbnails.get_path("default_x", category=WorkflowCategory.Default).parent != tmp_path


def test_without_a_category_the_workflow_decides(tmp_path: Path) -> None:
    thumbnails = WorkflowThumbnailFileStorageDisk(tmp_path)
    invoker = mock.Mock()
    invoker.services.workflow_records.get.return_value.workflow.meta.category = WorkflowCategory.Default
    thumbnails.start(invoker)

    assert thumbnails.get_path("default_x").name == "default_x.png"
    invoker.services.workflow_records.get.assert_called_once_with("default_x")
