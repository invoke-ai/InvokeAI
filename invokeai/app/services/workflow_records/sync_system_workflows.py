import pkgutil
from logging import Logger
from pathlib import Path

import semver

from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowValidator,
)

# TODO: When I remove a workflow from system_workflows/ and do a `pip install --upgrade .`, the file
# is not removed from site-packages! The logic to delete old system workflows below doesn't work
# for normal installs. It does work for editable. Not sure why.

system_workflows_dir = "system_workflows"


def get_system_workflows_from_json() -> list[Workflow]:
    app_workflows: list[Workflow] = []
    workflow_paths = (Path(__file__).parent / Path(system_workflows_dir)).glob("*.json")
    for workflow_path in workflow_paths:
        workflow_bytes = pkgutil.get_data(__name__, f"{system_workflows_dir}/{workflow_path.name}")
        if workflow_bytes is None:
            raise ValueError(f"Could not load system workflow: {workflow_path.name}")

        app_workflows.append(WorkflowValidator.validate_json(workflow_bytes))
    return app_workflows


def sync_system_workflows(workflow_records: WorkflowRecordsStorageBase, logger: Logger) -> None:
    """Syncs system workflows in the workflow_library database with the latest system workflows."""

    system_workflows = get_system_workflows_from_json()
    system_workflow_ids = [w.id for w in system_workflows]
    installed_workflows = workflow_records._get_all_system_workflows()
    installed_workflow_ids = [w.id for w in installed_workflows]

    for workflow in installed_workflows:
        if workflow.id not in system_workflow_ids:
            workflow_records._delete_system_workflow(workflow.id)
            logger.info(f"Deleted system workflow: {workflow.name}")

    for workflow in system_workflows:
        if workflow.id not in installed_workflow_ids:
            workflow_records._create_system_workflow(workflow)
            logger.info(f"Installed system workflow: {workflow.name}")
        else:
            installed_workflow = workflow_records.get(workflow.id).workflow
            installed_version = semver.Version.parse(installed_workflow.version)
            new_version = semver.Version.parse(workflow.version)

            if new_version.compare(installed_version) > 0:
                workflow_records._update_system_workflow(workflow)
                logger.info(f"Updated system workflow: {workflow.name}")
