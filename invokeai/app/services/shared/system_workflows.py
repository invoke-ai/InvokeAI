from logging import Logger
from pathlib import Path

import semver

import invokeai.app.assets.workflows as system_workflows_dir
from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowNotFoundError, WorkflowValidator

system_workflows = Path(system_workflows_dir.__path__[0]).glob("*.json")


def create_system_workflows(workflow_records: WorkflowRecordsStorageBase, logger: Logger) -> None:
    """Creates the system workflows."""
    for workflow_filename in system_workflows:
        with open(workflow_filename, "rb") as f:
            workflow_bytes = f.read()
            if workflow_bytes is None:
                raise ValueError(f"Could not find system workflow: {workflow_filename}")

            new_workflow = WorkflowValidator.validate_json(workflow_bytes)

            try:
                installed_workflow = workflow_records.get(new_workflow.id).workflow
                installed_version = semver.Version.parse(installed_workflow.version)
                new_version = semver.Version.parse(new_workflow.version)

                if new_version.compare(installed_version) > 0:
                    logger.info(f"Updating system workflow: {new_workflow.name}")
                    workflow_records._add_system_workflow(new_workflow)
            except WorkflowNotFoundError:
                logger.info(f"Installing system workflow: {new_workflow.name}")
                workflow_records._add_system_workflow(new_workflow)
