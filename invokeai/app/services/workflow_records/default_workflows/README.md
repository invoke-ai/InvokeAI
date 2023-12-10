# Default Workflows

Workflows placed in this directory will be synced to the `workflow_library` as
_default workflows_ on app startup.

- Default workflows are not editable by users. If they are loaded and saved,
  they will save as a copy of the default workflow.
- Default workflows must have the `meta.category` property set to `"default"`.
  An exception will be raised during sync if this is not set correctly.
- Default workflows appear on the "Default Workflows" tab of the Workflow
  Library.

After adding or updating default workflows, you **must** start the app up and
load them to ensure:

- The workflow loads without warning or errors
- The workflow runs successfully
