# Default Workflows

Workflows placed in this directory will be synced to the `workflow_library` as
_default workflows_ on app startup.

- Default workflows must have an id that starts with "default\_". The ID must be retained when the workflow is updated. You may need to do this manually.
- Default workflows are not editable by users. If they are loaded and saved,
  they will save as a copy of the default workflow.
- Default workflows must have the `meta.category` property set to `"default"`.
  An exception will be raised during sync if this is not set correctly.
- Default workflows appear on the "Default Workflows" tab of the Workflow
  Library.
- Default workflows should not reference any resources that are user-created or installed. That includes images and models. For example, if a default workflow references Juggernaut as an SDXL model, when a user loads the workflow, even if they have a version of Juggernaut installed, it will have a different UUID. They may see a warning. So, it's best to ship default workflows without any references to these types of resources.

After adding or updating default workflows, you **must** start the app up and
load them to ensure:

- The workflow loads without warning or errors
- The workflow runs successfully
