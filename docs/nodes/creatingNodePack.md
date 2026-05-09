# Creating a Node Pack for the Custom Node Manager

This guide explains how to structure your Git repository so it can be installed via InvokeAI's Custom Node Manager.

## Repository Structure

Your repository **is** the node pack. When a user installs it, the entire repo is cloned into the `nodes` directory.

### Minimum Required Structure

```
my-node-pack/
├── __init__.py          # Required: imports your node classes
├── my_node.py           # Your node implementation(s)
└── README.md            # Recommended: describe your nodes
```

The `__init__.py` at the root is **mandatory**. Without it, the pack will not be loaded.

### Recommended Structure

```
my-node-pack/
├── __init__.py          # Imports all node classes
├── requirements.txt     # Python dependencies (user-installed)
├── README.md            # Description, usage, examples
├── node_one.py          # Node implementation
├── node_two.py          # Node implementation
├── utils.py             # Shared utilities
└── workflows/           # Optional: workflow files
    ├── example_workflow.json
    └── advanced_workflow.json
```

## The `__init__.py` File

This file must import all invocation classes you want to register. Only classes imported here will be available in InvokeAI.

```python
from .node_one import MyFirstInvocation
from .node_two import MySecondInvocation
```

If you have nodes in subdirectories:

```python
from .nodes.image_tools import CropInvocation, ResizeInvocation
from .nodes.text_tools import ConcatInvocation
```

## Dependencies (`requirements.txt` or `pyproject.toml`)

If your nodes require additional Python packages, list them in a `requirements.txt` (or `pyproject.toml`) at the repository root:

```
numpy>=1.24
opencv-python>=4.8
```

The Custom Node Manager **does not** install these dependencies automatically — auto-installing into the running InvokeAI environment risks pulling in incompatible versions and breaking the application. After install, the UI shows the user a toast telling them that manual installation is required, and your README should document the exact install command (e.g. `pip install -r requirements.txt` from inside an activated InvokeAI environment).

**Important:** Avoid pinning versions too tightly. InvokeAI has its own dependencies, and version conflicts can cause issues. Use minimum version constraints (`>=`) where possible.

## Including Workflows

If your repository contains workflow `.json` files, they will be **automatically imported** into the user's workflow library during installation.

### Workflow Detection

The installer recursively scans your repository for `.json` files. A file is recognized as a workflow if it contains both `nodes` and `edges` keys at the top level.

### Tagging

Imported workflows are automatically tagged with `node-pack:<your-repo-name>` so users can filter for them in the workflow library. When the node pack is uninstalled, these workflows are also removed.

### Workflow Format

Workflows should follow the standard InvokeAI workflow format:

```json
{
  "name": "My Example Workflow",
  "author": "Your Name",
  "description": "Demonstrates how to use MyFirstInvocation",
  "version": "1.0.0",
  "contact": "",
  "tags": "example, my-node-pack",
  "notes": "",
  "meta": {
    "version": "3.0.0",
    "category": "user"
  },
  "exposedFields": [],
  "nodes": [...],
  "edges": [...]
}
```

**Tip:** The easiest way to create a workflow file is to build the workflow in InvokeAI's workflow editor, then export it via **Save As** and copy the `.json` file into your repository.

## Node Implementation

Each node is a Python class decorated with `@invocation()`. Here's a minimal example:

```python
from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.invocation_api import BaseInvocationOutput, invocation_output


@invocation_output("my_output")
class MyOutput(BaseInvocationOutput):
    result: str = OutputField(description="The result")


@invocation(
    "my_node",
    title="My Node",
    tags=["example", "custom"],
    category="custom",
    version="1.0.0",
)
class MyInvocation(BaseInvocation):
    """Does something useful."""

    input_text: str = InputField(default="", description="Input text")

    def invoke(self, context) -> MyOutput:
        return MyOutput(result=f"Processed: {self.input_text}")
```

For full details on the invocation API, see the [Invocation API documentation](invocation-api.md).

## Best Practices

- **Use a descriptive repository name** — it becomes the pack name shown in the UI
- **Include a README.md** with description, screenshots, and usage instructions
- **Version your nodes** using semver in the `@invocation()` decorator
- **Don't include large binary files** in your repository (models, weights, etc.)
- **Test your nodes** by placing the repo in the `nodes` directory before publishing
- **Include example workflows** so users can get started quickly
- **Tag your GitHub repository** with `invokeai-node` for discoverability
- **Avoid name collisions** — choose unique invocation type strings (e.g. `my_pack_resize` instead of just `resize`)

## Testing Your Pack

Before publishing, verify your pack works with the Custom Node Manager:

1. Create a Git repository with your node pack
2. Push it to GitHub (or any Git host)
3. In InvokeAI, go to the Nodes tab and install it via the Git URL
4. Verify your nodes appear in the workflow editor
5. Verify any included workflows are imported
6. Test uninstalling — nodes and workflows should be removed
