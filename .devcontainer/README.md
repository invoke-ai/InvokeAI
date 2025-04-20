# Overview

This folder contains the devcontainer setup used by InvokeAI.

## Common Problems

#### Launching backend after container rebuild results in package related error

eg: `Cannot reference module onnx-runtime/foo/bar`  
This can sometimes happen and appears to be related to `uv.lockfile` permissions, this can usually be fixed with `F1 -> Rebuild container`, or if that fails delete the uv lockfile and then rebuild the container.  
If both of these fail then there may be a legitimate package versioning issue, ask for help on the discord!

## VSCode Workspace

Within this folder there is an `InvokeAI.code-workspace` file.
Opening this workspace within the devcontainer is encouraged as it will provide easy access to the most commonly used folders and files, including:

- InvokeAI application data folder (database/models/config/etc).
- Mounted custom-node packages.
- InvokeAI frontend src
- InvokeAI backend src

## Custom Node Development

Any folders placed in the `.devcontainer/mounts/custom-nodes` directory will be mounted into the container.
Invoke will automatically load these folders as custom nodes packages.
Sub-folders in this directory may even contain their own git repositories without conflicting with the InvokeAI repository, meaning vscode will not show your custom-node projects files as uncommitted changes for the invokeai repository.

All normal development tooling will work as expected, including:

- Debugging, breakpoints, and variable inspection.
- Import resolution.
- Linting and formatting.
- Type checking.
