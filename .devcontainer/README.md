# Overview

This folder contains the devcontainer setup used by InvokeAI.  
If you are not familiar with devcontainers, we encourage you to check out the VSCode [devcontainer tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial) for more information!  
But in short; devcontainers are a one click solution to build, launch, and install an isolated/containerized development environment which is setup specifically for writing code for this repository.  
This means you can get started with development without having to manually install any dependencies or tools or manage configuration files, the container automatically does all of it for you.

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

## Common Problems

### Launching backend after container rebuild results in package related error

eg: `Cannot reference module onnx-runtime/foo/bar`  
This can sometimes happen and appears to be related to `uv.lockfile` permissions, this can usually be fixed with `F1 -> Rebuild container`, or if that fails delete the uv lockfile and then rebuild the container.  
If both of these fail then there may be a legitimate package versioning issue, ask for help on the discord!

### Python Hot-Reloading not working

First check the backend logs and verify that "Juririgged" has logged a message indicating it detected and is watching the file in question.  
For example:

```
[2025-04-24 23:53:32,956]::[jurigged]::INFO --> Watch /workspaces/InvokeAI/invokeai/app/api_app.py
[2025-04-24 23:53:32,956]::[jurigged]::INFO --> Watch /home/node/invokeai/nodes/siscos-nodes/__init__.py
```

If you have verified that the file is being watched, but the changes are not being picked up, then continue reading the next section.
If the below sections do not apply to you, then it is an issue with [jurigged](https://github.com/breuleux/jurigged).

### Jurigged not picking up changes

If you are developing on a Windows machine running [WSL](https://code.visualstudio.com/docs/remote/wsl-tutorial), then [this issue link is relevant to you](https://github.com/microsoft/WSL/issues/4739)  
In short, WSL2 does not support propagating windows file system events to 'inotify' events for the linux virtual machine.

To fix this, it is strongly recommended that you store your code on the WSL filesystem instead of the Windows filesystem.

#### Option 1: Clone the repository within the WSL filesystem

Open VSCode and press `F1` and select **WSL: Connect to WSL**, this will open a new VSCode window connected to the WSL filesystem.  
Afterwards, you can clone the InvokeAI repository using the command line, or by using the VSCode _Explore_ panel(_Ctrl+Shift+E_) which will show a **Clone Repository** button when no folder is opened.

#### Option 2: Move existing code to the WSL filesystem

First, open windows file explorer and enter `\\wsl$` in the address bar, this will show your WSL distributions.
More information can be found at [this link](https://learn.microsoft.com/en-us/windows/wsl/filesystems#view-your-current-directory-in-windows-file-explorer).

Next, open your preferred WSL distribution (default: _Ubuntu_) and decide where you want to store your code, its typically wise to use a path under your **home** directory (eg: `/home/<username>/code/<project>`).

Now, open VSCode and press `F1` and select **WSL: Connect to WSL**, once VSCode has opened in WSL you can click the **Open Folder** button in the VSCode _Explore_ panel(_Ctrl+Shift+E_) and select the correct folder from the dropdown menu.  
Afterwards, just press `F1` and select **Reopen in Container** to launch the devcontainer.

See [this link](https://code.visualstudio.com/remote/advancedcontainers/improve-performance) for more information.
