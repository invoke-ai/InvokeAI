## Custom Node Development

Any folders placed in this directory will be mounted into the container.
Invoke will automatically load these folders as custom nodes packages.
Sub-folders in this directory may even contain their own git repositories without conflicting with the InvokeAI repository, meaning vscode will not show your custom-node projects files as uncommitted changes for the invokeai repository.

All normal development tooling will work as expected, including:

- Debugging, breakpoints, and variable inspection.
- Import resolution.
- Linting and formatting.
- Type checking.
