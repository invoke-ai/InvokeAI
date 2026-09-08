# Development Containers

[Development Containers](https://containers.dev/) can be a good way to get a development environment running.
Containers also provide some isolation to the host system.

Dev containers may be run locally by your IDE
([VS Code](https://code.visualstudio.com/docs/devcontainers/containers),
[PyCharm Pro](https://www.jetbrains.com/help/pycharm/2026.1/connect-to-devcontainer.html)),
on a cloud IDE ([GitHub Codespaces](https://docs.github.com/en/codespaces/about-codespaces/what-are-codespaces)),
or by an editor-agnostic host ([DevPod](https://github.com/skevetter/devpod/)).


## Available Containers
- **CPU-only** has no GPU support. Smaller dependency size; no additional setup required.
- **CUDA** for NVIDIA GPUs. Installs torch with CUDA dependencies. Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html#generating-a-cdi-specification/) on the host.
  - **CUDA on podman** for those using [podman](https://podman.io/) to run containers [[VS Code](https://code.visualstudio.com/remote/advancedcontainers/docker-options#_podman), [PyCharm](https://www.jetbrains.com/help/pycharm/2026.1/podman.html)].
  - *CUDA on Docker* 🚧 **TODO** 🚧 for those using [Docker](https://www.docker.com/) to run containers.
<!-- TODO: ROCm? -->
<!-- TODO: What do MacOS hosts do? -->


## Customizing a Container

Customizating the container environment
(e.g. to use your existing `INVOKEAI_ROOT` data directory)
requires making a new configuration.

Copy one of the provided `devcontainer.json` to a new subdirectory such as `local/devcontainer.json` and edit it there. Change its `"name"` and other properties as desired.

See the comments in `base/partial.jsonc` for some suggestions.
<!-- Sadly, helpful comments do not make it through to the output of the merge process described in the next section. -->


## Implementation Details

Unfortunately [there is no extension mechanism for `devcontainer.json`](https://github.com/devcontainers/spec/issues/22),
so we've cobbled one together.

`base/partial.jsonc` contains the common base of the devcontainer configuration.

`merge-partial-configs.sh` merges it with other `partial.jsonc` files to form full `devcontainer.json` configs in subdirectories.

> [!NOTE]
> Maintainers should run `merge-partial-configs.sh` and **commit** the results after changing any public `partial.jsonc`.
> This keeps the configs available to dev clients on their initial load of the project.

Generated `devcontainer.json` files should not be edited, as any edits will be overwritten the next time the partials change.


### Differences from /docker/Dockerfile

Production containers are typically optimized for deployment size and strip away anything non-essential for running their single service.

Development containers are built for developer experience. They include all the tools to build, test, and debug the application.

Production containers include the application code in their image.

Development containers assume you're going to be changing the code,
so the image provides the environment but the application code is mounted separately at runtime.
