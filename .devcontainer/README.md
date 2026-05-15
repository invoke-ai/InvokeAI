# Development Containers

[Development Containers](https://containers.dev/) can be a good way to get a development environment running,
and provide some isolation to the host system.

Invoke is a non-trivial application to put in a container:
- It wants access to the host GPU.
- You may wish to have use an existing host directory for data (models and outputs).

Unfortunately [there is no extension mechanism for `devcontainer.json`](https://github.com/devcontainers/spec/issues/22),
so there's duplication.
To make local customizations, you'll need to make your own copy of one of the subdirectories here.
