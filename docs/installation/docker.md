---
title: Docker
---

!!! warning "macOS users"

    Docker can not access the GPU on macOS, so your generation speeds will be slow. Use the [installer](./installer.md) instead.

!!! tip "Linux and Windows Users"

    Configure Docker to access your machine's GPU.
    Docker Desktop on Windows [includes GPU support](https://www.docker.com/blog/wsl-2-gpu-support-for-docker-desktop-on-nvidia-gpus/).
    Linux users should follow the [NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) or [AMD](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) documentation.

## TL;DR

Ensure your Docker setup is able to use your GPU. Then:

    ```bash
    docker run --runtime=nvidia --gpus=all --publish 9090:9090 ghcr.io/invoke-ai/invokeai
    ```

Once the container starts up, open <http://localhost:9090> in your browser, install some models, and start generating.

## Build-It-Yourself

All the docker materials are located inside the [docker](https://github.com/invoke-ai/InvokeAI/tree/main/docker) directory in the Git repo.

    ```bash
    cd docker
    cp .env.sample .env
    docker compose up
    ```

We also ship the `run.sh` convenience script. See the `docker/README.md` file for detailed instructions on how to customize the docker setup to your needs.

### Prerequisites

#### Install [Docker](https://github.com/santisbon/guides#docker)

On the [Docker Desktop app](https://docs.docker.com/get-docker/), go to
Preferences, Resources, Advanced. Increase the CPUs and Memory to avoid this
[Issue](https://github.com/invoke-ai/InvokeAI/issues/342). You may need to
increase Swap and Disk image size too.

### Setup

Set up your environment variables. In the `docker` directory, make a copy of `.env.sample` and name it `.env`. Make changes as necessary.

Any environment variables supported by InvokeAI can be set here - please see the [CONFIGURATION](../configuration.md) for further detail.

At a minimum, you might want to set the `INVOKEAI_ROOT` environment variable
to point to the location where you wish to store your InvokeAI models, configuration, and outputs.

<figure markdown>

| Environment-Variable <img width="220" align="right"/> | Default value <img width="360" align="right"/> | Description                                                                                                                                        |
| ----------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INVOKEAI_ROOT`                                       | `~/invokeai`                                   | **Required** - the location of your InvokeAI root directory. It will be created if it does not exist.                                              |
| `HUGGING_FACE_HUB_TOKEN`                              |                                                | InvokeAI will work without it, but some of the integrations with HuggingFace (like downloading from models from private repositories) may not work |
| `GPU_DRIVER`                                          | `cuda`                                         | Optionally change this to `rocm` to build the image for AMD GPUs. NOTE: Use the `build.sh` script to build the image for this to take effect.      |

</figure>

#### Build the Image

Use the standard `docker compose build` command from within the `docker` directory.

If using an AMD GPU:
a: set the `GPU_DRIVER=rocm` environment variable in `docker-compose.yml` and continue using `docker compose build` as usual, or
b: set `GPU_DRIVER=rocm` in the `.env` file and use the `build.sh` script, provided for convenience

#### Run the Container

Use the standard `docker compose up` command, and generally the `docker compose` [CLI](https://docs.docker.com/compose/reference/) as usual.

Once the container starts up (and configures the InvokeAI root directory if this is a new installation), you can access InvokeAI at [http://localhost:9090](http://localhost:9090)

## Troubleshooting / FAQ

- Q: I am running on Windows under WSL2, and am seeing a "no such file or directory" error.
- A: Your `docker-entrypoint.sh` might have has Windows (CRLF) line endings, depending how you cloned the repository.
  To solve this, change the line endings in the `docker-entrypoint.sh` file to `LF`. You can do this in VSCode
  (`Ctrl+P` and search for "line endings"), or by using the `dos2unix` utility in WSL.
  Finally, you may delete `docker-entrypoint.sh` followed by `git pull; git checkout docker/docker-entrypoint.sh`
  to reset the file to its most recent version.
  For more information on this issue, see [Docker Desktop documentation](https://docs.docker.com/desktop/troubleshoot/topics/#avoid-unexpected-syntax-errors-use-unix-style-line-endings-for-files-in-containers)
