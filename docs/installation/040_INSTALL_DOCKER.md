---
title: Installing with Docker
---

# :fontawesome-brands-docker: Docker

!!! warning "macOS and AMD GPU Users"

    We highly recommend to Install InvokeAI locally using [these instructions](INSTALLATION.md),
    because Docker containers can not access the GPU on macOS.

!!! warning "AMD GPU Users"

    Container support for AMD GPUs has been reported to work by the community, but has not received
    extensive testing. Please make sure to set the `GPU_DRIVER=rocm` environment variable (see below), and
    use the `build.sh` script to build the image for this to take effect at build time.

!!! tip "Linux and Windows Users"

    For optimal performance, configure your Docker daemon to access your machine's GPU.
    Docker Desktop on Windows [includes GPU support](https://www.docker.com/blog/wsl-2-gpu-support-for-docker-desktop-on-nvidia-gpus/).
    Linux users should install and configure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Why containers?

They provide a flexible, reliable way to build and deploy InvokeAI.
See [Processes](https://12factor.net/processes) under the Twelve-Factor App
methodology for details on why running applications in such a stateless fashion is important.

The container is configured for CUDA by default, but can be built to support AMD GPUs
by setting the `GPU_DRIVER=rocm` environment variable at Docker image build time.

Developers on Apple silicon (M1/M2/M3): You
[can't access your GPU cores from Docker containers](https://github.com/pytorch/pytorch/issues/81224)
and performance is reduced compared with running it directly on macOS but for
development purposes it's fine. Once you're done with development tasks on your
laptop you can build for the target platform and architecture and deploy to
another environment with NVIDIA GPUs on-premises or in the cloud.

## TL;DR

This assumes properly configured Docker on Linux or Windows/WSL2. Read on for detailed customization options.

    ```bash
    # docker compose commands should be run from the `docker` directory
    cd docker
    docker compose up
    ```

## Installation in a Linux container (desktop)

### Prerequisites

#### Install [Docker](https://github.com/santisbon/guides#docker)

On the [Docker Desktop app](https://docs.docker.com/get-docker/), go to
Preferences, Resources, Advanced. Increase the CPUs and Memory to avoid this
[Issue](https://github.com/invoke-ai/InvokeAI/issues/342). You may need to
increase Swap and Disk image size too.

#### Get a Huggingface-Token

Besides the Docker Agent you will need an Account on
[huggingface.co](https://huggingface.co/join).

After you succesfully registered your account, go to
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), create
a token and copy it, since you will need in for the next step.

### Setup

Set up your environmnent variables. In the `docker` directory, make a copy of `.env.sample` and name it `.env`. Make changes as necessary.

Any environment variables supported by InvokeAI can be set here - please see the [CONFIGURATION](../features/CONFIGURATION.md) for further detail.

At a minimum, you might want to set the `INVOKEAI_ROOT` environment variable
to point to the location where you wish to store your InvokeAI models, configuration, and outputs.

<figure markdown>

| Environment-Variable <img width="220" align="right"/> | Default value <img width="360" align="right"/> | Description                                                                                                                                                                                       |
| ----------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `INVOKEAI_ROOT`                                       | `~/invokeai`                                   | **Required** - the location of your InvokeAI root directory. It will be created if it does not exist.
| `HUGGING_FACE_HUB_TOKEN`                              |                                                | InvokeAI will work without it, but some of the integrations with HuggingFace (like downloading from models from private repositories) may not work|
| `GPU_DRIVER`                                          | `cuda`                                         | Optionally change this to `rocm` to build the image for AMD GPUs. NOTE: Use the `build.sh` script to build the image for this to take effect.

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
- A: Your `docker-entrypoint.sh` file likely has Windows (CRLF) as opposed to Unix (LF) line endings,
    and you may have cloned this repository before the issue was fixed. To solve this, please change
    the line endings in the `docker-entrypoint.sh` file to `LF`. You can do this in VSCode
    (`Ctrl+P` and search for "line endings"), or by using the `dos2unix` utility in WSL.
    Finally, you may delete `docker-entrypoint.sh` followed by  `git pull; git checkout docker/docker-entrypoint.sh`
    to reset the file to its most recent version.
    For more information on this issue, please see the [Docker Desktop documentation](https://docs.docker.com/desktop/troubleshoot/topics/#avoid-unexpected-syntax-errors-use-unix-style-line-endings-for-files-in-containers)
