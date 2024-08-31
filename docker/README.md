# Invoke in Docker

First things first:

- Ensure that Docker can use your [NVIDIA][nvidia docker docs] or [AMD][amd docker docs] GPU.
- This document assumes a Linux system, but should work similarly under Windows with WSL2.
- We don't recommend running Invoke in Docker on macOS at this time. It works, but very slowly.

## Quickstart

No `docker compose`, no persistence, single command, using the official images:

**CUDA (NVIDIA GPU):**

```bash
docker run --runtime=nvidia --gpus=all --publish 9090:9090 ghcr.io/invoke-ai/invokeai
```

**ROCm (AMD GPU):**

```bash
docker run --device /dev/kfd --device /dev/dri --publish 9090:9090 ghcr.io/invoke-ai/invokeai:main-rocm
```

Open `http://localhost:9090` in your browser once the container finishes booting, install some models, and generate away!

### Data persistence

To persist your generated images and downloaded models outside of the container, add a `--volume/-v` flag to the above command, e.g.:

```bash
docker run --volume /some/local/path:/invokeai {...etc...}
```

`/some/local/path/invokeai` will contain all your data.
It can *usually* be reused between different installs of Invoke. Tread with caution and read the release notes!

## Customize the container

The included `run.sh` script is a convenience wrapper around `docker compose`. It can be helpful for passing additional build arguments to `docker compose`. Alternatively, the familiar `docker compose` commands work just as well.

```bash
cd docker
cp .env.sample .env
# edit .env to your liking if you need to; it is well commented.
./run.sh
```

It will take a few minutes to build the image the first time. Once the application starts up, open `http://localhost:9090` in your browser to invoke!

>[!TIP]
>When using the `run.sh` script, the container will continue running after Ctrl+C. To shut it down, use the `docker compose down` command.

## Docker setup in detail

#### Linux

1. Ensure buildkit is enabled in the Docker daemon settings (`/etc/docker/daemon.json`)
2. Install the `docker compose` plugin using your package manager, or follow a [tutorial](https://docs.docker.com/compose/install/linux/#install-using-the-repository).
    - The deprecated `docker-compose` (hyphenated) CLI probably won't work. Update to a recent version.
3. Ensure docker daemon is able to access the GPU.
    - [NVIDIA docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    - [AMD docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)

#### macOS

> [!TIP]
> You'll be better off installing Invoke directly on your system, because Docker can not use the GPU on macOS.

If you are still reading:

1. Ensure Docker has at least 16GB RAM
2. Enable VirtioFS for file sharing
3. Enable `docker compose` V2 support

This is done via Docker Desktop preferences.

### Configure the Invoke Environment

1. Make a copy of `.env.sample` and name it `.env` (`cp .env.sample .env` (Mac/Linux) or `copy example.env .env` (Windows)). Make changes as necessary. Set `INVOKEAI_ROOT` to an absolute path to the desired location of the InvokeAI runtime directory. It may be an existing directory from a previous installation (post 4.0.0).
1. Execute `run.sh`

The image will be built automatically if needed.

The runtime directory (holding models and outputs) will be created in the location specified by `INVOKEAI_ROOT`. The default location is `~/invokeai`. Navigate to the Model Manager tab and install some models before generating.

### Use a GPU

- Linux is *recommended* for GPU support in Docker.
- WSL2 is *required* for Windows.
- only `x86_64` architecture is supported.

The Docker daemon on the system must be already set up to use the GPU. In case of Linux, this involves installing `nvidia-docker-runtime` and configuring the `nvidia` runtime as default. Steps will be different for AMD. Please see Docker/NVIDIA/AMD documentation for the most up-to-date instructions for using your GPU with Docker.

To use an AMD GPU, set `GPU_DRIVER=rocm` in your `.env` file before running `./run.sh`.

## Customize

Check the `.env.sample` file. It contains some environment variables for running in Docker. Copy it, name it `.env`, and fill it in with your own values. Next time you run `run.sh`, your custom values will be used.

You can also set these values in `docker-compose.yml` directly, but `.env` will help avoid conflicts when code is updated.

Values are optional, but setting `INVOKEAI_ROOT` is highly recommended. The default is `~/invokeai`. Example:

```bash
INVOKEAI_ROOT=/Volumes/WorkDrive/invokeai
HUGGINGFACE_TOKEN=the_actual_token
CONTAINER_UID=1000
GPU_DRIVER=cuda
```

Any environment variables supported by InvokeAI can be set here. See the [Configuration docs](https://invoke-ai.github.io/InvokeAI/features/CONFIGURATION/) for further detail.

---

[nvidia docker docs]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
[amd docker docs]: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html
