# Invoke in Docker

First things first:

- Ensure that Docker can use your [NVIDIA][nvidia docker docs] or [AMD][amd docker docs] GPU.
- This document assumes a Linux system, but should work similarly under Windows with WSL2.
- We don't recommend running Invoke in Docker on macOS at this time. It works, but very slowly.

## Quickstart

Choose the command for your GPU. New installations store accounts, images and models in the `invokeai-data` Docker volume. For an existing installation, replace `invokeai-data` with the volume name or absolute host directory that holds your InvokeAI data.
If your data lives inside the old container, [recover it before replacing the container](https://v7.invoke.ai/configuration/docker/#recover-data-before-replacing-a-container).

**CUDA (NVIDIA GPU):**

```bash
docker run --runtime=nvidia --gpus=all --volume invokeai-data:/invokeai \
  --publish 127.0.0.1:9090:9090 ghcr.io/invoke-ai/invokeai
```

**ROCm (AMD GPU):**

```bash
docker run --device /dev/kfd --device /dev/dri \
  --env RENDER_GROUP_ID=$(getent group render | cut -d: -f3) \
  --volume invokeai-data:/invokeai --publish 127.0.0.1:9090:9090 ghcr.io/invoke-ai/invokeai:main-rocm
```

`RENDER_GROUP_ID` must match the host's `render` group. Without it, Invoke can't access the GPU.

Open `http://localhost:9090` on the Docker host and create the administrator account before giving other people access. Anyone who can reach a fresh installation can claim this account. If Docker runs on a remote host, run `ssh -N -L 127.0.0.1:9090:127.0.0.1:9090 user@docker-host` from your computer with your SSH login and host, then open the same URL locally.

## Customize the container

The included `run.sh` script passes build arguments to `docker compose` and checks the data mount before replacing a ROCm container. If you use Compose directly for a ROCm upgrade, [check the old container's data mount](https://v7.invoke.ai/configuration/docker/#recover-data-before-replacing-a-container) before running `up`, `down` or `rm`. Older containers may hold the only copy of your data in their writable layer.

For a new installation:

```bash
cd docker
cp .env.sample .env
# edit .env to your liking if you need to; it is well commented.
./run.sh
```

Compose publishes the port on the Docker host's loopback address by default. Once the image builds and Invoke starts, open `http://localhost:9090` on the host or through the SSH tunnel above to create the administrator.

To share the installation after setup, keep multi-user mode enabled. Set `HOST_INVOKEAI_IP` in `.env` to the host's LAN IP and run `./run.sh` again with the same data directory. For `docker run`, recreate the container with the volume containing your administrator and replace `127.0.0.1` in the port mapping with the host's LAN IP. If you use a reverse proxy, enable its route to Invoke only after setup.

**Updating an existing Compose setup:** Compose publishes the port on `127.0.0.1` unless `HOST_INVOKEAI_IP` is set, so after you update, other devices can no longer connect. Create the administrator on the host first. Then set `HOST_INVOKEAI_IP` in `.env` to the host's LAN IP, or `0.0.0.0` for every interface, and run `./run.sh` again.

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
- only `x86_64` architecture is supported for GPU acceleration; `arm64` (`aarch64`) builds are CPU-only (see below).

The Docker daemon on the system must be already set up to use the GPU. In case of Linux, this involves installing `nvidia-docker-runtime` and configuring the `nvidia` runtime as default. Steps will be different for AMD. Please see Docker/NVIDIA/AMD documentation for the most up-to-date instructions for using your GPU with Docker.

To use an AMD GPU, set `GPU_DRIVER=rocm` in your `.env` file before running `./run.sh`.

### ARM64 / aarch64 (CPU-only)

Linux ARM64 devices (Raspberry Pi 5, other SBCs, ARM servers) can build and run the image in CPU-only mode. From the repository root:

```bash
docker build -f docker/Dockerfile --build-arg GPU_DRIVER=cpu -t invokeai:cpu .
```

To cross-compile the arm64 image from an x86_64 machine, use `buildx`:

```bash
docker buildx build --platform linux/arm64 -f docker/Dockerfile --build-arg GPU_DRIVER=cpu -t invokeai:arm64-cpu .
```

On arm64, torch/torchvision are installed from PyPI (CPU wheels) regardless of `GPU_DRIVER` — CUDA and ROCm acceleration are not available. API-backed models (e.g. GPT Image, Gemini) work without a GPU.

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

Any environment variables supported by InvokeAI can be set here. See the [Configuration docs](https://invoke.ai/configuration/invokeai-yaml/) for further detail.

---

[nvidia docker docs]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
[amd docker docs]: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html
