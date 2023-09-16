---
title: Installing with Docker
---

# :fontawesome-brands-docker: Docker

!!! warning "For most users"

    We highly recommend to Install InvokeAI locally using [these instructions](INSTALLATION.md)

!!! tip "For developers"

    For container-related development tasks or for enabling easy
    deployment to other environments (on-premises or cloud), follow these
    instructions.

    For general use, install locally to leverage your machine's GPU.

## Why containers?

They provide a flexible, reliable way to build and deploy InvokeAI. You'll also
use a Docker volume to store the largest model files and image outputs as a
first step in decoupling storage and compute. Future enhancements can do this
for other assets. See [Processes](https://12factor.net/processes) under the
Twelve-Factor App methodology for details on why running applications in such a
stateless fashion is important.

You can specify the target platform when building the image and running the
container. You'll also need to specify the InvokeAI requirements file that
matches the container's OS and the architecture it will run on.

Developers on Apple silicon (M1/M2): You
[can't access your GPU cores from Docker containers](https://github.com/pytorch/pytorch/issues/81224)
and performance is reduced compared with running it directly on macOS but for
development purposes it's fine. Once you're done with development tasks on your
laptop you can build for the target platform and architecture and deploy to
another environment with NVIDIA GPUs on-premises or in the cloud.

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

Set the fork you want to use and other variables.

!!! tip

    I preffer to save my env vars
    in the repository root in a `.env` (or `.envrc`) file to automatically re-apply
    them when I come back.

The build- and run- scripts contain default values for almost everything,
besides the [Hugging Face Token](https://huggingface.co/settings/tokens) you
created in the last step.

Some Suggestions of variables you may want to change besides the Token:

<figure markdown>

| Environment-Variable <img width="220" align="right"/> | Default value <img width="360" align="right"/> | Description                                                                                                                                                                                       |
| ----------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HUGGING_FACE_HUB_TOKEN`                              | No default, but **required**!                  | This is the only **required** variable, without it you can't download the huggingface models                                                                                                      |
| `REPOSITORY_NAME`                                     | The Basename of the Repo folder                | This name will used as the container repository/image name                                                                                                                                        |
| `VOLUMENAME`                                          | `${REPOSITORY_NAME,,}_data`                    | Name of the Docker Volume where model files will be stored                                                                                                                                        |
| `ARCH`                                                | arch of the build machine                      | Can be changed if you want to build the image for another arch                                                                                                                                    |
| `CONTAINER_REGISTRY`                                  | ghcr.io                                        | Name of the Container Registry to use for the full tag                                                                                                                                            |
| `CONTAINER_REPOSITORY`                                | `$(whoami)/${REPOSITORY_NAME}`                 | Name of the Container Repository                                                                                                                                                                  |
| `CONTAINER_FLAVOR`                                    | `cuda`                                         | The flavor of the image to built, available options are `cuda`, `rocm` and `cpu`. If you choose `rocm` or `cpu`, the extra-index-url will be selected automatically, unless you set one yourself. |
| `CONTAINER_TAG`                                       | `${INVOKEAI_BRANCH##*/}-${CONTAINER_FLAVOR}`   | The Container Repository / Tag which will be used                                                                                                                                                 |
| `INVOKE_DOCKERFILE`                                   | `Dockerfile`                                   | The Dockerfile which should be built, handy for development                                                                                                                                       |
| `PIP_EXTRA_INDEX_URL`                                 |                                                | If you want to use a custom pip-extra-index-url                                                                                                                                                   |

</figure>

#### Build the Image

I provided a build script, which is located next to the Dockerfile in
`docker/build.sh`. It can be executed from repository root like this:

```bash
./docker/build.sh
```

The build Script not only builds the container, but also creates the docker
volume if not existing yet.

#### Run the Container

After the build process is done, you can run the container via the provided
`docker/run.sh` script

```bash
./docker/run.sh
```

When used without arguments, the container will start the webserver and provide
you the link to open it. But if you want to use some other parameters you can
also do so.

!!! example "run script example"

    ```bash
    ./docker/run.sh "banana sushi" -Ak_lms -S42 -s10
    ```

    This would generate the legendary "banana sushi" with Seed 42, k_lms Sampler and 10 steps.

    Find out more about available CLI-Parameters at [features/CLI.md](../../features/CLI/#arguments)

---

## Running the container on your GPU

If you have an Nvidia GPU, you can enable InvokeAI to run on the GPU by running
the container with an extra environment variable to enable GPU usage and have
the process run much faster:

```bash
GPU_FLAGS=all ./docker/run.sh
```

This passes the `--gpus all` to docker and uses the GPU.

If you don't have a GPU (or your host is not yet setup to use it) you will see a
message like this:

`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].`

You can use the full set of GPU combinations documented here:

https://docs.docker.com/config/containers/resource_constraints/#gpu

For example, use `GPU_FLAGS=device=GPU-3a23c669-1f69-c64e-cf85-44e9b07e7a2a` to
choose a specific device identified by a UUID.

---

!!! warning "Deprecated"

    From here on you will find the the previous Docker-Docs, which will still
    provide some usefull informations.

## Usage (time to have fun)

### Startup

If you're on a **Linux container** the `invoke` script is **automatically
started** and the output dir set to the Docker volume you created earlier.

If you're **directly on macOS follow these startup instructions**. With the
Conda environment activated (`conda activate ldm`), run the interactive
interface that combines the functionality of the original scripts `txt2img` and
`img2img`: Use the more accurate but VRAM-intensive full precision math because
half-precision requires autocast and won't work. By default the images are saved
in `outputs/img-samples/`.

```Shell
python3 scripts/invoke.py --full_precision
```

You'll get the script's prompt. You can see available options or quit.

```Shell
invoke> -h
invoke> q
```

### Text to Image

For quick (but bad) image results test with 5 steps (default 50) and 1 sample
image. This will let you know that everything is set up correctly. Then increase
steps to 100 or more for good (but slower) results. The prompt can be in quotes
or not.

```Shell
invoke> The hulk fighting with sheldon cooper -s5 -n1
invoke> "woman closeup highly detailed"  -s 150
# Reuse previous seed and apply face restoration
invoke> "woman closeup highly detailed"  --steps 150 --seed -1 -G 0.75
```

You'll need to experiment to see if face restoration is making it better or
worse for your specific prompt.

If you're on a container the output is set to the Docker volume. You can copy it
wherever you want. You can download it from the Docker Desktop app, Volumes,
my-vol, data. Or you can copy it from your Mac terminal. Keep in mind
`docker cp` can't expand `*.png` so you'll need to specify the image file name.

On your host Mac (you can use the name of any container that mounted the
volume):

```Shell
docker cp dummy:/data/000001.928403745.png /Users/<your-user>/Pictures
```

### Image to Image

You can also do text-guided image-to-image translation. For example, turning a
sketch into a detailed drawing.

`strength` is a value between 0.0 and 1.0 that controls the amount of noise that
is added to the input image. Values that approach 1.0 allow for lots of
variations but will also produce images that are not semantically consistent
with the input. 0.0 preserves image exactly, 1.0 replaces it completely.

Make sure your input image size dimensions are multiples of 64 e.g. 512x512.
Otherwise you'll get `Error: product of dimension sizes > 2**31'`. If you still
get the error
[try a different size](https://support.apple.com/guide/preview/resize-rotate-or-flip-an-image-prvw2015/mac#:~:text=image's%20file%20size-,In%20the%20Preview%20app%20on%20your%20Mac%2C%20open%20the%20file,is%20shown%20at%20the%20bottom.)
like 512x256.

If you're on a Docker container, copy your input image into the Docker volume

```Shell
docker cp /Users/<your-user>/Pictures/sketch-mountains-input.jpg dummy:/data/
```

Try it out generating an image (or more). The `invoke` script needs absolute
paths to find the image so don't use `~`.

If you're on your Mac

```Shell
invoke> "A fantasy landscape, trending on artstation" -I /Users/<your-user>/Pictures/sketch-mountains-input.jpg --strength 0.75  --steps 100 -n4
```

If you're on a Linux container on your Mac

```Shell
invoke> "A fantasy landscape, trending on artstation" -I /data/sketch-mountains-input.jpg --strength 0.75  --steps 50 -n1
```

### Web Interface

You can use the `invoke` script with a graphical web interface. Start the web
server with:

```Shell
python3 scripts/invoke.py --full_precision --web
```

If it's running on your Mac point your Mac web browser to
<http://127.0.0.1:9090>

Press Control-C at the command line to stop the web server.

### Notes

Some text you can add at the end of the prompt to make it very pretty:

```Shell
cinematic photo, highly detailed, cinematic lighting, ultra-detailed, ultrarealistic, photorealism, Octane Rendering, cyberpunk lights, Hyper Detail, 8K, HD, Unreal Engine, V-Ray, full hd, cyberpunk, abstract, 3d octane render + 4k UHD + immense detail + dramatic lighting + well lit + black, purple, blue, pink, cerulean, teal, metallic colours, + fine details, ultra photoreal, photographic, concept art, cinematic composition, rule of thirds, mysterious, eerie, photorealism, breathtaking detailed, painting art deco pattern, by hsiao, ron cheng, john james audubon, bizarre compositions, exquisite detail, extremely moody lighting, painted by greg rutkowski makoto shinkai takashi takeuchi studio ghibli, akihiko yoshida
```

The original scripts should work as well.

```Shell
python3 scripts/orig_scripts/txt2img.py --help
python3 scripts/orig_scripts/txt2img.py --ddim_steps 100 --n_iter 1 --n_samples 1  --plms --prompt "new born baby kitten. Hyper Detail, Octane Rendering, Unreal Engine, V-Ray"
python3 scripts/orig_scripts/txt2img.py --ddim_steps 5   --n_iter 1 --n_samples 1  --plms --prompt "ocean" # or --klms
```
