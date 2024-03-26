# Installation Troubleshooting

!!! info "How to Reinstall"

    Many issues can be resolved by re-installing the application. You won't lose any data by re-installing. We suggest downloading the [latest release] and using it to re-install the application.

    When you run the installer, you'll have an option to select the version to install. If you aren't ready to upgrade, you choose the current version to fix a broken install.

If the troubleshooting steps on this page don't get you up and running, please either [create an issue] or hop on [discord] for help.

## OSErrors on Windows while installing dependencies

During a zip file installation or an online update, installation stops
with an error like this:

![broken-dependency-screenshot](../assets/troubleshooting/broken-dependency.png){:width="800px"}

To resolve this, re-install the application as described above.

## Stable Diffusion XL generation fails after trying to load UNet

InvokeAI is working in other respects, but when trying to generate
images with Stable Diffusion XL you get a "Server Error". The text log
in the launch window contains this log line above several more lines of
error messages:

`INFO --> Loading model:D:\LONG\PATH\TO\MODEL, type sdxl:main:unet`

This failure mode occurs when there is a network glitch during
downloading the very large SDXL model.

To address this, first go to the Model Manager and delete the
Stable-Diffusion-XL-base-1.X model. Then, click the HuggingFace tab,
paste the Repo ID stabilityai/stable-diffusion-xl-base-1.0 and install
the model.

## Package dependency conflicts

If you have previously installed InvokeAI or another Stable Diffusion
package, the installer may occasionally pick up outdated libraries and
either the installer or `invoke` will fail with complaints about
library conflicts.

To resolve this, re-install the application as described above.

## InvokeAI runs extremely slowly on Linux or Windows systems

The most frequent cause of this problem is when the installation
process installed the CPU-only version of the torch machine-learning
library, rather than a version that takes advantage of GPU
acceleration. To confirm this issue, look at the InvokeAI startup
messages. If you see a message saying ">> Using device CPU", then
this is what happened.

To resolve this, re-install the application as described above. Be sure to select the correct GPU device.

## Invalid configuration file

Everything seems to install ok, you get a `ValidationError` when starting up the app.

This is caused by an invalid setting in the `invokeai.yaml` configuration file. The error message should tell you what is wrong.

Check the [configuration docs] for more detail about the settings and how to specify them.

## Out of Memory Issues

The models are large, VRAM is expensive, and you may find yourself
faced with Out of Memory errors when generating images. Here are some
tips to reduce the problem:

<h3>4 GB of VRAM</h3>

This should be adequate for 512x512 pixel images using Stable Diffusion 1.5
and derived models, provided that you do not use the NSFW checker. It won't be loaded unless you go into the UI settings and turn it on.

If you are on a CUDA-enabled GPU, we will automatically use xformers or torch-sdp to reduce VRAM requirements, though you can explicitly configure this. See the [configuration docs].

<h3>6 GB of VRAM</h3>

This is a border case. Using the SD 1.5 series you should be able to
generate images up to 640x640 with the NSFW checker enabled, and up to
1024x1024 with it disabled.

If you run into persistent memory issues there are a series of
environment variables that you can set before launching InvokeAI that
alter how the PyTorch machine learning library manages memory. See
<https://pytorch.org/docs/stable/notes/cuda.html#memory-management> for
a list of these tweaks.

<h3>12 GB of VRAM</h3>

This should be sufficient to generate larger images up to about 1280x1280.

[create an issue]: https://github.com/invoke-ai/InvokeAI/issues
[discord]: https://discord.gg/ZmtBAhwWhy
[configuration docs]: ../features/CONFIGURATION.md

## Memory Leak (Linux)

If you notice a memory leak, it could be caused to memory fragmentation as models are loaded and/or moved from CPU to GPU.

A workaround is to tune memory allocation with an environment variable:

```bash
# Force blocks >1MB to be allocated with `mmap` so that they are released to the system immediately when they are freed.
MALLOC_MMAP_THRESHOLD_=1048576
```

!!! warning "Speed vs Memory Tradeoff"

    Your generations may be slower overall when setting this environment variable.

!!! info "Possibly dependent on `libc` implementation"

    It's not known if this issue occurs with other `libc` implementations such as `musl`.

    If you encounter this issue and your system uses a different implementation, please try this environment variable and let us know if it fixes the issue.

<h3>Detailed Discussion</h3>

Python (and PyTorch) relies on the memory allocator from the C Standard Library (`libc`). On linux, with the GNU C Standard Library implementation (`glibc`), our memory access patterns have been observed to cause severe memory fragmentation.

This fragmentation results in large amounts of memory that has been freed but can't be released back to the OS. Loading models from disk and moving them between CPU/CUDA seem to be the operations that contribute most to the fragmentation.

This memory fragmentation issue can result in OOM crashes during frequent model switching, even if `ram` (the max RAM cache size) is set to a reasonable value (e.g. a OOM crash with `ram=16` on a system with 32GB of RAM).

This problem may also exist on other OSes, and other `libc` implementations. But, at the time of writing, it has only been investigated on linux with `glibc`.

To better understand how the `glibc` memory allocator works, see these references:

- Basics: <https://www.gnu.org/software/libc/manual/html_node/The-GNU-Allocator.html>
- Details: <https://sourceware.org/glibc/wiki/MallocInternals>

Note the differences between memory allocated as chunks in an arena vs. memory allocated with `mmap`. Under `glibc`'s default configuration, most model tensors get allocated as chunks in an arena making them vulnerable to the problem of fragmentation.
