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

### 4 GB of VRAM

This should be adequate for 512x512 pixel images using Stable Diffusion 1.5
and derived models, provided that you do not use the NSFW checker. It won't be loaded unless you go into the UI settings and turn it on.

If you are on a CUDA-enabled GPU, we will automatically use xformers or torch-sdp to reduce VRAM requirements, though you can explicitly configure this. See the [configuration docs].

### 6 GB of VRAM

This is a border case. Using the SD 1.5 series you should be able to
generate images up to 640x640 with the NSFW checker enabled, and up to
1024x1024 with it disabled.

If you run into persistent memory issues there are a series of
environment variables that you can set before launching InvokeAI that
alter how the PyTorch machine learning library manages memory. See
<https://pytorch.org/docs/stable/notes/cuda.html#memory-management> for
a list of these tweaks.

### 12 GB of VRAM

This should be sufficient to generate larger images up to about 1280x1280.

[create an issue]: https://github.com/invoke-ai/InvokeAI/issues
[discord]: https://discord.gg/ZmtBAhwWhy
[configuration docs]: ../features/CONFIGURATION.md
