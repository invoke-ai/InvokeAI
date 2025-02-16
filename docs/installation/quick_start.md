# Invoke Community Edition Quick Start

Welcome to Invoke! Follow these steps to install, update, and get started creating.

## Step 1: System Requirements

Invoke runs on Windows 10+, macOS 14+ and Linux (Ubuntu 20.04+ is well-tested).

Hardware requirements vary significantly depending on model and image output size. The requirements below are rough guidelines.

- All Apple Silicon (M1, M2, etc) Macs work, but 16GB+ memory is recommended.
- AMD GPUs are supported on Linux only. The VRAM requirements are the same as Nvidia GPUs.

!!! info "Hardware Requirements (Windows/Linux)"

    === "SD1.5 - 512×512"

        - GPU: Nvidia 10xx series or later, 4GB+ VRAM.
        - Memory: At least 8GB RAM.
        - Disk: 10GB for base installation plus 30GB for models.

    === "SDXL - 1024×1024"

        - GPU: Nvidia 20xx series or later, 8GB+ VRAM.
        - Memory: At least 16GB RAM.
        - Disk: 10GB for base installation plus 100GB for models.

    === "FLUX - 1024×1024"

        - GPU: Nvidia 20xx series or later, 10GB+ VRAM.
        - Memory: At least 32GB RAM.
        - Disk: 10GB for base installation plus 200GB for models.

More detail on system requirements can be found [here](./requirements.md).

## Step 2: Download

Download the most launcher for your operating system:

- [Download for Windows](https://download.invoke.ai/Invoke%20Community%20Edition.exe)
- [Download for macOS](https://download.invoke.ai/Invoke%20Community%20Edition.dmg)
- [Download for Linux](https://download.invoke.ai/Invoke%20Community%20Edition.AppImage)

## Step 3: Install or Update

Run the launcher you just downloaded, click **Install** and follow the instructions to get set up.

If you have an existing Invoke installation, you can select it and let the launcher manage the install. You'll be able to update or launch the installation.

!!! warning "Problem running the launcher on macOS"

    macOS may not allow you to run the launcher. We are working to resolve this by signing the launcher executable. Until that is done, you can either use the [legacy scripts](./legacy_scripts.md) to install, or manually flag the launcher as safe:

    - Open the **Invoke-Installer-mac-arm64.dmg** file.
    - Drag the launcher to **Applications**.
    - Open a terminal.
    - Run `xattr -d 'com.apple.quarantine' /Applications/Invoke\ Community\ Edition.app`.

    You should now be able to run the launcher.

## Step 4: Launch

Once installed, click **Finish**, then **Launch** to start Invoke.

The very first run after an installation or update will take a few extra moments to get ready.

!!! tip "Server Mode"

    The launcher runs Invoke as a desktop application. You can enable **Server Mode** in the launcher's settings to disable this and instead access the UI through your web browser.

## Step 5: Install Models

With Invoke started up, you'll need to install some models.

The quickest way to get started is to install a **Starter Model** bundle. If you already have a model collection, Invoke can use it.

!!! info "Install Models"

    === "Install a Starter Model bundle"

        1. Go to the **Models** tab.
        2. Click **Starter Models** on the right.
        3. Click one of the bundles to install its models. Refer to the [system requirements](#step-1-confirm-system-requirements) if you're unsure which model architecture will work for your system.

    === "Use my model collection"

        4. Go to the **Models** tab.
        5. Click **Scan Folder** on the right.
        6. Paste the path to your models collection and click **Scan Folder**.
        7. With **In-place install** enabled, Invoke will leave the model files where they are. If you disable this, **Invoke will move the models into its own folders**.

You’re now ready to start creating!

## Step 6: Learn the Basics

We recommend watching our [Getting Started Playlist](https://www.youtube.com/playlist?list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO). It covers essential features and workflows, including:

- Generating your first image.
- Using control layers and reference guides.
- Refining images with advanced workflows.

## Troubleshooting

If installation fails, retrying the install in Repair Mode may fix it. There's a checkbox to enable this on the Review step of the install flow.

If that doesn't fix it, [clearing the `uv` cache](https://docs.astral.sh/uv/reference/cli/#uv-cache-clean) might do the trick:

- Open and start the dev console (button at the bottom-left of the launcher).
- Run `uv cache clean`.
- Retry the installation. Enable Repair Mode for good measure.

If you are still unable to install, try installing to a different location and see if that works.

If you still have problems, ask for help on the Invoke [discord](https://discord.gg/ZmtBAhwWhy).

## Other Installation Methods

- You can install the Invoke application as a python package. See our [manual install](./manual.md) docs.
- You can run Invoke with docker. See our [docker install](./docker.md) docs.
- You can still use our legacy scripts to install and run Invoke. See the [legacy scripts](./legacy_scripts.md) docs.

## Need Help?

- Visit our [Support Portal](https://support.invoke.ai).
- Watch the [Getting Started Playlist](https://www.youtube.com/playlist?list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO).
- Join the conversation on [Discord][discord link].

[discord link]: https://discord.gg/ZmtBAhwWhy
