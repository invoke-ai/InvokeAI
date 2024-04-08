# Automatic Install

The installer is used for both new installs and updates.

Both release and pre-release versions can be installed using it. It also supports install a wheel if needed.

Be sure to review the [installation requirements] and ensure your system has everything it needs to install Invoke.

## Getting the Latest Installer

Download the `InvokeAI-installer-vX.Y.Z.zip` file from the [latest release] page. It is at the bottom of the page, under **Assets**.

After unzipping the installer, you should have a `InvokeAI-Installer` folder with some files inside, including `install.bat` and `install.sh`.

## Running the Installer

!!! tip

    Windows users should first double-click the `WinLongPathsEnabled.reg` file to prevent a failed installation due to long file paths.

Double-click the install script:

=== "Windows"

    ```sh
    install.bat
    ```

=== "Linux/macOS"

    ```sh
    install.sh
    ```

!!! info "Running the Installer from the commandline"

    You can also run the install script from cmd/powershell (Windows) or terminal (Linux/macOS).

!!! warning "Untrusted Publisher (Windows)"

    You may get a popup saying the file comes from an `Untrusted Publisher`. Click `More Info` and `Run Anyway` to get past this.

The installation process is simple, with a few prompts:

- Select the version to install. Unless you have a specific reason to install a specific version, select the default (the latest version).
- Select location for the install. Be sure you have enough space in this folder for the base application, as described in the [installation requirements].
- Select a GPU device.

!!! info "Slow Installation"

    The installer needs to download several GB of data and install it all. It may appear to get stuck at 99.9% when installing `pytorch` or during a step labeled "Installing collected packages".

    If it is stuck for over 10 minutes, something has probably gone wrong and you should close the window and restart.

## Running the Application

Find the install location you selected earlier. Double-click the launcher script to run the app:

=== "Windows"

    ```sh
    invoke.bat
    ```

=== "Linux/macOS"

    ```sh
    invoke.sh
    ```

Choose the first option to run the UI. After a series of startup messages, you'll see something like this:

```
Uvicorn running on http://127.0.0.1:9090 (Press CTRL+C to quit)
```

Copy the URL into your browser and you should see the UI.

## First-time Setup

You will need to [install some models] before you can generate.

Check the [configuration docs] for details on configuring the application.

## Updating

Updating is exactly the same as installing - download the latest installer, choose the latest version and off you go.

!!! info "Dependency Resolution Issues"

    We've found that pip's dependency resolution can cause issues when upgrading packages. One very common problem was pip "downgrading" torch from CUDA to CPU, but things broke in other novel ways.

    The installer doesn't have this kind of problem, so we use it for updating as well.

## Installation Issues

If you have installation issues, please review the [FAQ]. You can also [create an issue] or ask for help on [discord].

[installation requirements]: INSTALLATION.md#installation-requirements
[FAQ]: ../help/FAQ.md
[install some models]: 050_INSTALLING_MODELS.md
[configuration docs]: ../features/CONFIGURATION.md
[latest release]: https://github.com/invoke-ai/InvokeAI/releases/latest
[create an issue]: https://github.com/invoke-ai/InvokeAI/issues
[discord]: https://discord.gg/ZmtBAhwWhy
