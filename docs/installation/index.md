# Installation and Updating Overview

Before installing, review the [installation requirements](./requirements.md) to ensure your system is set up properly.

See the [FAQ](../faq.md) for frequently-encountered installation issues.

If you need more help, join our [discord](https://discord.gg/ZmtBAhwWhy) or [create a GitHub issue](https://github.com/invoke-ai/InvokeAI/issues).

## Automated Installer & Updates

✅ The automated [installer](./installer.md) is the best way to install Invoke.

⬆️ The same installer is also the best way to update Invoke - simply rerun it for the same folder you installed to.

The installation process simply manages installation for the core libraries & application dependencies that run Invoke.

Models, images, or other assets in the Invoke root folder won't be affected by the installation process.

## Manual Install

If you are familiar with python and want more control over the packages that are installed, you can [install Invoke manually via PyPI](./manual.md).

Updates are managed by reinstalling the latest version through PyPi.

## Developer Install

If you want to contribute to InvokeAI, you'll need to set up a [dev environment](../contributing/dev-environment.md).

## Docker

Invoke publishes docker images. See the [docker installation guide](./docker.md) for details.

## Other Installation Guides

- [PyPatchMatch](./patchmatch.md)
- [Installing Models](./models.md)
