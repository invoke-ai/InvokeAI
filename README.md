<div align="center">

![project hero](https://github.com/invoke-ai/InvokeAI/assets/31807370/6e3728c7-e90e-4711-905c-3b55844ff5be)

# Invoke - Professional Creative AI Tools for Visual Media

#### To learn more about Invoke, or implement our Business solutions, visit [invoke.com]

[![discord badge]][discord link] [![latest release badge]][latest release link] [![github stars badge]][github stars link] [![github forks badge]][github forks link] [![CI checks on main badge]][CI checks on main link] [![latest commit to main badge]][latest commit to main link] [![github open issues badge]][github open issues link] [![github open prs badge]][github open prs link] [![translation status badge]][translation status link]

</div>

Invoke is a leading creative engine built to empower professionals and enthusiasts alike. Generate and create stunning visual media using the latest AI-driven technologies. Invoke offers an industry leading web-based UI, and serves as the foundation for multiple commercial products.

Invoke is available in two editions:

| **Community Edition**                                                                                                      | **Professional Edition**                                                                            |
|----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **For users looking for a locally installed, self-hosted and self-managed service**                                         | **For users or teams looking for a cloud-hosted, fully managed service**                            |
| - Free to use under a commercially-friendly license                                                                         | - Monthly subscription fee with three different plan levels                                         |
| - Download and install on compatible hardware                                                                               | - Offers additional benefits, including multi-user support, improved model training, and more                          |
| - Includes all core studio features: generate, refine, iterate on images, and build workflows                               | - Hosted in the cloud for easy, secure model access and scalability                                               |
| Quick Start -> [Installation and Updates][installation docs]                                                                     | More Information -> [www.invoke.com/pricing](https://www.invoke.com/pricing)                        |


![Highlighted Features - Canvas and Workflows](https://github.com/invoke-ai/InvokeAI/assets/31807370/708f7a82-084f-4860-bfbe-e2588c53548d)

# Documentation
| **Quick Links**                                                                                                      | 
|----------------------------------------------------------------------------------------------------------------------------|
|  [Installation and Updates][installation docs] - [Documentation and Tutorials][docs home] - [Bug Reports][github issues] - [Contributing][contributing docs]  | 

</div>

## How to Install and Update Invoke Community Edition

Welcome to Invoke! Follow these steps to install, update, and get started with Invoke. 

You can also follow out [installation documentation][installation docs].

### Step 1: Confirm System Requirements

**Before you start, ensure your system meets the following requirements:**

| **Minimum Requirements**                                                                                                      | 
|----------------------------------------------------------------------------------------------------------------------------|
|**Operating System:** Windows 10+, macOS 11.0+, or Linux (Ubuntu 20.04+ recommended).|
|**GPU:** |
|  - **NVIDIA:** GTX 1060 or higher (6GB VRAM), CUDA 11.3+. |
|  - **AMD:** RX 5700 or higher, ROCm 5.4+ (Linux only).|
|**RAM:** 8GB or more.|
|**Disk Space:** 10GB free for installation (30GB+ recommended for models).|

**Recommended for Best Performance**
- **GPU:** NVIDIA RTX 20 Series or AMD RDNA2 GPUs with 8GB+ VRAM.
- **RAM:** 16GB or more.
- **Disk Space:** 100GB if working with multiple models.

### Step 2: Download the Launcher

Download the most recent launcher for your operating system:

- [Download for Windows](https://download.invoke.ai/Invoke-Installer-windows-x64.exe)
- [Download for macOS](https://download.invoke.ai/Invoke-Installer-mac-arm64.dmg)
- [Download for Linux](https://download.invoke.ai/Invoke-Installer-linux-x86_64.AppImage)

### Step 3: Install or Update Invoke

Run the launcher you just downloaded. You’ll have two options:

1. **Launch / update from an existing installation:**
   - If you installed Invoke previously, click *Select an existing installation* to connect to it. You'll be able to update or launch the existing installation.
   
2. **Launch from a fresh installation:**
   - Click *Install* to set up a new instance of Invoke.
   - Follow the on-screen instructions to complete the setup.

### Step 4: Run Invoke from Your Browser

Once installed, click Finish, then Launch to start Invoke.

- The very first run after an installation will take a few extra moments to get ready. It will be faster after the first run.

### Step 5: Install a Starter Model Pack or Locate Models On Your Hard Drive

After launching Invoke:

1. Go to the **Model Manager** tab.
2. Install one of the suggested Starter Model packs. If you already have models installed on your hard drive, you can provide that folder location and Invoke will automatically add those models to your studio.

You’re now ready to start creating!

### Step 6: Learn the Basics

We recommend watching the following resources to get started:

[**Getting Started Playlist**](https://www.youtube.com/playlist?list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO)

This playlist covers essential features and workflows, including:

- Generating your first image.
- Using control layers and reference guides.
- Refining images with advanced workflows.

---

## Advanced Installation Options

### Manual Installation

For detailed instructions on manual setup, see our [Manual Installation Guide](https://invoke-ai.github.io/InvokeAI/installation/manual/).

### Docker Installation

Run Invoke in a containerized environment with our [Docker Installation Guide](https://invoke-ai.github.io/InvokeAI/installation/docker/).

---

## Need Help?

- Visit our [Support Portal](https://support.invoke.ai).
- Watch the [Getting Started Playlist](https://www.youtube.com/playlist?list=PLvWK1Kc8iXGrQy8r9TYg6QdUuJ5MMx-ZO).
- Join the conversation on [Discord][discord link].


## Troubleshooting, FAQ and Support

Please review our [FAQ][faq] for solutions to common installation problems and other issues.

For more help, please join our [Discord][discord link].

## Features

Full details on features can be found in [our documentation][features docs].

### Web Server & UI

Invoke runs a locally hosted web server & React UI with an industry-leading user experience.

### Unified Canvas

The Unified Canvas is a fully integrated canvas implementation with support for all core generation capabilities, in/out-painting, brush tools, and more. This creative tool unlocks the capability for artists to create with AI as a creative collaborator, and can be used to augment AI-generated imagery, sketches, photography, renders, and more.

### Workflows & Nodes

Invoke offers a fully featured workflow management solution, enabling users to combine the power of node-based workflows with the easy of a UI. This allows for customizable generation pipelines to be developed and shared by users looking to create specific workflows to support their production use-cases.

### Board & Gallery Management

Invoke features an organized gallery system for easily storing, accessing, and remixing your content in the Invoke workspace. Images can be dragged/dropped onto any Image-base UI element in the application, and rich metadata within the Image allows for easy recall of key prompts or settings used in your workflow.

### Other features

- Support for both ckpt and diffusers models
- SD1.5, SD2.0, SDXL, and FLUX support
- Upscaling Tools
- Embedding Manager & Support
- Model Manager & Support
- Workflow creation & management
- Node-Based Architecture

## Contributing

Anyone who wishes to contribute to this project - whether documentation, features, bug fixes, code cleanup, testing, or code reviews - is very much encouraged to do so.

Get started with contributing by reading our [contribution documentation][contributing docs], joining the [#dev-chat] or the GitHub discussion board.

We hope you enjoy using Invoke as much as we enjoy creating it, and we hope you will elect to become part of our community.

## Thanks

Invoke is a combined effort of [passionate and talented people from across the world][contributors]. We thank them for their time, hard work and effort.

Original portions of the software are Copyright © 2024 by respective contributors.

[features docs]: https://invoke-ai.github.io/InvokeAI/features/database/
[faq]: https://invoke-ai.github.io/InvokeAI/faq/
[contributors]: https://invoke-ai.github.io/InvokeAI/contributing/contributors/
[invoke.com]: https://www.invoke.com/about
[github issues]: https://github.com/invoke-ai/InvokeAI/issues
[docs home]: https://invoke-ai.github.io/InvokeAI
[installation docs]: https://invoke-ai.github.io/InvokeAI/installation/
[#dev-chat]: https://discord.com/channels/1020123559063990373/1049495067846524939
[contributing docs]: https://invoke-ai.github.io/InvokeAI/contributing/
[CI checks on main badge]: https://flat.badgen.net/github/checks/invoke-ai/InvokeAI/main?label=CI%20status%20on%20main&cache=900&icon=github
[CI checks on main link]: https://github.com/invoke-ai/InvokeAI/actions?query=branch%3Amain
[discord badge]: https://flat.badgen.net/discord/members/ZmtBAhwWhy?icon=discord
[discord link]: https://discord.gg/ZmtBAhwWhy
[github forks badge]: https://flat.badgen.net/github/forks/invoke-ai/InvokeAI?icon=github
[github forks link]: https://useful-forks.github.io/?repo=invoke-ai%2FInvokeAI
[github open issues badge]: https://flat.badgen.net/github/open-issues/invoke-ai/InvokeAI?icon=github
[github open issues link]: https://github.com/invoke-ai/InvokeAI/issues?q=is%3Aissue+is%3Aopen
[github open prs badge]: https://flat.badgen.net/github/open-prs/invoke-ai/InvokeAI?icon=github
[github open prs link]: https://github.com/invoke-ai/InvokeAI/pulls?q=is%3Apr+is%3Aopen
[github stars badge]: https://flat.badgen.net/github/stars/invoke-ai/InvokeAI?icon=github
[github stars link]: https://github.com/invoke-ai/InvokeAI/stargazers
[latest commit to main badge]: https://flat.badgen.net/github/last-commit/invoke-ai/InvokeAI/main?icon=github&color=yellow&label=last%20dev%20commit&cache=900
[latest commit to main link]: https://github.com/invoke-ai/InvokeAI/commits/main
[latest release badge]: https://flat.badgen.net/github/release/invoke-ai/InvokeAI/development?icon=github
[latest release link]: https://github.com/invoke-ai/InvokeAI/releases/latest
[translation status badge]: https://hosted.weblate.org/widgets/invokeai/-/svg-badge.svg
[translation status link]: https://hosted.weblate.org/engage/invokeai/
[nvidia docker docs]: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
[amd docker docs]: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html
