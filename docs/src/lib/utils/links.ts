export const externalLinks = {
  // Nvidia
  cudaDownloads: 'https://developer.nvidia.com/cuda-downloads',
  nvidiaRuntime: 'https://developer.nvidia.com/container-runtime',
  cudnnDocs: 'https://developer.nvidia.com/cudnn',
  cudnnSupport: 'https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html',
  cudnnDownload: 'https://developer.nvidia.com/cudnn',

  // AMD
  rocmDocs: 'https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html',
  rocmDocker: 'https://github.com/ROCm/ROCm-docker',

  // Python
  pythonDownload: 'https://www.python.org/downloads/',
  msvcRedist: 'https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist',
  uvDocs: 'https://docs.astral.sh/uv/concepts/python-versions/#installing-a-python-version',

  // InvokeAI
  discord: 'https://discord.gg/ZmtBAhwWhy',
  github: 'https://github.com/invoke-ai/InvokeAI',
  launcher: 'https://github.com/invoke-ai/launcher/releases/latest',
  support: 'https://support.invoke.ai',

  // Launcher
  launcherWindows: 'https://github.com/invoke-ai/launcher/releases/latest/download/Invoke.Community.Edition.Setup.latest.exe',
  launcherMacOS: 'https://github.com/invoke-ai/launcher/releases/latest/download/Invoke.Community.Edition-latest-arm64.dmg',
  launcherLinux: 'https://github.com/invoke-ai/launcher/releases/latest/download/Invoke.Community.Edition-latest.AppImage',
} as const;

export const internalLinks = {
  quickStart: '/getting-started/quick_start',
  lowVram: '/configuration/low-vram-mode',
  // ... etc
} as const;
