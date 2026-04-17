# FAQ

If the troubleshooting steps on this page don't get you up and running, please either [create an issue] or hop on [discord] for help.

## How to Install

Follow the [Quick Start guide](./installation/quick_start.md) to install Invoke.

## Downloading models and using existing models

The Model Manager tab in the UI provides a few ways to install models, including using your already-downloaded models. You'll see a popup directing you there on first startup. For more information, see the [model install docs].

## Missing models after updating from v3

If you find some models are missing after updating from v3, it's likely they weren't correctly registered before the update and didn't get picked up in the migration.

You can use the `Scan Folder` tab in the Model Manager UI to fix this. The models will either be in the old, now-unused `autoimport` folder, or your `models` folder.

- Find and copy your install's old `autoimport` folder path, install the main install folder.
- Go to the Model Manager and click `Scan Folder`.
- Paste the path and scan.
- IMPORTANT: Uncheck `Inplace install`.
- Click `Install All` to install all found models, or just install the models you want.

Next, find and copy your install's `models` folder path (this could be your custom models folder path, or the `models` folder inside the main install folder).

Follow the same steps to scan and import the missing models.

## Slow generation

- Check the [system requirements] to ensure that your system is capable of generating images.
- Follow the [Low-VRAM mode guide](./features/low-vram.md) to optimize performance.
- Check that your generations are happening on your GPU (if you have one). Invoke will log what is being used for generation upon startup. If your GPU isn't used, re-install to and ensure you select the appropriate GPU option.
- If you are on Windows with an Nvidia GPU, you may have exceeded your GPU's VRAM capacity and are triggering Nvidia's "sysmem fallback". There's a guide to opt out of this behaviour in the [Low-VRAM mode guide](./features/low-vram.md).

## Triton error on startup

This can be safely ignored. Invoke doesn't use Triton, but if you are on Linux and wish to dismiss the error, you can install Triton.

## Unable to Copy on Firefox

Firefox does not allow Invoke to directly access the clipboard by default. As a result, you may be unable to use certain copy functions. You can fix this by configuring Firefox to allow access to write to the clipboard:

- Go to `about:config` and click the Accept button
- Search for `dom.events.asyncClipboard.clipboardItem`
- Set it to `true` by clicking the toggle button
- Restart Firefox

## Replicate image found online

Most example images with prompts that you'll find on the internet have been generated using different software, so you can't expect to get identical results. In order to reproduce an image, you need to replicate the exact settings and processing steps, including (but not limited to) the model, the positive and negative prompts, the seed, the sampler, the exact image size, any upscaling steps, etc.

## Invalid configuration file

Everything seems to install ok, you get a `ValidationError` when starting up the app.

This is caused by an invalid setting in the `invokeai.yaml` configuration file. The error message should tell you what is wrong.

Check the [configuration docs] for more detail about the settings and how to specify them.

## Out of Memory Errors

The models are large, VRAM is expensive, and you may find yourself faced with Out of Memory errors when generating images. Follow our [Low-VRAM mode guide](./features/low-vram.md) to configure Invoke to prevent these.

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

[model install docs]: ./installation/models.md
[system requirements]: ./installation/requirements.md
[create an issue]: https://github.com/invoke-ai/InvokeAI/issues
[discord]: https://discord.gg/ZmtBAhwWhy
[configuration docs]: ./configuration.md
