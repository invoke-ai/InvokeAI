import os
from typing import Any

# Suppress the HuggingFace tokenizers fork-after-parallelism warning. The Rust
# ``tokenizers`` library warms a thread pool the first time a tokenizer is used
# (e.g. the UMT5 / T5 text encoder during Wan / FLUX / SD3 conditioning), then
# complains every time we fork() afterwards — which we do, on every MP4 encode,
# because imageio's FFMPEG plugin shells out to ffmpeg via subprocess.Popen.
# The warning is harmless (the child correctly falls back to single-threaded
# tokenization before exec()) but it spams the log on every video generation.
#
# This MUST execute before any HF library is imported. The pyproject console-script
# (``invokeai-web = invokeai.app.run_app:run_app``) reaches this module first via
# ``from invokeai.app.run_app import run_app``, so setting the env var at module
# level — not inside ``run_app()`` — guarantees it lands before any transitive HF
# import. Use ``setdefault`` so anyone who explicitly exports ``true`` upstream
# keeps their value.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ROCm: run MIOpen's convolution-kernel search in FAST mode. The default mode benchmarks
# every candidate kernel the first time a convolution shape is seen, which on a VAE-shaped
# workload (many layers x spatial tiles x temporal chunks, separately per dtype) costs minutes
# of "warm-up" per new shape set, and torch's MIOpen ``chooseAlgorithm`` falls back to a
# *global* ``emptyCache`` whenever a candidate's workspace allocation fails - on multi-GPU
# installs that convoys the peer device mid-step (see ``install_peer_aware_empty_cache``).
# FAST looks the shape up in MIOpen's find-db and takes the first viable kernel otherwise;
# with a warmed find-db the difference in kernel quality is negligible. MIOpen reads the
# variable when it is first used, so, like the tokenizer flag, it is set at module level -
# before torch is even imported - rather than inside ``run_app()``, so no early import can run
# a convolution ahead of it. On
# CUDA/CPU/MPS builds MIOpen is absent and the variable is inert. ``setdefault`` keeps an
# explicitly exported value (e.g. ``MIOPEN_FIND_MODE=NORMAL`` to re-tune the find-db).
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")


def get_app():
    """Import the app and event loop. We wrap this in a function to more explicitly control when it happens, because
    importing from api_app does a bunch of stuff - it's more like calling a function than importing a module.
    """
    from invokeai.app.api_app import app, loop

    return app, loop


def run_app() -> None:
    """The main entrypoint for the app."""
    import asyncio
    import sys
    import threading
    import traceback

    from invokeai.frontend.cli.arg_parser import InvokeAIArgs

    # Parse the CLI arguments before doing anything else, which ensures CLI args correctly override settings from other
    # sources like `invokeai.yaml` or env vars.
    InvokeAIArgs.parse_args()

    import uvicorn

    from invokeai.app.services.config.config_default import get_config
    from invokeai.app.util.torch_cuda_allocator import (
        apply_rocm_windows_allocator_default,
        configure_torch_cuda_allocator,
    )
    from invokeai.backend.util.logging import InvokeAILogger

    # Load config.
    app_config = get_config()

    logger = InvokeAILogger.get_logger(config=app_config)

    # Configure the torch CUDA memory allocator.
    # NOTE: It is important that this happens before torch is imported.
    if app_config.pytorch_cuda_alloc_conf:
        configure_torch_cuda_allocator(app_config.pytorch_cuda_alloc_conf, logger)
    else:
        apply_rocm_windows_allocator_default(logger)

    # This import must happen after configure_torch_cuda_allocator() is called, because the module imports torch.
    from invokeai.app.invocations.baseinvocation import InvocationRegistry
    from invokeai.app.invocations.load_custom_nodes import load_custom_nodes
    from invokeai.backend.util.devices import TorchDevice

    torch_device_name = TorchDevice.get_generation_devices_summary(app_config.generation_devices)
    logger.info(f"Using torch device: {torch_device_name}")

    # Import from startup_utils here to avoid importing torch before configure_torch_cuda_allocator() is called.
    from invokeai.app.util.startup_utils import (
        apply_monkeypatches,
        check_cudnn,
        enable_dev_reload,
        find_open_port,
        register_mime_types,
    )
    from invokeai.backend.krea2.attention import resolve_krea2_sdpa_backends

    # Find an open port, and modify the config accordingly.
    first_open_port = find_open_port(app_config.port)
    if app_config.port != first_open_port:
        orig_config_port = app_config.port
        app_config.port = first_open_port
        logger.warning(f"Port {orig_config_port} is already in use. Using port {app_config.port}.")

    # Miscellaneous startup tasks.
    apply_monkeypatches()
    register_mime_types()
    check_cudnn(logger)
    # Fail here rather than inside a generation: the value is read per generation, so a typo would
    # otherwise surface as a failed queue item minutes after the server came up.
    resolve_krea2_sdpa_backends()

    # Initialize the app and event loop.
    app, loop = get_app()

    # Load custom nodes. This must be done after importing the Graph class, which itself imports all modules from the
    # invocations module. The ordering here is implicit, but important - we want to load custom nodes after all the
    # core nodes have been imported so that we can catch when a custom node clobbers a core node.
    load_custom_nodes(custom_nodes_path=app_config.custom_nodes_path, logger=logger)

    # Check all invocations and ensure their outputs are registered.
    for invocation in InvocationRegistry.get_invocation_classes():
        invocation_type = invocation.get_type()
        output_annotation = invocation.get_output_annotation()
        if output_annotation not in InvocationRegistry.get_output_classes():
            logger.warning(
                f'Invocation "{invocation_type}" has unregistered output class "{output_annotation.__name__}"'
            )

    if app_config.dev_reload:
        # load_custom_nodes seems to bypass jurrigged's import sniffer, so be sure to call it *after* they're already
        # imported.
        enable_dev_reload(custom_nodes_path=app_config.custom_nodes_path)

    # When running behind a reverse proxy that serves the app under a sub-path (e.g. `/invoke`),
    # wrap the app so route matching and openapi/docs work for both proxy styles (strip & preserve).
    # Also honor X-Forwarded-* headers. Only enabled when `base_url` is set, so default installations
    # are unaffected.
    from invokeai.app.api_app import SubPathASGIMiddleware

    asgi_app: Any = app
    proxy_kwargs: dict[str, Any] = {}
    if app_config.base_url:
        asgi_app = SubPathASGIMiddleware(app, app_config.base_url)
        # `proxy_headers=True` is already uvicorn's default; we only need to widen the trusted proxies.
        proxy_kwargs = {"forwarded_allow_ips": app_config.forwarded_allow_ips}

    # Start the server.
    config = uvicorn.Config(
        app=asgi_app,
        host=app_config.host,
        port=app_config.port,
        loop="asyncio",
        log_level=app_config.log_level_network,
        ssl_certfile=app_config.ssl_certfile,
        ssl_keyfile=app_config.ssl_keyfile,
        **proxy_kwargs,
    )
    server = uvicorn.Server(config)

    # replace uvicorn's loggers with InvokeAI's for consistent appearance
    uvicorn_logger = InvokeAILogger.get_logger("uvicorn")
    uvicorn_logger.handlers.clear()
    for hdlr in logger.handlers:
        uvicorn_logger.addHandler(hdlr)

    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        logger.info("InvokeAI shutting down...")
        # Gracefully shut down services (e.g. model download and install managers) so that any
        # active work is completed or cleanly cancelled before the process exits.
        from invokeai.app.api.dependencies import ApiDependencies

        ApiDependencies.shutdown()

        # Cancel any pending asyncio tasks (e.g. socket.io ping tasks) so that loop.close() does
        # not emit "Task was destroyed but it is pending!" warnings for each one.
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        # Shut down the asyncio default thread executor. asyncio.to_thread() (used e.g. in the
        # session queue for SQLite operations during generation) creates non-daemon threads via the
        # event loop's default ThreadPoolExecutor. Without this call those threads remain alive and
        # cause threading._shutdown() to hang indefinitely after the process's main code finishes.
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

        # After graceful shutdown, log any non-daemon threads that are still alive. These are the
        # threads that will cause Python's threading._shutdown() to block, preventing the process
        # from exiting cleanly. This helps identify threads that need to be fixed or joined.
        frames = sys._current_frames()
        for thread in threading.enumerate():
            if thread.daemon or thread is threading.main_thread():
                continue
            frame = frames.get(thread.ident)
            stack = "".join(traceback.format_stack(frame)) if frame else "(no frame available)"
            logger.warning(
                f"Non-daemon thread still alive after shutdown: {thread.name!r} "
                f"(ident={thread.ident})\nStack trace:\n{stack}"
            )
