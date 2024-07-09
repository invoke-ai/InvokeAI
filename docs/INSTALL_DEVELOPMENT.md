# Developer Install

!!! warning

    InvokeAI uses a SQLite database. By running on `main`, you accept responsibility for your database. This
    means making regular backups (especially before pulling) and/or fixing it yourself in the event that a
    PR introduces a schema change.

    If you don't need persistent backend storage, you can use an ephemeral in-memory database by setting
    `use_memory_db: true` in your `invokeai.yaml` file. You'll also want to set `scan_models_on_startup: true`
    so that your models are registered on startup.

    If this is untenable, you should run the application via the official installer or a manual install of the
    python package from PyPI. These releases will not break your database.

If you have an interest in how InvokeAI works, or you would like to add features or bugfixes, you are encouraged to install the source code for InvokeAI.

!!! info "Why do I need the frontend toolchain?"

    The repo doesn't contain a build of the frontend. You'll be responsible for rebuilding it (or running it in dev mode) to use the app, as described in the [frontend dev toolchain] docs.

<h2> Installation </h2>

1. [Fork and clone] the [InvokeAI repo].
1. Follow the [manual installation] docs to create a new virtual environment for the development install.
   - Create a new folder outside the repo root for the installation and create the venv inside that folder.
   - When installing the InvokeAI package, add `-e` to the command so you get an [editable install].
1. Install the [frontend dev toolchain] and do a production build of the UI as described.
1. You can now run the app as described in the [manual installation] docs.

As described in the [frontend dev toolchain] docs, you can run the UI using a dev server. If you do this, you won't need to continually rebuild the frontend. Instead, you run the dev server and use the app with the server URL it provides.

[Fork and clone]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo
[InvokeAI repo]: https://github.com/invoke-ai/InvokeAI
[frontend dev toolchain]: ../contributing/frontend/OVERVIEW.md
[manual installation]: ./020_INSTALL_MANUAL.md
[editable install]: https://pip.pypa.io/en/latest/cli/pip_install/#cmdoption-e
