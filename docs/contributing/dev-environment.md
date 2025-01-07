# Dev Environment

To make changes to Invoke's backend, frontend or documentation, you'll need to set up a dev environment.

If you only want to make changes to the docs site, you can skip the frontend dev environment setup as described in the below guide.

If you just want to use Invoke, you should use the [launcher][launcher link].

!!! warning

    Invoke uses a SQLite database. When you run the application as a dev install, you accept responsibility for your database. This means making regular backups (especially before pulling) and/or fixing it yourself in the event that a PR introduces a schema change.

    If you don't need to persist your db, you can use an ephemeral in-memory database by setting `use_memory_db: true` in your `invokeai.yaml` file. You'll also want to set `scan_models_on_startup: true` so that your models are registered on startup.

## Setup

1. Run through the [requirements][requirements link].

2. [Fork and clone][forking link] the [InvokeAI repo][repo link].

3. Create an directory for user data (images, models, db, etc). This is typically at `~/invokeai`, but if you already have a non-dev install, you may want to create a separate directory for the dev install.

4. Follow the [manual install][manual install link] guide, with some modifications to the install command:

      - Use `.` instead of `invokeai` to install from the current directory. You don't need to specify the version.

      - Add `-e` after the `install` operation to make this an [editable install][editable install link]. That means your changes to the python code will be reflected when you restart the Invoke server.

      - When installing the `invokeai` package, add the `dev`, `test` and `docs` package options to the package specifier. You may or may not need the `xformers` option - follow the manual install guide to figure that out. So, your package specifier will be either `".[dev,test,docs]"` or `".[dev,test,docs,xformers]"`. Note the quotes!

     With the modifications made, the install command should look something like this:

      ```sh
      uv pip install -e ".[dev,test,docs,xformers]" --python 3.11 --python-preference only-managed --index=https://download.pytorch.org/whl/cu124 --reinstall
      ```

5. At this point, you should have Invoke installed, a venv set up and activated, and the server running. But you will see a warning in the terminal that no UI was found. If you go to the URL for the server, you won't get a UI.

      This is because the UI build is not distributed with the source code. You need to build it manually. End the running server instance.

      If you only want to edit the docs, you can stop here and skip to the **Documentation** section below.

6. Install the frontend dev toolchain:

      - [`nodejs`](https://nodejs.org/) (v20+)

      - [`pnpm`](https://pnpm.io/8.x/installation) (must be v8 - not v9!)

7. Do a production build of the frontend:

      ```sh
      cd <PATH_TO_INVOKEAI_REPO>/invokeai/frontend/web
      pnpm i
      pnpm build
      ```

8. Restart the server and navigate to the URL. You should get a UI. After making changes to the python code, restart the server to see those changes.

## Updating the UI

You'll need to run `pnpm build` every time you pull in new changes.

Another option is to skip the build and instead run the UI in dev mode:

```sh
pnpm dev
```

This starts a vite dev server for the UI at `127.0.0.1:5173`, which you will use instead of `127.0.0.1:9090`.

The dev mode is substantially slower than the production build but may be more convenient if you just need to test things out. It will hot-reload the UI as you make changes to the frontend code. Sometimes the hot-reload doesn't work, and you need to manually refresh the browser tab.

## Documentation

The documentation is built with `mkdocs`. It provides a hot-reload dev server for the docs. Start it with `mkdocs serve`.

[launcher link]: ../installation/quick_start.md
[forking link]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo
[requirements link]: ../installation/requirements.md
[repo link]: https://github.com/invoke-ai/InvokeAI
[manual install link]: ../installation/manual.md
[editable install link]: https://pip.pypa.io/en/latest/cli/pip_install/#cmdoption-e
