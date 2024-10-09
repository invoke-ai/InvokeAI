# Dev Environment

To make changes to Invoke's backend, frontend, or documentation, you'll need to set up a dev environment.

If you just want to use Invoke, you should use the [installer][installer link].

!!! info "Why do I need the frontend toolchain?"

    The repo doesn't contain a build of the frontend. You'll be responsible for rebuilding it every time you pull in new changes, or run it in dev mode (which incurs a substantial performance penalty).

!!! warning

    Invoke uses a SQLite database. When you run the application as a dev install, you accept responsibility for your database. This means making regular backups (especially before pulling) and/or fixing it yourself in the event that a PR introduces a schema change.

    If you don't need to persist your db, you can use an ephemeral in-memory database by setting `use_memory_db: true` in your `invokeai.yaml` file. You'll also want to set `scan_models_on_startup: true` so that your models are registered on startup.

## Setup

1. Run through the [requirements][requirements link].
1. [Fork and clone][forking link] the [InvokeAI repo][repo link].
1. Create an directory for user data (images, models, db, etc). This is typically at `~/invokeai`, but if you already have a non-dev install, you may want to create a separate directory for the dev install.
1. Create a python virtual environment inside the directory you just created:

   ```sh
   python3 -m venv .venv --prompt InvokeAI-Dev
   ```

1. Activate the venv (you'll need to do this every time you want to run the app):

   ```sh
   source .venv/bin/activate
   ```

1. Install the repo as an [editable install][editable install link]:

   ```sh
   pip install -e ".[dev,test,xformers]" --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu121
   ```

   Refer to the [manual installation][manual install link]] instructions for more determining the correct install options. `xformers` is optional, but `dev` and `test` are not.

1. Install the frontend dev toolchain:

   - [`nodejs`](https://nodejs.org/) (recommend v20 LTS)
   - [`pnpm`](https://pnpm.io/installation#installing-a-specific-version) (must be v8 - not v9!)

1. Do a production build of the frontend:

   ```sh
   pnpm build
   ```

1. Start the application:

   ```sh
   python scripts/invokeai-web.py
   ```

1. Access the UI at `localhost:9090`.

## Updating the UI

You'll need to run `pnpm build` every time you pull in new changes. Another option is to skip the build and instead run the app in dev mode:

```sh
pnpm dev
```

This starts a dev server at `localhost:5173`, which you will use instead of `localhost:9090`.

The dev mode is substantially slower than the production build but may be more convenient if you just need to test things out.

## Documentation

The documentation is built with `mkdocs`. To preview it locally, you need a additional set of packages installed.

```sh
# after activating the venv
pip install -e ".[docs]"
```

Then, you can start a live docs dev server, which will auto-refresh when you edit the docs:

```sh
mkdocs serve
```

On macOS and Linux, there is a `make` target for this:

```sh
make docs
```

[installer link]: ../installation/installer.md
[forking link]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo
[requirements link]: ../installation/requirements.md
[repo link]: https://github.com/invoke-ai/InvokeAI
[manual install link]: ../installation/manual.md
[editable install link]: https://pip.pypa.io/en/latest/cli/pip_install/#cmdoption-e
