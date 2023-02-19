---
title: docs
---

# :simple-readthedocs: MkDocs-Material

If you want to contribute to the docs, there is a easy way to verify the results
of your changes before commiting them.

Just follow the steps in the [Pull-Requests](010_PULL_REQUEST.md) docs, there we
already
[create a venv and install the docs extras](010_PULL_REQUEST.md#install-in-editable-mode).
When installed it's as simple as:

```sh
mkdocs serve
```

This will build the docs locally and serve them on your local host, even
auto-refresh is included, so you can just update a doc, save it and tab to the
browser, without the needs of restarting the `mkdocs serve`.

More information about the "mkdocs flavored markdown syntax" can be found
[here](https://squidfunk.github.io/mkdocs-material/reference/).

## :material-microsoft-visual-studio-code:VS-Code

We also provide a
[launch configuration for VS-Code](../IDE-Settings/vs-code.md#launchjson) which
includes a `mkdocs serve` entrypoint as well. You also don't have to worry about
the formatting since this is automated via prettier, but this is of course not
limited to VS-Code.
