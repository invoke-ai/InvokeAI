---
title: Pull-Request
---

# :octicons-git-pull-request-16: Pull-Request

## pre-requirements

To follow the steps in this tutorial you will need:

-   [GitHub](https://github.com) account
-   [git](https://git-scm.com/downloads) source controll
-   Text / Code Editor (personally I preffer
    [Visual Studio Code](https://code.visualstudio.com/Download))
-   Terminal:
    -   If you are on Linux/MacOS you can use bash or zsh
    -   for Windows Users the commands are written for PowerShell

## Fork Repository

The first step to be done if you want to contribute to InvokeAI, is to fork the
rpeository.

Since you are already reading this doc, the easiest way to do so is by clicking
[here](https://github.com/invoke-ai/InvokeAI/fork). You could also open
[InvokeAI](https://github.com/invoke-ai/InvoekAI) and click on the "Fork" Button
in the top right.

## Clone your fork

After you forked the Repository, you should clone it to your dev machine:

=== ":fontawesome-brands-linux:Linux / :simple-apple:macOS"

    ``` sh
    git clone https://github.com/<github username>/InvokeAI \
    && cd InvokeAI
    ```

=== ":fontawesome-brands-windows:Windows"

    ``` powershell
    git clone https://github.com/<github username>/InvokeAI `
    && cd InvokeAI
    ```

## Install in Editable Mode

To install InvokeAI in editable mode, (as always) we recommend to create and
activate a venv first. Afterwards you can install the InvokeAI Package,
including dev and docs extras in editable mode, follwed by the installation of
the pre-commit hook:

=== ":fontawesome-brands-linux:Linux / :simple-apple:macOS"

    ``` sh
    python -m venv .venv \
      --prompt InvokeAI \
      --upgrade-deps \
    && source .venv/bin/activate \
    && pip install \
      --upgrade-deps \
      --use-pep517 \
      --editable=".[dev,docs]" \
    && pre-commit install
    ```

=== ":fontawesome-brands-windows:Windows"

    ``` powershell
    python -m venv .venv `
      --prompt InvokeAI `
      --upgrade-deps `
    && .venv/scripts/activate.ps1 `
    && pip install `
      --upgrade `
      --use-pep517 `
      --editable=".[dev,docs]" `
    && pre-commit install
    ```

## Create a branch

Make sure you are on main branch, from there create your feature branch:

=== ":fontawesome-brands-linux:Linux / :simple-apple:macOS"

    ``` sh
    git checkout main \
    && git pull \
    && git checkout -B <branch name>
    ```

=== ":fontawesome-brands-windows:Windows"

    ``` powershell
    git checkout main `
    && git pull `
    && git checkout -B <branch name>
    ```

## Commit your changes

When you are done with adding / updating content, you need to commit those
changes to your repository before you can actually open an PR:

```{ .sh .annotate }
git add <files you have changed> # (1)!
git commit -m "A commit message which describes your change"
git push
```

1. Replace this with a space seperated list of the files you changed, like:
   `README.md foo.sh bar.json baz`

## Create a Pull Request

After pushing your changes, you are ready to create a Pull Request. just head
over to your fork on [GitHub](https://github.com), which should already show you
a message that there have been recent changes on your feature branch and a green
button which you could use to create the PR.

The default target for your PRs would be the main branch of
[invoke-ai/InvokeAI](https://github.com/invoke-ai/InvokeAI)

Another way would be to create it in VS-Code or via the GitHub CLI (or even via
the GitHub CLI in a VS-Code Terminal Window 🤭):

```sh
gh pr create
```

The CLI will inform you if there are still unpushed commits on your branch. It
will also prompt you for things like the the Title and the Body (Description) if
you did not already pass them as arguments.
