---
title: How to Contribute
---

## pre-requirements

To follow the steps in this tutorial you will need the following:

- [git](https://git-scm.com/downloads)
- [GitHub](https://github.com) account
- A Code Editor (personally I use Visual Studio Code)

## Fork Repository

The first step to be done if you want to contribute to InvokeAI, is to fork the
rpeository.

The easiest way to do so is by clicking
[here](https://github.com/invoke-ai/InvokeAI/fork). It is also possible by
opening [InvokeAI](https://github.com/invoke-ai/InvoekAI) and click on the
"Fork" Button in the top right.

## Clone your fork

After you forked the Repository, you should clone it to your dev machine:

=== "Linux/MacOS"

    ```sh
    git clone https://github.com/<github username>/InvokeAI \
    && cd InvokeAI
    ```

=== "Windows"

    ```powershell
    git clone https://github.com/<github username>/InvokeAI `
    && cd InvokeAI
    ```

## Install in Editable Mode

To install InvokeAI in editable mode, (as always) we recommend to create and
activate a venv first. Afterwards you can install the InvokeAI Package,
including dev and docs extras in editable mode, follwed by the installation of
the pre-commit hook:

=== "Linux/MacOS"

    ```sh
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

=== "Windows"

    ```powershell
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

=== "Linux/MacOS"

    ```sh
    git checkout main \
    && git pull \
    && git checkout -B <branch name>
    ```

=== "Windows"

    ```powershell
    git checkout main `
    && git pull `
    && git checkout -B <branch name>
    ```
