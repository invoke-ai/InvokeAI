---
title: Visual Studio Code
---

# :material-microsoft-visual-studio-code:Visual Studio Code

The Workspace Settings are stored in the project (repository) root and get
higher priorized than your user settings.

This helps to have different settings for different projects, while the user
settings get used as a default value if no workspace settings are provided.

## tasks.json

First we will create a task configuration which will create a virtual
environment and update the deps (pip, setuptools and wheel).

Into this venv we will then install the pyproject.toml in editable mode with
dev, docs and test dependencies.

```json title=".vscode/tasks.json"
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Create virtual environment",
            "detail": "Create .venv and upgrade pip, setuptools and wheel",
            "command": "python3",
            "args": [
                "-m",
                "venv",
                ".venv",
                "--prompt",
                "InvokeAI",
                "--upgrade-deps"
            ],
            "runOptions": {
                "instanceLimit": 1,
                "reevaluateOnRerun": true
            },
            "group": {
                "kind": "build"
            },
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared",
                "showReuseMessage": true,
                "clear": false
            }
        },
        {
            "label": "build InvokeAI",
            "detail": "Build pyproject.toml with extras dev, docs and test",
            "command": "${workspaceFolder}/.venv/bin/python3",
            "args": [
                "-m",
                "pip",
                "install",
                "--use-pep517",
                "--editable",
                ".[dev,docs,test]"
            ],
            "dependsOn": "Create virtual environment",
            "dependsOrder": "sequence",
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared",
                "showReuseMessage": true,
                "clear": false
            }
        }
    ]
}
```

The fastest way to build InvokeAI now is ++cmd+shift+b++

## launch.json

This file is used to define debugger configurations, so that you can one-click
launch and monitor the application, set halt points to inspect specific states,
...

```json title=".vscode/launch.json"
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "invokeai web",
            "type": "python",
            "request": "launch",
            "program": ".venv/bin/invokeai",
            "justMyCode": true
        },
        {
            "name": "invokeai cli",
            "type": "python",
            "request": "launch",
            "program": ".venv/bin/invokeai",
            "justMyCode": true
        },
        {
            "name": "mkdocs serve",
            "type": "python",
            "request": "launch",
            "program": ".venv/bin/mkdocs",
            "args": ["serve"],
            "justMyCode": true
        }
    ]
}
```

Then you only need to hit ++f5++ and the fun begins :nerd: (It is asumed that
you have created a virtual environment via the [tasks](#tasksjson) from the
previous step.)

## extensions.json

A list of recommended vscode-extensions to make your life easier:

```json title=".vscode/extensions.json"
{
    "recommendations": [
        "editorconfig.editorconfig",
        "github.vscode-pull-request-github",
        "ms-python.black-formatter",
        "ms-python.flake8",
        "ms-python.isort",
        "ms-python.python",
        "ms-python.vscode-pylance",
        "redhat.vscode-yaml",
        "tamasfe.even-better-toml",
        "eamodio.gitlens",
        "foxundermoon.shell-format",
        "timonwong.shellcheck",
        "esbenp.prettier-vscode",
        "davidanson.vscode-markdownlint",
        "yzhang.markdown-all-in-one",
        "bierner.github-markdown-preview",
        "ms-azuretools.vscode-docker",
        "mads-hartmann.bash-ide-vscode"
    ]
}
```

## settings.json

With bellow settings your files already get formated when you save them (only
your modifications if available), which will help you to not run into trouble
with the pre-commit hooks. If the hooks fail, they will prevent you from
commiting, but most hooks directly add a fixed version, so that you just need to
stage and commit them:

```json title=".vscode/settings.json"
{
    "[json]": {
        "editor.defaultFormatter": "esbenp.prettier-vscode",
        "editor.quickSuggestions": {
            "comments": false,
            "strings": true,
            "other": true
        },
        "editor.suggest.insertMode": "replace",
        "gitlens.codeLens.scopes": ["document"]
    },
    "[jsonc]": {
        "editor.defaultFormatter": "esbenp.prettier-vscode",
        "editor.formatOnSave": true,
        "editor.formatOnSaveMode": "modificationsIfAvailable"
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.formatOnSaveMode": "file"
    },
    "[toml]": {
        "editor.defaultFormatter": "tamasfe.even-better-toml",
        "editor.formatOnSave": true,
        "editor.formatOnSaveMode": "modificationsIfAvailable"
    },
    "[yaml]": {
        "editor.defaultFormatter": "esbenp.prettier-vscode",
        "editor.formatOnSave": true,
        "editor.formatOnSaveMode": "modificationsIfAvailable"
    },
    "[markdown]": {
        "editor.defaultFormatter": "esbenp.prettier-vscode",
        "editor.rulers": [80],
        "editor.unicodeHighlight.ambiguousCharacters": false,
        "editor.unicodeHighlight.invisibleCharacters": false,
        "diffEditor.ignoreTrimWhitespace": false,
        "editor.wordWrap": "on",
        "editor.quickSuggestions": {
            "comments": "off",
            "strings": "off",
            "other": "off"
        },
        "editor.formatOnSave": true,
        "editor.formatOnSaveMode": "modificationsIfAvailable"
    },
    "[shellscript]": {
        "editor.defaultFormatter": "foxundermoon.shell-format"
    },
    "[ignore]": {
        "editor.defaultFormatter": "foxundermoon.shell-format"
    },
    "editor.rulers": [88],
    "evenBetterToml.formatter.alignEntries": false,
    "evenBetterToml.formatter.allowedBlankLines": 1,
    "evenBetterToml.formatter.arrayAutoExpand": true,
    "evenBetterToml.formatter.arrayTrailingComma": true,
    "evenBetterToml.formatter.arrayAutoCollapse": true,
    "evenBetterToml.formatter.columnWidth": 88,
    "evenBetterToml.formatter.compactArrays": true,
    "evenBetterToml.formatter.compactInlineTables": true,
    "evenBetterToml.formatter.indentEntries": false,
    "evenBetterToml.formatter.inlineTableExpand": true,
    "evenBetterToml.formatter.reorderArrays": true,
    "evenBetterToml.formatter.reorderKeys": true,
    "evenBetterToml.formatter.compactEntries": false,
    "evenBetterToml.schema.enabled": true,
    "python.analysis.typeCheckingMode": "basic",
    "python.formatting.provider": "black",
    "python.languageServer": "Pylance",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests",
        "--cov=ldm",
        "--cov-branch",
        "--cov-report=term:skip-covered"
    ],
    "yaml.schemas": {
        "https://json.schemastore.org/prettierrc.json": "${workspaceFolder}/.prettierrc.yaml"
    }
}
```
