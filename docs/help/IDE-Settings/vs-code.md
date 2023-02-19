---
title: Visual Studio Code
---

# :material-microsoft-visual-studio-code:Visual Studio Code

The Workspace Settings are stored in the project (repository) root and get
higher priorized than your user settings.

This helps to have different settings for different projects, while the user
settings get used as a default value if no workspace settings are provided.

## launch.json

It is asumed that you have created a virtual environment as `.venv`:

```sh
python -m venv .venv --prompt="InvokeAI" --upgrade-deps
```

This is the most simplified version of launching `invokeai --web` with the
debugger attached:

```json title=".vscode/launch.json"
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "invokeai --web",
            "type": "python",
            "request": "launch",
            "program": ".venv/bin/invokeai",
            "args": ["--web"],
            "justMyCode": true
        }
    ]
}
```

Then you only need to hit ++F5++ and the fun begins :nerd:

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

With those settings your files already get formated when you save them, which
will help you to not run into trouble with the pre-commit hooks, which will
prevent you from commiting if the formaters are failing

```json title=".vscode/settings.json"
{
    "[json]": {
        "editor.defaultFormatter": "esbenp.prettier-vscode",
        "editor.quickSuggestions": {
            "strings": true
        },
        "editor.suggest.insertMode": "replace",
        "files.insertFinalNewline": true,
        "gitlens.codeLens.scopes": ["document"]
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
        "editor.defaultFormatter": "redhat.vscode-yaml",
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
        }
    },
    "editor.rulers": [88],
    "editor.defaultFormatter": "esbenp.prettier-vscode",
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
        "https://json.schemastore.org/prettierrc.json": "${workspaceFolder}/.prettierrc"
    }
}
```
