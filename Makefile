# simple Makefile with scripts that are otherwise hard to remember
# to use, run from the repo root `make <command>`

# Runs ruff, fixing any safely-fixable errors and formatting
ruff:
		ruff check . --fix
		ruff format .

# Runs ruff, fixing all errors it can fix and formatting
ruff-unsafe:
		ruff check . --fix --unsafe-fixes
		ruff format .

# Runs mypy, using the config in pyproject.toml
mypy:
		mypy scripts/invokeai-web.py

# Runs mypy, ignoring the config in pyproject.toml but still ignoring missing (untyped) imports
# (many files are ignored by the config, so this is useful for checking all files)
mypy-all:
		mypy scripts/invokeai-web.py --config-file= --ignore-missing-imports