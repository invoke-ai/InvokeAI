# simple Makefile with scripts that are otherwise hard to remember
# to use, run from the repo root `make <command>`

default: help

help:
	@echo Developer commands:
	@echo
	@echo "ruff                     Run ruff, fixing any safely-fixable errors and formatting"
	@echo "ruff-unsafe              Run ruff, fixing all fixable errors and formatting"
	@echo "mypy                     Run mypy using the config in pyproject.toml to identify type mismatches and other coding errors"
	@echo "mypy-all                 Run mypy ignoring the config in pyproject.tom but still ignoring missing imports"
	@echo "test                     Run the unit tests."
	@echo "update-config-docstring  Update the app's config docstring so mkdocs can autogenerate it correctly."
	@echo "frontend-install         Install the pnpm modules needed for the front end"
	@echo "frontend-build           Build the frontend in order to run on localhost:9090"
	@echo "frontend-dev             Run the frontend in developer mode on localhost:5173"
	@echo "frontend-typegen         Generate types for the frontend from the OpenAPI schema"
	@echo "installer-zip            Build the installer .zip file for the current version"
	@echo "tag-release              Tag the GitHub repository with the current version (use at release time only!)"

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

# Run the unit tests
test:
	pytest ./tests

# Update config docstring
update-config-docstring:
	python scripts/update_config_docstring.py

# Install the pnpm modules needed for the front end
frontend-install:
	rm -rf invokeai/frontend/web/node_modules
	cd invokeai/frontend/web && pnpm install

# Build the frontend
frontend-build:
	cd invokeai/frontend/web && pnpm build

# Run the frontend in dev mode
frontend-dev:
	cd invokeai/frontend/web && pnpm dev

frontend-typegen:
	cd invokeai/frontend/web && python ../../../scripts/generate_openapi_schema.py | pnpm typegen

# Installer zip file
installer-zip:
	cd installer && ./create_installer.sh

# Tag the release
tag-release:
	cd installer && ./tag_release.sh

