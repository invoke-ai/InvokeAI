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
	@echo "frontend-install         Install the pnpm modules needed for the frontend"
	@echo "frontend-build           Build the frontend for localhost:9090"
	@echo "frontend-test            Run the frontend test suite once"
	@echo "frontend-dev             Run the frontend in developer mode on localhost:5173"
	@echo "frontend-openapi         Generate the OpenAPI schema"
	@echo "frontend-typegen         Generate types for the frontend from the OpenAPI schema"
	@echo "frontend-lint            Run frontend checks and fixable lint/format steps"
	@echo "wheel                    Build the wheel for the current version"
	@echo "tag-release              Tag the GitHub repository with the current version (use at release time only!)"
	@echo "openapi                  Generate the OpenAPI schema for the app, outputting to stdout"
	@echo "docs-install             Install the pnpm modules needed for the docs site"
	@echo "docs-dev                 Serve the astro starlight docs site with live reload"
	@echo "docs-build               Build the docs site for production"
	@echo "docs-preview             Preview the docs site locally"

# Runs ruff, fixing any safely-fixable errors and formatting
ruff:
	cd invokeai && uv tool run ruff@0.11.2 format

# Runs ruff, fixing all errors it can fix and formatting
ruff-unsafe:
	ruff check . --fix --unsafe-fixes
	ruff format

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

# Install the pnpm modules needed for the front end
frontend-install:
	rm -rf invokeai/frontend/web/node_modules
	cd invokeai/frontend/web && pnpm install

# Build the frontend
frontend-build:
	cd invokeai/frontend/web && pnpm build

# Run the frontend test suite once
frontend-test:
	cd invokeai/frontend/web && pnpm run test:run

# Run the frontend in dev mode
frontend-dev:
	cd invokeai/frontend/web && pnpm dev

# Generate the OpenAPI Schema for the app
frontend-openapi:
	cd invokeai/frontend/web && \
	python ../../../scripts/generate_openapi_schema.py > openapi.json && \
	pnpm prettier --write openapi.json

frontend-typegen:
	cd invokeai/frontend/web && python ../../../scripts/generate_openapi_schema.py | pnpm typegen

frontend-lint:
	cd invokeai/frontend/web/src && \
	pnpm lint:tsc && \
	pnpm lint:dpdm && \
	pnpm lint:eslint --fix && \
	pnpm lint:prettier --write

# Tag the release
wheel:
	cd scripts && ./build_wheel.sh

# Tag the release
tag-release:
	cd scripts && ./tag_release.sh

# Generate the OpenAPI Schema for the app
openapi:
	python scripts/generate_openapi_schema.py

# Install the pnpm modules needed for the docs site
docs-install:
	cd docs && pnpm install

# Serve the astro starlight docs site w/ live reload
docs-dev:
	cd docs && pnpm run dev

docs-build:
	cd docs && DEPLOY_TARGET='custom' pnpm run build

docs-preview:
	cd docs && pnpm run preview
