#!/usr/bin/env bash
set -e

# This script is provided for backwards compatibility with the old docker setup.
# it doesn't do much aside from wrapping the usual docker compose CLI.

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
cd "$SCRIPTDIR" || exit 1

docker compose up -d
docker compose logs -f
