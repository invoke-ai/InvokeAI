#!/usr/bin/env bash
set -e

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
cd "$SCRIPTDIR" || exit 1

docker-compose up --build -d
docker-compose logs -f
