#!/usr/bin/env bash
set -e

build_args=""

[[ -f ".env" ]] && build_args=$(awk '$1 ~ /\=[^$]/ {print "--build-arg " $0 " "}' .env)

echo "docker-compose build args:"
echo $build_args

docker-compose build $build_args
