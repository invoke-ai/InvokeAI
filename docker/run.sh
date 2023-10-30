#!/usr/bin/env bash
set -e

run() {
  local scriptdir=$(dirname "${BASH_SOURCE[0]}")
  cd "$scriptdir" || exit 1

  local build_args=""
  local service_name="invokeai-cpu"

  [[ -f ".env" ]] &&
    build_args=$(awk '$1 ~ /=[^$]/ {print "--build-arg " $0 " "}' .env) &&
    service_name="invokeai-$(awk -F '=' '/GPU_DRIVER/ {print $2}' .env)"

  printf "%s\n" "docker compose build args:"
  printf "%s\n" "$build_args"

  docker compose build $build_args
  unset build_args

  printf "%s\n" "starting service $service_name"
  docker compose up -d $service_name
  docker compose logs -f
}

run
