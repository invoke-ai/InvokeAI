#!/usr/bin/env bash
set -e -o pipefail

run() {
  local scriptdir=$(dirname "${BASH_SOURCE[0]}")
  cd "$scriptdir" || exit 1

  local build_args=""
  local profile=""

  touch .env
  while IFS='=' read -r key value; do
    if [[ ! $key =~ ^# && ! -z $value ]]; then
      build_args+=" --build-arg $key=$value"
      export "$key=$value"
    fi
  done < .env
  profile="$(awk -F '=' '/GPU_DRIVER/ {print $2}' .env)"

  [[ -z "$profile" ]] && profile="nvidia"

  local service_name="invokeai-$profile"

  if [[ ! -z "$build_args" ]]; then
    printf "%s\n" "docker compose build args:"
    printf "%s\n" "$build_args"
  fi

  docker compose build $build_args $service_name
  unset build_args

  printf "%s\n" "starting service $service_name"
  # `touch compose.override.yaml` in case user doesn't use it (i.e., doesn't have it)
  touch compose.override.yaml
  docker compose --profile "$profile" -f docker-compose.yml -f compose.override.yaml up -d "$service_name"
  docker compose logs -f
}

run
