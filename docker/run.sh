#!/usr/bin/env bash
set -e -o pipefail

run() {
  local scriptdir=$(dirname "${BASH_SOURCE[0]}")
  cd "$scriptdir" || exit 1

  local build_args=""
  local profile=""

  # create .env file if it doesn't exist, otherwise docker compose will fail
  touch .env

  # parse .env file for build args
  build_args=$(awk '$1 ~ /=[^$]/ && $0 !~ /^#/ {print "--build-arg " $0 " "}' .env) &&
  profile="$(awk -F '=' '/GPU_DRIVER/ {print $2}' .env)"

  # default to 'cuda' profile
  [[ -z "$profile" ]] && profile="cuda"

  local service_name="invokeai-$profile"

  if [[ ! -z "$build_args" ]]; then
    printf "%s\n" "docker compose build args:"
    printf "%s\n" "$build_args"
  fi

  docker compose build $build_args $service_name
  unset build_args

  printf "%s\n" "starting service $service_name"
  docker compose --profile "$profile" up -d "$service_name"
  docker compose logs -f
}

run
