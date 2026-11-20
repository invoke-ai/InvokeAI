#!/usr/bin/env bash
set -e -o pipefail

check_rocm_data_mount() {
  local containers container mounted
  # Older ROCm services could write to a host path interpreted inside the container.
  local check_mount='
    {{- range .Mounts}}
      {{- if or (eq .Type "bind") (eq .Type "volume")}}
        {{- $setting := printf "INVOKEAI_ROOT=%s" .Destination}}
        {{- range $.Config.Env}}{{if eq . $setting}}mounted{{end}}{{end}}
      {{- end}}
    {{- end}}'
  containers=$(docker compose --profile rocm ps --all --quiet invokeai-rocm)
  for container in $containers; do
    mounted=$(docker inspect --format "$check_mount" "$container")
    if [[ "$mounted" != "mounted" ]]; then
      printf 'Container %s has no verified data mount at its INVOKEAI_ROOT. Recreating it could delete your data.\n' "$container" >&2
      printf '%s\n' 'Recover the data first: https://v7.invoke.ai/configuration/docker/#recover-data-before-replacing-a-container' >&2
      return 1
    fi
  done
}

run() {
  local scriptdir=$(dirname "${BASH_SOURCE[0]}")
  cd "$scriptdir" || exit 1

  local build_args=""
  local profile=""

  # create .env file if it doesn't exist, otherwise docker compose will fail
  touch .env

  # parse .env file for build args
  build_args=$(awk '$1 ~ /=[^$]/ && $0 !~ /^#/ {print "--build-arg " $0 " "}' .env) &&
  profile="$(awk -F '=' '$1 == "GPU_DRIVER" {profile=$2} END {print profile}' .env)"

  # default to 'cuda' profile
  [[ -z "$profile" ]] && profile="cuda"

  local service_name="invokeai-$profile"

  if [[ "$profile" == "rocm" ]]; then
    check_rocm_data_mount
  fi

  if [[ ! -z "$build_args" ]]; then
    printf "%s\n" "docker compose build args:"
    printf "%s\n" "$build_args"
  fi

  docker compose build $build_args $service_name
  unset build_args

  printf "%s\n" "starting service $service_name"
  docker compose --profile "$profile" up -d "$service_name"
  docker compose --profile "$profile" logs -f
}

run
