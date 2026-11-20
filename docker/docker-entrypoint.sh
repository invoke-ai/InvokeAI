#!/bin/bash
set -e -o pipefail

### Container entrypoint
# Runs the CMD as defined by the Dockerfile or passed to `docker run`
# Can be used to configure the runtime dir
# Bypass by using ENTRYPOINT or `--entrypoint`

### Set INVOKEAI_ROOT pointing to a valid runtime directory
# Otherwise configure the runtime dir first.

### Set the CONTAINER_UID envvar to match your user.
# Ensures files created in the container are owned by you:
#   docker run --rm -it -v /some/path:/invokeai -e CONTAINER_UID=$(id -u) <this image>
# Default UID: 1000 chosen due to popularity on Linux systems. Possibly 501 on MacOS.

USER_ID=${CONTAINER_UID:-1000}
USER=ubuntu
# if the user does not exist, create it. It is expected to be present on ubuntu >=24.x
_=$(id ${USER} 2>&1) || useradd -u ${USER_ID} ${USER}
# ensure the UID is correct
usermod -u ${USER_ID} ${USER} 1>/dev/null

## ROCM specific configuration
# render group within the container must match the host render group
# otherwise the container will not be able to access the host GPU.
if [[ -v "RENDER_GROUP_ID" ]] && [[ ! -z "${RENDER_GROUP_ID}" ]]; then
  # ensure the render group exists
  groupmod -g ${RENDER_GROUP_ID} render
  usermod -a -G render ${USER}
  usermod -a -G video ${USER}
fi


### Set the $PUBLIC_KEY env var to enable SSH access.
# We do not install openssh-server in the image by default to avoid bloat.
# but it is useful to have the full SSH server e.g. on Runpod.
# (use SCP to copy files to/from the image, etc)
if [[ "${PUBLIC_KEY:-}" =~ [^[:space:]] ]]; then
  authorized_keys="${HOME}/.ssh/authorized_keys"
  if [[ -s "$authorized_keys" ]]; then
    while IFS= read -r key; do
      if [[ "$key" =~ [^[:space:]] ]] && ! grep -Fxq -- "$key" "$authorized_keys"; then
        printf 'PUBLIC_KEY does not match %s. Set it to an existing key line or update the file from a trusted session before restarting.\n' "$authorized_keys" >&2
        exit 1
      fi
    done <<< "$PUBLIC_KEY"
  else
    (
      umask 077
      mkdir -p "${HOME}/.ssh"
      printf '%s\n' "$PUBLIC_KEY" >"$authorized_keys"
    )
  fi
  if ! command -v sshd >/dev/null; then
    apt-get update
    apt-get install -y openssh-server
  fi
  service ssh start
fi

mkdir -p "${INVOKEAI_ROOT}"
chown --recursive ${USER} "${INVOKEAI_ROOT}" || true
cd "${INVOKEAI_ROOT}"
export HF_HOME=${HF_HOME:-$INVOKEAI_ROOT/.cache/huggingface}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$INVOKEAI_ROOT/.matplotlib}

# Run the CMD as the Container User (not root).
exec gosu ${USER} "$@"
