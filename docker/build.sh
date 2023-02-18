#!/usr/bin/env bash
set -e

# If you want to build a specific flavor, set the CONTAINER_FLAVOR environment variable
#   e.g. CONTAINER_FLAVOR=cpu ./build.sh
#   Possible Values are:
#     - cpu
#     - cuda
#     - rocm
#   Don't forget to also set it when executing run.sh
#   if it is not set, the script will try to detect the flavor by itself.
#
# Doc can be found here:
#   https://invoke-ai.github.io/InvokeAI/installation/040_INSTALL_DOCKER/

SCRIPTDIR=$(dirname "${BASH_SOURCE[0]}")
cd "$SCRIPTDIR" || exit 1

source ./env.sh

DOCKERFILE=${INVOKE_DOCKERFILE:-./Dockerfile}

# print the settings
echo -e "You are using these values:\n"
echo -e "Container engine:\t${CONTAINER_ENGINE}"
echo -e "Dockerfile:\t\t${DOCKERFILE}"
echo -e "index-url:\t\t${PIP_EXTRA_INDEX_URL:-none}"
echo -e "Volumename:\t\t${VOLUMENAME}"
echo -e "Platform:\t\t${PLATFORM}"
echo -e "Container Registry:\t${CONTAINER_REGISTRY}"
echo -e "Container Repository:\t${CONTAINER_REPOSITORY}"
echo -e "Container Tag:\t\t${CONTAINER_TAG}"
echo -e "Container Flavor:\t${CONTAINER_FLAVOR}"
echo -e "Container Image:\t${CONTAINER_IMAGE}\n"

# Create outputs directory if it does not exist
[[ -d ./outputs ]] || mkdir ./outputs

# Create docker volume
if [[ -n "$(${CONTAINER_ENGINE} volume ls -f name="${VOLUMENAME}" -q)" ]]; then # Note: newer versions of podman: podman volume exists ${VOLUMENAME} 
    echo -e "Volume already exists\n"
else
    echo -n "creating ${CONTAINER_ENGINE} volume "
    "${CONTAINER_ENGINE}" volume create "${VOLUMENAME}"
fi

# Build Container
DOCKER_BUILDKIT=1 "${CONTAINER_ENGINE}" build \
    --platform="${PLATFORM:-linux/amd64}" \
    --tag="${CONTAINER_IMAGE:-invokeai}" \
    ${CONTAINER_FLAVOR:+--build-arg="CONTAINER_FLAVOR=${CONTAINER_FLAVOR}"} \
    ${PIP_EXTRA_INDEX_URL:+--build-arg="PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}"} \
    ${PIP_PACKAGE:+--build-arg="PIP_PACKAGE=${PIP_PACKAGE}"} \
    --file="${DOCKERFILE}" \
    ..

# Podman only:  set ownership for user 1000:1000 (appuser) the right way
if [[ ${CONTAINER_ENGINE} == "podman" ]] ; then
   echo Setting ownership for container\'s appuser on /data and /data/outputs
   podman run \
      --mount type=volume,src="${VOLUMENAME}",target=/data \
      --user root --entrypoint "/bin/chown" "${CONTAINER_IMAGE:-invokeai}" \
      -R 1000:1000 /data
   podman run \
      --mount type=bind,source="$(pwd)"/outputs,target=/data/outputs \
      --user root --entrypoint "/bin/chown" "${CONTAINER_IMAGE:-invokeai}" \
      -R 1000:1000 /data/outputs
fi
