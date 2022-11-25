#!/usr/bin/env bash
set -e
# IMPORTANT: You need to have a token on huggingface.co to be able to download the checkpoint!!!
# configure values by using env when executing build.sh
# f.e. env ARCH=aarch64 GITHUB_INVOKE_AI=https://github.com/yourname/yourfork.git ./build.sh

source ./docker-build/env.sh || echo "please run from repository root" || exit 1

pip_requirements=${PIP_REQUIREMENTS:-requirements-lin-cuda.txt}

# print the settings
echo "You are using these values:"
echo -e "project_name:\t\t ${project_name}"
echo -e "volumename:\t\t ${volumename}"
echo -e "arch:\t\t\t ${arch}"
echo -e "platform:\t\t ${platform}"
echo -e "invokeai_tag:\t\t ${invokeai_tag}\n"

_runAlpine() {
  docker run \
    --rm \
    --interactive \
    --tty \
    --mount source="$volumename",target=/data \
    --workdir /data \
    alpine "$@"
}

_checkVolumeContent() {
  _runAlpine ls -lhA /data/models
}

if [[ -n "$(docker volume ls -f name="${volumename}" -q)" ]]; then
  echo "Volume already exists"
else
  echo -n "createing docker volume "
  docker volume create "${volumename}"
fi

# Build Container
docker build \
  --platform="${platform}" \
  --tag "${invokeai_tag}" \
  --build-arg project_name="${project_name}" \
  --build-arg pip_requirements="${pip_requirements}" \
  --file ./docker-build/Dockerfile \
  .

docker run \
  --rm \
  --platform "$platform" \
  --name "$project_name" \
  --hostname "$project_name" \
  --mount source="$volumename",target=/data \
  --mount type=bind,source="$HOME/.huggingface",target=/root/.huggingface \
  --env HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} \
  "${invokeai_tag}" \
  scripts/configure_invokeai.py --yes
