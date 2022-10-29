#!/usr/bin/env bash
set -e
# IMPORTANT: You need to have a token on huggingface.co to be able to download the checkpoint!!!
# configure values by using env when executing build.sh
# f.e. env ARCH=aarch64 GITHUB_INVOKE_AI=https://github.com/yourname/yourfork.git ./build.sh

source ./docker-build/env.sh || echo "please run from repository root" || exit 1

invokeai_conda_version=${INVOKEAI_CONDA_VERSION:-py39_4.12.0-${platform/\//-}}
invokeai_conda_prefix=${INVOKEAI_CONDA_PREFIX:-\/opt\/conda}
invokeai_conda_env_file=${INVOKEAI_CONDA_ENV_FILE:-environment.yml}
invokeai_git=${INVOKEAI_GIT:-https://github.com/invoke-ai/InvokeAI.git}
huggingface_token=${HUGGINGFACE_TOKEN?}

# print the settings
echo "You are using these values:"
echo -e "project_name:\t\t ${project_name}"
echo -e "volumename:\t\t ${volumename}"
echo -e "arch:\t\t\t ${arch}"
echo -e "platform:\t\t ${platform}"
echo -e "invokeai_conda_version:\t ${invokeai_conda_version}"
echo -e "invokeai_conda_prefix:\t ${invokeai_conda_prefix}"
echo -e "invokeai_conda_env_file: ${invokeai_conda_env_file}"
echo -e "invokeai_git:\t\t ${invokeai_git}"
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

_copyCheckpoints() {
  echo "creating subfolders for models and outputs"
  _runAlpine mkdir models
  _runAlpine mkdir outputs
  echo -n "downloading sd-v1-4.ckpt"
  _runAlpine wget --header="Authorization: Bearer ${huggingface_token}" -O models/sd-v1-4.ckpt https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
  echo "done"
  echo "downloading GFPGANv1.4.pth"
  _runAlpine wget -O models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
}

_checkVolumeContent() {
  _runAlpine ls -lhA /data/models
}

_getModelMd5s() {
  _runAlpine  \
    alpine sh -c "md5sum /data/models/*"
}

if [[ -n "$(docker volume ls -f name="${volumename}" -q)" ]]; then
  echo "Volume already exists"
  if [[ -z "$(_checkVolumeContent)" ]]; then
    echo "looks empty, copying checkpoint"
    _copyCheckpoints
  fi
  echo "Models in ${volumename}:"
  _checkVolumeContent
else
  echo -n "createing docker volume "
  docker volume create "${volumename}"
  _copyCheckpoints
fi

# Build Container
docker build \
  --platform="${platform}" \
  --tag "${invokeai_tag}" \
  --build-arg project_name="${project_name}" \
  --build-arg conda_version="${invokeai_conda_version}" \
  --build-arg conda_prefix="${invokeai_conda_prefix}" \
  --build-arg conda_env_file="${invokeai_conda_env_file}" \
  --build-arg invokeai_git="${invokeai_git}" \
  --file ./docker-build/Dockerfile \
  .
