#!/usr/bin/env bash
set -e

# IMPORTANT: You need to have a token on huggingface.co to be able to download the checkpoints!!!
# configure values by using env when executing build.sh f.e. `env ARCH=aarch64 ./build.sh`

source ./docker-build/env.sh \
  || echo "please execute docker-build/build.sh from repository root" \
  || exit 1

# check if HUGGINGFACE_TOKEN is available
huggingface_token=${HUGGINGFACE_TOKEN:?Please set your token for Huggingface as HUGGINGFACE_TOKEN}

pip_requirements=${PIP_REQUIREMENTS:-requirements-lin-cuda.txt}
dockerfile=${INVOKE_DOCKERFILE:-docker-build/Dockerfile}

# print the settings
echo -e "You are using these values:\n"
echo -e "Dockerfile:\t ${dockerfile}"
echo -e "requirements:\t ${pip_requirements}"
echo -e "volumename:\t ${volumename}"
echo -e "arch:\t\t ${arch}"
echo -e "platform:\t ${platform}"
echo -e "invokeai_tag:\t ${invokeai_tag}\n"

if [[ -n "$(docker volume ls -f name="${volumename}" -q)" ]]; then
  echo "Volume already exists"
  echo
else
  echo -n "createing docker volume "
  docker volume create "${volumename}"
fi

# Build Container
docker build \
  --platform="${platform}" \
  --tag="${invokeai_tag}" \
  --build-arg="PIP_REQUIREMENTS=${pip_requirements}" \
  --file="${dockerfile}" \
  .

# download required models non-interactive
docker run \
  --rm \
  --platform="$platform" \
  --name="$project_name" \
  --hostname="$project_name" \
  --mount="source=$volumename,target=/data" \
  --env="HUGGINGFACE_TOKEN=${huggingface_token}" \
  --entrypoint="python3" \
  "${invokeai_tag}" \
  scripts/configure_invokeai.py --yes
