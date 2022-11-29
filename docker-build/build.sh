#!/usr/bin/env bash
set -e

# IMPORTANT: You need to have a token on huggingface.co to be able to download the checkpoints!!!
# configure values by using env when executing build.sh f.e. `env ARCH=aarch64 ./build.sh`

source ./docker-build/env.sh \
  || echo "please execute docker-build/build.sh from repository root" \
  || exit 1

pip_requirements=${PIP_REQUIREMENTS:-requirements-lin-cuda.txt}
dockerfile=${INVOKE_DOCKERFILE:-docker-build/Dockerfile}

# print the settings
echo "You are using these values:"
echo -e "Dockerfile:\t\t ${dockerfile}"
echo -e "requirements:\t\t ${pip_requirements}"
echo -e "volumename:\t\t ${volumename}"
echo -e "arch:\t\t\t ${arch}"
echo -e "platform:\t\t ${platform}"
echo -e "invokeai_tag:\t\t ${invokeai_tag}\n"

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

docker run \
  --rm \
  --platform="$platform" \
  --name="$project_name" \
  --hostname="$project_name" \
  --mount="source=$volumename,target=/data" \
  --mount="type=bind,source=$HOME/.huggingface,target=/root/.huggingface" \
  --env="HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}" \
  --entrypoint="python3" \
  "${invokeai_tag}" \
  scripts/configure_invokeai.py --yes
