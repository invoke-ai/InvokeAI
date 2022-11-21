# Copyright (c) 2022 Eugene Brodsky (https://github.com/ebr)

# Directory in the container where the INVOKEAI_ROOT will be mounted
INVOKEAI_ROOT=/mnt/invokeai
# Host directory to contain the model cache. Will be mounted at INVOKEAI_ROOT path in the container
INVOKEAI_CACHEDIR=${HOME}/invokeai

DOCKER_BUILDKIT=1
IMAGE=local/invokeai:latest

USER=$(shell id -u)
GROUP=$(shell id -g)

# All downloaded models and config will end up in ${INVOKEAI_CACHEDIR}.
# Contents can be moved to a persistent storage and used to rehydrate the cache on another host

build:
	docker buildx build -t local/invokeai:latest -f Dockerfile.cloud ..

# Populate the cache.
# First, pre-seed the config dir on the host with the content from the image,
# such that the model preload step can run with the config mounted and pre-populated.
# Then, run `load-models` to cache models, VAE, other static data.
load-models:
	docker run --rm -it \
		-v ${INVOKEAI_CACHEDIR}/configs:/mnt/configs \
		--entrypoint bash ${IMAGE} \
		-c "cp -r ./configs/* /mnt/configs/"
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${INVOKEAI_CACHEDIR}:${INVOKEAI_ROOT} \
		-v ${INVOKEAI_CACHEDIR}/.cache:/root/.cache \
		--entrypoint bash ${IMAGE} \
		-c "micromamba -n invokeai run python scripts/load_models.py --root ${INVOKEAI_ROOT}"
	sudo chown -R ${USER}:${GROUP} ${INVOKEAI_CACHEDIR}

# Run the container with the cache mounted and the web server exposed on port 9090
run:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${INVOKEAI_CACHEDIR}:${INVOKEAI_ROOT} \
		-v ${INVOKEAI_CACHEDIR}/.cache:/root/.cache \
		--entrypoint bash -p9090:9090 ${IMAGE} \
		-c "micromamba -n invokeai run python scripts/invoke.py --web --host 0.0.0.0 --root ${INVOKEAI_ROOT}"

# Run the container with the cache mounted and open a bash shell instead of the Invoke CLI or webserver
shell:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${INVOKEAI_CACHEDIR}:${INVOKEAI_ROOT} \
		-v ${INVOKEAI_CACHEDIR}/.cache:/root/.cache \
		-e INVOKEAI_ROOT=${INVOKEAI_ROOT} \
		--entrypoint bash ${IMAGE} --

.PHONY: build preload run shell