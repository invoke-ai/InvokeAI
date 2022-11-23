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
	docker build -t local/invokeai:latest -f Dockerfile.cloud ..

# Copy only the content of config dir first, such that the configuration
# script can run with the expected config dir already populated.
# Then, run the configuration script.
configure:
	docker run --rm -it \
		-v ${INVOKEAI_CACHEDIR}/configs:/mnt/configs \
		--entrypoint bash ${IMAGE} \
		-c "cp -r ./configs/* /mnt/configs/"
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${INVOKEAI_CACHEDIR}:${INVOKEAI_ROOT} \
		-v ${INVOKEAI_CACHEDIR}/.cache:/root/.cache \
		${IMAGE} \
		-c "scripts/configure_invokeai.py --root ${INVOKEAI_ROOT}"
	sudo chown -R ${USER}:${GROUP} ${INVOKEAI_CACHEDIR}

# Run the container with the cache mounted and the web server exposed on port 9090
web:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${INVOKEAI_CACHEDIR}:${INVOKEAI_ROOT} \
		-v ${INVOKEAI_CACHEDIR}/.cache:/root/.cache \
		--entrypoint bash -p9090:9090 ${IMAGE} \
		-c "scripts/invoke.py --web --host 0.0.0.0 --root ${INVOKEAI_ROOT}"

cli:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${INVOKEAI_CACHEDIR}:${INVOKEAI_ROOT} \
		-v ${INVOKEAI_CACHEDIR}/.cache:/root/.cache \
		--entrypoint bash ${IMAGE} \
		-c "scripts/invoke.py --root ${INVOKEAI_ROOT}"

# Run the container with the cache mounted and open a bash shell instead of the Invoke CLI or webserver
shell:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${INVOKEAI_CACHEDIR}:${INVOKEAI_ROOT} \
		-v ${INVOKEAI_CACHEDIR}/.cache:/root/.cache \
		-e INVOKEAI_ROOT=${INVOKEAI_ROOT} \
		--entrypoint bash ${IMAGE} --

.PHONY: build configure web cli shell