# Directory in the container where the INVOKEAI_ROOT (runtime dir) will be mounted
INVOKEAI_ROOT=/mnt/invokeai
# Host directory to contain the runtime dir. Will be mounted at INVOKEAI_ROOT path in the container
HOST_MOUNT_PATH=${HOME}/invokeai

DOCKER_BUILDKIT=1
IMAGE=local/invokeai:latest

USER=$(shell id -u)
GROUP=$(shell id -g)

# All downloaded models, config, etc will end up in ${HOST_MOUNT_PATH} on the host.
# This is consistent with the expected non-Docker behaviour.
# Contents can be moved to a persistent storage and used to prime the cache on another host.

build:
	docker build -t local/invokeai:latest -f Dockerfile.cloud ..

configure:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${HOST_MOUNT_PATH}:${INVOKEAI_ROOT} \
		-v ${HOST_MOUNT_PATH}/.cache:/root/.cache \
		-e INVOKEAI_ROOT=${INVOKEAI_ROOT} \
		--entrypoint bash \
		${IMAGE} -c "scripts/configure_invokeai.py"
	sudo chown -R ${USER}:${GROUP} ${HOST_MOUNT_PATH}

# Run the container with the runtime dir mounted and the web server exposed on port 9090
web:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${HOST_MOUNT_PATH}:${INVOKEAI_ROOT} \
		-v ${HOST_MOUNT_PATH}/.cache:/root/.cache \
		-e INVOKEAI_ROOT=${INVOKEAI_ROOT} \
		--entrypoint bash -p9090:9090 ${IMAGE} \
		-c "scripts/invoke.py --web --host 0.0.0.0 --root ${INVOKEAI_ROOT}"

# Run the cli with the runtime dir mounted
cli:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${HOST_MOUNT_PATH}:${INVOKEAI_ROOT} \
		-v ${HOST_MOUNT_PATH}/.cache:/root/.cache \
		-e INVOKEAI_ROOT=${INVOKEAI_ROOT} \
		--entrypoint bash ${IMAGE} \
		-c "scripts/invoke.py --root ${INVOKEAI_ROOT}"

# Run the container with the cache mounted and open a bash shell instead of the Invoke CLI or webserver
shell:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${HOST_MOUNT_PATH}:${INVOKEAI_ROOT} \
		-v ${HOST_MOUNT_PATH}/.cache:/root/.cache \
		-e INVOKEAI_ROOT=${INVOKEAI_ROOT} \
		--entrypoint bash ${IMAGE} --

.PHONY: build configure web cli shell
