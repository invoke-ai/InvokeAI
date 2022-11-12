# Copyright (c) 2022 Eugene Brodsky (https://github.com/ebr)

# InvokeAI root directory in the container
INVOKEAI_ROOT=/invokeai
# Destination directory for cache on the host
INVOKEAI_CACHEDIR=${HOME}/invokeai

DOCKER_BUILDKIT=1
IMAGE=local/invokeai:latest

# Downloads will end up in ${INVOKEAI_CACHEDIR}.
# Contents can be moved to a persistent storage and used to rehydrate the cache,
# to be mounted into the container.

# TBD: cache dir may be modified at runtime due to checksum calculations
# for custom models. Ideally the checksums would be precomputed and stored in the cache.
# Also CLI requires RW on first run when populating torch cache.

build:
	docker buildx build -t local/invokeai:latest -f Dockerfile.cloud ..

# Populate the cache. Config is copied first, so it's there for the model preload step.
load-models: _copy-config
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${INVOKEAI_CACHEDIR}:${INVOKEAI_ROOT} \
		-v ${INVOKEAI_CACHEDIR}/root-cache:/root/.cache \
		${IMAGE} \
		-c "micromamba -r /opt/conda -n invokeai run python scripts/load_models.py"

run:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${INVOKEAI_CACHEDIR}:${INVOKEAI_ROOT} \
		-v ${INVOKEAI_CACHEDIR}/root-cache:/root/.cache \
		--entrypoint bash -p9090:9090 \
		${IMAGE} \
		-c "micromamba -r /opt/conda -n invokeai run python scripts/invoke.py --web --host 0.0.0.0"

shell:
	docker run --rm -it --runtime=nvidia --gpus=all \
		-v ${INVOKEAI_CACHEDIR}:${INVOKEAI_ROOT} \
		-v ${INVOKEAI_CACHEDIR}/root-cache:/root/.cache \
		--entrypoint bash \
		${IMAGE} --

# This is an intermediate task that copies the contents of the config dir into the cache.
# This prepares the cached config dir, so that when we run the model preload with the config mounted,
# it is already populated.
_copy-config:
	docker run --rm -it \
		-v ${INVOKEAI_CACHEDIR}/config:${INVOKEAI_ROOT}/tmp/config \
		--workdir ${INVOKEAI_ROOT} \
		--entrypoint bash ${IMAGE} \
		-c "cp -r ./tmp/config/* ./config/"


.PHONY: build preload run shell _copy-config