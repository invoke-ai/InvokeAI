#!/usr/bin/env bash

project_name=${PROJECT_NAME:-invokeai}
volumename=${VOLUMENAME:-${project_name}_data}
arch=${ARCH:-x86_64}
platform=${PLATFORM:-Linux/${arch}}
invokeai_tag=${INVOKEAI_TAG:-${project_name}:${arch}}
gpus=${GPU_FLAGS:+--gpus=${GPU_FLAGS}}
cmd_override=${CMD_OVERRIDE:-'--web --host=0.0.0.0'}	       

export project_name
export volumename
export arch
export platform
export invokeai_tag
export gpus
export cmd_override
