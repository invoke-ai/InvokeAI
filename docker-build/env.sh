#!/usr/bin/env bash

project_name=${PROJECT_NAME:-invokeai}
volumename=${VOLUMENAME:-${project_name}_data}
arch=${ARCH:-x86_64}
platform=${PLATFORM:-Linux/${arch}}
invokeai_tag=${INVOKEAI_TAG:-${project_name}:${arch}}

export project_name
export volumename
export arch
export platform
export invokeai_tag
