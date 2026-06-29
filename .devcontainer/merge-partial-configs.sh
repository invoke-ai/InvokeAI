#!/bin/sh
for d in cpu podman-cuda; do
    pnpx @gradientedge/merge-jsonc@1.1.0 \
        --indent 4 --array-merge concat \
        --out "${d}"/devcontainer.json base/partial.jsonc "${d}"/partial.jsonc
done
