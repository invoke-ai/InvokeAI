#!/bin/sh
set -ex
for d in "$@" ; do
    # Redirect input from /dev/null to prevent it from hanging on prompts.
    # It should stop node from thinking stdin.isTTY and prompting in the first place, but it doesn't?
    # The exit code is 0 even when this nulls out a prompt. :(
    pnpm -C "${d}" install --frozen-lockfile --config.confirmModulesPurge=false < /dev/null
done
