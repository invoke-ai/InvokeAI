#!/bin/sh
set -e
export UV_NO_PROGRESS=1

set -x
if [ ! -e .venv/bin/python ] ; then
    if [ ! -d .venv ] ; then
        uv venv < /dev/null
    else
        # There is a venv present; maybe it's just missing the uv-managed python.
        uv python install < /dev/null
        # If that didn't work, clear the venv and rebuild it.
	    .venv/bin/python --version || uv venv --clear < /dev/null
    fi
fi

