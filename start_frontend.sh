#!/bin/sh

cd /home/lstein/Projects/invokeai-v7-webv2-fields/invokeai/frontend/webv2
INVOKEAI_DEV_HOSTS=gorgon INVOKEAI_DEV_BACKEND=http://127.0.0.1:9090 pnpm run dev
