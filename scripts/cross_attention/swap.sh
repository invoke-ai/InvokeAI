#!/usr/bin/env bash

python ./scripts/swap.py \
    --prompt "a cake with jelly beans decorations" \
    --n_samples 3 \
    --strength 0.99 \
    --sprompt "a cake with decorations" \
    --is-swap
