#!/usr/bin/env bash

python visualize_attmap.py \
    --root /root/media/data1/sdm/attenmaps_person \
    --save_dir ./atten_bear_visualization/ \
    --slevel 0 \
    --elevel 2 \
    --stime 0 \
    --etime 49 \
    --res 1 \
    --token_idx 3 \
    --img_path ./outputs/txt2img-samples/grid_0001.png
