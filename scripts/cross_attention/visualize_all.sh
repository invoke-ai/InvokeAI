#!/usr/bin/env bash

attenmap="/root/media/data1/sdm/attenmaps_apples_swap_orig"
sample_name="A_basket_full_of_apples_tar"
token_idx=5

for a in 1,1,0  0,0,1 0,0,2 0,0,3  2,2,1 2,2,2 2,2,3  0,2,1 0,2,2 0,2,3
do
        IFS=',' read item1 item2 item3 <<< "${a}"

        python visualize_attmap.py \
            --root ${attenmap} \
            --save_dir ./atten_${sample_name}_${token_idx}/ \
            --slevel ${item1} \
            --elevel ${item2} \
            --stime 0 \
            --etime 49 \
            --res ${item3} \
            --token_idx ${token_idx} \
            --img_path ./outputs/swap-samples/${sample_name}.png
done

python visualize_comp.py \
    --root ./atten_${sample_name}_${token_idx} \
    --token_idx ${token_idx}
