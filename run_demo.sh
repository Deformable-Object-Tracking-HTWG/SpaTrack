#!/bin/bash
python chunked_demo.py \
--vid_name /home/TP_tracking/Documents/SpaTrack/assets/download/Gymnastik_1_5s/rgbd_data/Gymnastik_1_5s \
--model spatracker \
--chunk_size 40 \
--downsample 1.0 \
--grid_size 20 \
--outdir ../../subs_prep/output \
--fps_vis 30 \
--rgbd \
