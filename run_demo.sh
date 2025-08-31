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
--ref-images "/home/TP_tracking/Documents/SpaTrack/eval_tof_utils/reference_images/" \
--sam-checkpoint "/home/TP_tracking/Documents/segment_anything/sam_vit_h_4b8939.pth" \
--generate-mask \
