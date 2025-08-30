python eval_pipeline.py \
--input-data "/home/TP_tracking/Development/nexcloud_files/TOF_camera/TOF_camera" \
--ref-images "/home/TP_tracking/Documents/SpaTrack/eval_tof_utils/reference_images/" \
--sam-checkpoint "/home/TP_tracking/Documents/segment_anything/sam_vit_h_4b8939.pth" \
--sam-model-type vit_h \
--device cuda \
--base-dir "evaluation"