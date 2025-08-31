import os
import sys
import argparse
import time
import numpy as np
from PIL import Image
import cv2
from easydict import EasyDict as edict
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from moviepy.editor import ImageSequenceClip
import moviepy


# Import necessary modules from the existing project
from models.spatracker.predictor import SpaTrackerPredictor
from models.spatracker.utils.visualizer import Visualizer, read_video_from_path
from models.spatracker.models.core.spatracker.spatracker import get_points_on_a_grid
from eval_tof_utils.masking import create_semantic_mask
from mde import MonoDEst

# ---------------------- Helpers ----------------------
def resolve_path(root: str, name: str) -> str:
    """
    Join to root only if 'name' is not absolute.
    Return absolute, normalized path.
    """
    p = Path(name)
    if not p.is_absolute():
        p = Path(root) / p
    return str(p.resolve())


# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description="Chunked Object Tracking with SpaTracker (segmentation mask only in the first chunk)")
parser.add_argument('--root', type=str, default='./assets', help='Path to the folder with the video and optionally depth maps')
parser.add_argument('--vid_name', type=str, default='breakdance', help='Name of the video or path (with or without .mp4)')
parser.add_argument('--mask_name', type=str, default=None, help='Name or path of the segmentation mask file (e.g., my_mask.png). If None, uses vid_name.png or a full mask.')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--model', type=str, default='spatracker', help='Model name (only spatracker is supported)')
parser.add_argument('--downsample', type=float, default=0.8, help='Processing scale factor. <1 downsamples, >1 upsamples if video is too small.')
parser.add_argument('--grid_size', type=int, default=50, help='Grid size for initial point sampling (used only in the first chunk with mask)')
parser.add_argument('--outdir', type=str, default='./vis_results', help='Output directory (a timestamped subfolder will be created inside)')
parser.add_argument('--fps', type=float, default=1.0, help='Frame sampling *step* for processing (e.g., 1.0 processes every frame, 2.0 every other frame)')
parser.add_argument('--len_track', type=int, default=10, help='Length of the track visualization trail')
parser.add_argument('--fps_vis', type=int, default=30, help='FPS for the output visualization video')
parser.add_argument('--crop', action='store_true', help='Crop the video based on a fixed factor')
parser.add_argument('--crop_factor', type=float, default=1.0, help='Crop factor if --crop is used (e.g., 1.0 for 384x512 base)')
parser.add_argument('--backward', action='store_true', help='Enable backward tracking')
parser.add_argument('--vis_support', action='store_true', help='Visualize support points (if model supports it)')
parser.add_argument('--query_frame', type=int, default=0, help='Absolute query frame index for grid initialization (used for the first chunk). Should be 0 for mask on 1st abs. frame.')
parser.add_argument('--point_size', type=int, default=3, help='Size of the visualized points')
parser.add_argument('--rgbd', action='store_true', help='Whether to use RGBD as input (expects pre-computed depth maps)')
parser.add_argument('--chunk_size', type=int, default=30, help='Number of frames per processing chunk (output segment length)')
parser.add_argument('--s_length_model', type=int, default=12, help="SpaTracker's internal processing window length (S_length)")
parser.add_argument("--generate-mask", action="store_true", help="whether to generate a segmentation mask")
parser.add_argument("--ref-images", default="", help="Directory with reference images used for CLIP similarity")
parser.add_argument("--center-priority", action="store_true", help="whether prefer segment closest to center (overrides CLIP)")
parser.add_argument("--target_class", type=str, default = None, help="text prompt used for CLIP similarity")
parser.add_argument("--sam-checkpoint", type=str, help="Path to SAM Checkpoint")
parser.add_argument('--depth_model', type=str, default='zoe', help="Which Depth Estimation Model to use (zoe, depth_anything)")


args = parser.parse_args()
# args.rgbd

# ---------------------- Setup & Preparation ----------------------
print(f"MoviePy version: {moviepy.__version__}")

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

root_dir = args.root

vid_name = args.vid_name
if not vid_name.lower().endswith(".mp4"):
    vid_name += ".mp4"
    print("Filename did not end with .mp4, so '.mp4' was added.")

vid_path = resolve_path(root_dir, vid_name)
video_stem = Path(vid_path).stem  # Used for file names

if args.mask_name:
    seg_path = resolve_path(root_dir, args.mask_name)
else:
    seg_path = resolve_path(root_dir, args.vid_name + '.png')

# --- Create timestamped run directory inside outdir ---
base_outdir = args.outdir
os.makedirs(base_outdir, exist_ok=True)
timestamp = time.strftime("%Y-%m-%d_%H%M%S")  # local time
run_dir = os.path.join(base_outdir, f"{timestamp}_{video_stem}")
os.makedirs(run_dir, exist_ok=True)
outdir = run_dir
print(f"All outputs will be saved to: {outdir}")

print("Loading video...")
print(f"Resolved video path: {vid_path} (exists: {os.path.exists(vid_path)})")
print(f"Resolved mask path:  {seg_path} (exists: {os.path.exists(seg_path)})")

# robust video load (raises with clear error if it fails)
video_np = read_video_from_path(vid_path)  # Returns uint8 HWC RGB numpy
video_full_torch = torch.from_numpy(video_np).permute(0, 3, 1, 2)[None].float()  # B T C H W, values 0-255

H_raw, W_raw = video_full_torch.shape[-2:]
mask_path = None
if args.generate_mask:
    try:
        mask_path = create_semantic_mask(
            video_path=vid_path,
            output_folder=str(outdir),
            reference_images_dir=args.ref_images,
            sam_checkpoint=args.sam_checkpoint,
            use_center_priority=args.center_priority,
            target_class=args.target_class
        )
    finally:
        seg_path = mask_path
if os.path.exists(seg_path):
    print(f"Loading segmentation mask from: {seg_path}")
    segm_mask_np_raw = np.array(Image.open(seg_path))
    if len(segm_mask_np_raw.shape) == 3:
        segm_mask_np_raw = (segm_mask_np_raw[..., :3].mean(axis=-1) > 0).astype(np.uint8)
    if segm_mask_np_raw.shape[0] != H_raw or segm_mask_np_raw.shape[1] != W_raw:
        print(f"Resizing segmentation mask from {segm_mask_np_raw.shape} to raw video dimensions {H_raw}x{W_raw}")
        segm_mask_np_raw = cv2.resize(segm_mask_np_raw, (W_raw, H_raw), interpolation=cv2.INTER_NEAREST)
else:
    print(f"Segmentation mask not found at: {seg_path}. Using a full-image mask for the first chunk if grid_size > 0.")
    segm_mask_np_raw = np.ones((H_raw, W_raw), dtype=np.uint8)

H_after_crop, W_after_crop = H_raw, W_raw
segm_mask_np_after_crop = segm_mask_np_raw
crop_transform = None

if args.crop:
    print("Applying center crop...")
    base_crop_h, base_crop_w = 384, 512
    target_crop_h = int(base_crop_h * args.crop_factor)
    target_crop_w = int(base_crop_w * args.crop_factor)

    crop_transform = transforms.CenterCrop((target_crop_h, target_crop_w))

    video_full_torch_list = [crop_transform(video_full_torch[0, t]) for t in range(video_full_torch.shape[1])]
    video_full_torch = torch.stack(video_full_torch_list, dim=0).unsqueeze(0)

    segm_mask_torch_temp = torch.from_numpy(segm_mask_np_raw[None, None]).float()
    segm_mask_torch_cropped = crop_transform(segm_mask_torch_temp)
    segm_mask_np_after_crop = segm_mask_torch_cropped[0, 0].numpy().astype(np.uint8)

    H_after_crop, W_after_crop = video_full_torch.shape[-2:]
    print(f"Video and mask cropped to: {H_after_crop}x{W_after_crop}")

processing_scale_factor = args.downsample
if H_after_crop > 0 and W_after_crop > 0:
    ref_h, ref_w = 640.0, 960.0
    if H_after_crop > W_after_crop:
        processing_scale_factor = max(args.downsample, (ref_h / H_after_crop))
    elif W_after_crop > H_after_crop:
        processing_scale_factor = max(args.downsample, (ref_w / W_after_crop))
    else:
        processing_scale_factor = max(args.downsample, (ref_h / H_after_crop))
else:
    print("Warning: Video dimensions after crop are zero. Using default processing_scale_factor from args.")

if processing_scale_factor != 1.0:
    print(f"Resizing video with factor: {processing_scale_factor:.2f} for model processing")
    if video_full_torch.shape[1] > 0:
        batch_of_frames = video_full_torch[0]
        video_full_torch_resized_frames = F.interpolate(batch_of_frames, scale_factor=processing_scale_factor, mode='bilinear', align_corners=True, recompute_scale_factor=True)
        video_full_torch_resized = video_full_torch_resized_frames.unsqueeze(0)
    else:
        video_full_torch_resized = video_full_torch
else:
    video_full_torch_resized = video_full_torch
print(f"Video shape for model processing (before frame selection): {video_full_torch_resized.shape}")

# --- Safe frame step (args.fps is actually a step) ---
step = max(1, int(round(args.fps)))
frame_indices_to_process = torch.arange(0, video_full_torch.shape[1], step).long()

video_processed_full = video_full_torch_resized[:, frame_indices_to_process]

T_full_processed = video_processed_full.shape[1]
H_processed, W_processed = video_processed_full.shape[-2:]
print(f"Processed video for model has {T_full_processed} frames after selection (step={step}), with resolution {H_processed}x{W_processed}.")

video_for_overlay_original_res = video_full_torch[:, frame_indices_to_process].cpu()  # float 0-255
print(f"Video for final overlay has {video_for_overlay_original_res.shape[1]} frames, with resolution {H_after_crop}x{W_after_crop}.")

if segm_mask_np_after_crop.shape[0] != H_processed or segm_mask_np_after_crop.shape[1] != W_processed:
    segm_mask_processed_np = cv2.resize(segm_mask_np_after_crop, (W_processed, H_processed), interpolation=cv2.INTER_NEAREST)
else:
    segm_mask_processed_np = segm_mask_np_after_crop

# ---------------------- Model & Depth Predictor Setup ----------------------
if args.model != "spatracker":
    raise ValueError("Only 'spatracker' model is supported.")

print("Initializing SpaTracker model...")
model = SpaTrackerPredictor(
    checkpoint=os.path.join('./checkpoints/spaT_final.pth'),
    interp_shape=(384, 512),
    seq_length=args.s_length_model
)
if torch.cuda.is_available():
    model = model.to(device)

depth_predictor_model_instance = None
depths_full_torch_processed = None
if args.rgbd:
    DEPTH_DIR = os.path.join(root_dir, args.vid_name + "_depth")
    if not os.path.isdir(DEPTH_DIR):
        print(f"Warning: Depth directory {DEPTH_DIR} not found. Proceeding without pre-loaded depth.")
        args.rgbd = False
    else:
        print(f"Loading depth maps from {DEPTH_DIR}...")
        depths_list_for_selected_frames = []
        all_original_depth_files = sorted([os.path.join(DEPTH_DIR, fname) for fname in os.listdir(DEPTH_DIR) if fname.endswith((".npy", ".png"))])

        for original_frame_idx in frame_indices_to_process.tolist():
            if original_frame_idx < len(all_original_depth_files):
                depth_map_path = all_original_depth_files[original_frame_idx]
                if depth_map_path.endswith(".npy"):
                    depth_orig_res = np.load(depth_map_path)
                else:
                    depth_img = Image.open(depth_map_path)
                    depth_orig_res = np.array(depth_img).astype(np.float32)
                    if len(depth_orig_res.shape) == 3:
                        depth_orig_res = depth_orig_res[..., 0]

                depth_cropped_res = depth_orig_res
                if args.crop and crop_transform is not None:
                    temp_depth_torch = torch.from_numpy(depth_orig_res[None, None, :, :]).float()
                    try:
                        temp_depth_cropped = crop_transform(temp_depth_torch)
                        depth_cropped_res = temp_depth_cropped[0, 0].numpy()
                    except RuntimeError as e:
                        print(f"Warning: Could not apply crop_transform to depth map {depth_map_path} (shape {temp_depth_torch.shape}). Error: {e}. Using uncropped depth and resizing.")

                if depth_cropped_res.shape[0] != H_processed or depth_cropped_res.shape[1] != W_processed:
                    depth_resized = cv2.resize(depth_cropped_res, (W_processed, H_processed), interpolation=cv2.INTER_NEAREST)
                else:
                    depth_resized = depth_cropped_res
                depths_list_for_selected_frames.append(depth_resized)
            else:
                print(f"Warning: Depth map for original frame index {original_frame_idx} not found. Using zeros.")
                depths_list_for_selected_frames.append(np.zeros((H_processed, W_processed), dtype=np.float32))

        if depths_list_for_selected_frames:
            depths_full_torch_processed = torch.from_numpy(np.stack(depths_list_for_selected_frames, axis=0)).float()[:, None]
            if torch.cuda.is_available():
                depths_full_torch_processed = depths_full_torch_processed.to(device)
        else:
            print(f"Warning: No depth maps loaded from {DEPTH_DIR}, even though --rgbd was set. Proceeding without pre-loaded depth.")
            args.rgbd = False

if not args.rgbd:

    if args.depth_model == 'zoe':
        print("Initializing Monocular Depth Estimator (ZoeDepth NK)...")
        monodepth = MonoDEst(edict({"mde_name": "zoedepth_nk"}))
    elif args.depth_model == 'depth_anything':
        print("Initializing Monocular Depth Estimator (Depth Anything)...")
        monodepth = MonoDEst(edict({"mde_name": "depthAny"}))
    else:
        print('ERROR: Depth Model not known')

    depth_predictor_model_instance = monodepth.model
    if torch.cuda.is_available():
        depth_predictor_model_instance = depth_predictor_model_instance.to(device)
    depth_predictor_model_instance.eval()
# ---------------------- Chunking and Processing ----------------------
S_MODEL_INTERNAL_WINDOW = args.s_length_model
overlap_frames = S_MODEL_INTERNAL_WINDOW // 2

chunk_size = args.chunk_size
num_chunks = (T_full_processed + chunk_size - 1) // chunk_size
if T_full_processed == 0:
    num_chunks = 0
print(f"Video will be processed in {num_chunks} chunk(s) of up to {chunk_size} frames each.")

all_pred_tracks_list = []
all_pred_visibility_list = []
processed_video_frames_for_moviepy = []

prev_queries_for_next_chunk = None

for chunk_idx in range(num_chunks):
    print(f"\n--- Processing Chunk {chunk_idx + 1}/{num_chunks} ---")

    actual_chunk_start_global = chunk_idx * chunk_size
    actual_chunk_end_global = min((chunk_idx + 1) * chunk_size, T_full_processed)
    num_frames_in_actual_chunk = actual_chunk_end_global - actual_chunk_start_global

    if num_frames_in_actual_chunk <= 0:
        print(f"Skipping empty or invalid chunk {chunk_idx + 1}")
        continue

    current_dynamic_overlap = overlap_frames if chunk_idx > 0 else 0
    model_feed_start_global = actual_chunk_start_global - current_dynamic_overlap
    model_feed_end_global = actual_chunk_end_global
    model_feed_start_global = max(0, model_feed_start_global)

    print(f"Actual chunk frame indices (global, from processed video): {actual_chunk_start_global} to {actual_chunk_end_global - 1}")
    print(f"Model will be fed frames (global, from processed video): {model_feed_start_global} to {model_feed_end_global - 1}")

    chunk_video_for_model = video_processed_full[:, model_feed_start_global:model_feed_end_global]
    if torch.cuda.is_available():
        chunk_video_for_model = chunk_video_for_model.to(device)

    depth_chunk_for_model = None
    if args.rgbd and depths_full_torch_processed is not None and depths_full_torch_processed.shape[0] > 0:
        depth_chunk_for_model = depths_full_torch_processed[model_feed_start_global:model_feed_end_global].unsqueeze(0)
        if torch.cuda.is_available():
            depth_chunk_for_model = depth_chunk_for_model.to(device)

    query_time_in_model_feed = actual_chunk_start_global - model_feed_start_global

    current_queries_for_model_input = None
    current_grid_size_for_model_input = 0
    current_segm_mask_for_model_input = None
    grid_query_frame_for_model_feed_input = 0

    if chunk_idx == 0:
        if args.grid_size > 0:
            print(f"Generating initial queries for first chunk using grid_size {args.grid_size} and mask.")
            grid_pts_on_processed_res = get_points_on_a_grid(args.grid_size, (H_processed, W_processed), device=torch.device('cpu'))

            point_mask_indices_y = grid_pts_on_processed_res[0, :, 1].round().long().clamp(0, H_processed - 1)
            point_mask_indices_x = grid_pts_on_processed_res[0, :, 0].round().long().clamp(0, W_processed - 1)
            point_mask = segm_mask_processed_np[point_mask_indices_y.numpy(), point_mask_indices_x.numpy()].astype(bool)

            masked_grid_pts = grid_pts_on_processed_res[:, point_mask]

            num_masked_points_to_take = min(800, masked_grid_pts.shape[1])
            if masked_grid_pts.shape[1] > num_masked_points_to_take:
                perm_indices = torch.randperm(masked_grid_pts.shape[1])[:num_masked_points_to_take]
                masked_grid_pts = masked_grid_pts[:, perm_indices]

            if masked_grid_pts.shape[1] > 0:
                current_queries_for_model_input = torch.zeros(1, masked_grid_pts.shape[1], 3, device=torch.device('cpu'))
                if args.query_frame != 0:
                    print(f"Warning: args.query_frame is {args.query_frame}. For segmentation mask on the *very first video frame* (after selection), set --query_frame 0.")
                current_queries_for_model_input[0, :, 0] = args.query_frame
                current_queries_for_model_input[0, :, 1:] = masked_grid_pts[0]

                if torch.cuda.is_available():
                    current_queries_for_model_input = current_queries_for_model_input.to(device)
                print(f"Generated {current_queries_for_model_input.shape[1]} initial queries from masked grid.")
            else:
                print("Warning: Segmentation mask resulted in 0 points from the grid. SpaTracker may run in dense mode or track no points.")
                current_queries_for_model_input = torch.empty(1, 0, 3).to(device if torch.cuda.is_available() else 'cpu')

            current_grid_size_for_model_input = 0
            current_segm_mask_for_model_input = None
            grid_query_frame_for_model_feed_input = 0
        else:
            print("No grid_size for the first chunk. SpaTracker will run in dense mode (no initial queries).")
            current_queries_for_model_input = None
    else:
        if prev_queries_for_next_chunk is not None and prev_queries_for_next_chunk.shape[1] > 0:
            current_queries_for_model_input = prev_queries_for_next_chunk.clone()
            current_queries_for_model_input[0, :, 0] = query_time_in_model_feed
            if torch.cuda.is_available():
                current_queries_for_model_input = current_queries_for_model_input.to(device)
            print(f"Using {current_queries_for_model_input.shape[1]} propagated queries. Query time for model input: {query_time_in_model_feed}")
        else:
            print(f"Warning: No valid queries from previous chunk for chunk {chunk_idx + 1}. SpaTracker may run in dense mode.")
            current_queries_for_model_input = None

    start_time_tracking = time.time()
    if not args.rgbd and depth_predictor_model_instance is not None and torch.cuda.is_available():
        depth_predictor_model_instance = depth_predictor_model_instance.to(device)

    with torch.no_grad():
        pred_tracks, pred_visibility, T_Firsts = model(
            chunk_video_for_model,
            video_depth=depth_chunk_for_model,
            queries=current_queries_for_model_input,
            segm_mask=current_segm_mask_for_model_input,
            grid_size=current_grid_size_for_model_input,
            grid_query_frame=grid_query_frame_for_model_feed_input,
            backward_tracking=args.backward,
            depth_predictor=depth_predictor_model_instance if (not args.rgbd and depth_predictor_model_instance is not None) else None,
            wind_length=S_MODEL_INTERNAL_WINDOW
        )
    end_time_tracking = time.time()
    print(f"Tracking for chunk {chunk_idx+1} (model input length {chunk_video_for_model.shape[1]}) took {end_time_tracking - start_time_tracking:.2f} seconds.")

    pred_slice_start = query_time_in_model_feed
    pred_slice_end = pred_slice_start + num_frames_in_actual_chunk

    pred_tracks_actual_chunk = pred_tracks[:, pred_slice_start:pred_slice_end]
    pred_visibility_actual_chunk = pred_visibility[:, pred_slice_start:pred_slice_end]

    all_pred_tracks_list.append(pred_tracks_actual_chunk.cpu())
    all_pred_visibility_list.append(pred_visibility_actual_chunk.cpu())

    video_actual_chunk_cpu = video_processed_full[:, actual_chunk_start_global:actual_chunk_end_global].cpu()
    for t_idx_in_actual_chunk in range(video_actual_chunk_cpu.shape[1]):
        frame_t = video_actual_chunk_cpu[0, t_idx_in_actual_chunk].permute(1, 2, 0).numpy()
        if frame_t.dtype in (np.float32, np.float64):
            if frame_t.min() >= 0 and frame_t.max() <= 1.0:
                frame_t = frame_t * 255.0
        frame_t = np.clip(frame_t, 0, 255).astype(np.uint8)
        processed_video_frames_for_moviepy.append(frame_t)

    if pred_tracks.shape[1] > 0 and pred_tracks.shape[2] > 0:
        num_points_in_pred = pred_tracks.shape[2]
        prev_queries_for_next_chunk = torch.zeros(1, num_points_in_pred, 3, device=pred_tracks.device)
        prev_queries_for_next_chunk[0, :, 1:] = pred_tracks[0, -1, :, :2]
        prev_queries_for_next_chunk[0, :, 0] = 0
    else:
        prev_queries_for_next_chunk = None
        print(f"Warning: Chunk {chunk_idx+1} model output resulted in no tracks to propagate ({pred_tracks.shape}).")

    del chunk_video_for_model, depth_chunk_for_model, current_queries_for_model_input, current_segm_mask_for_model_input
    del pred_tracks, pred_visibility, T_Firsts, pred_tracks_actual_chunk, pred_visibility_actual_chunk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if not all_pred_tracks_list or not processed_video_frames_for_moviepy:
    print("No tracks or video frames were processed/predicted. Exiting.")
    if processed_video_frames_for_moviepy:
        final_clip_moviepy = ImageSequenceClip(processed_video_frames_for_moviepy, fps=args.fps_vis)
        final_output_file = os.path.join(outdir, f"{video_stem}_final_no_tracks_processed_res.mp4")
        try:
            final_clip_moviepy.write_videofile(final_output_file, codec="libx264", fps=args.fps_vis, logger='bar')
            print(f"Final video (no tracks, processed resolution) saved to: {final_output_file}")
        except Exception as e:
            print(f"Error saving final video (no tracks) with moviepy: {e}")
            print(traceback.format_exc())
        finally:
            if 'final_clip_moviepy' in locals() and hasattr(final_clip_moviepy, 'close'):
                final_clip_moviepy.close()
    sys.exit()

final_pred_tracks_full_video = torch.cat(all_pred_tracks_list, dim=1)
final_pred_visibility_full_video = torch.cat(all_pred_visibility_list, dim=1)

print("\n--- Visualizing Full Prediction ---")

# --- STEP 1: Save a sample frame of the video intended for visualization ---
if video_for_overlay_original_res.nelement() > 0 and video_for_overlay_original_res.shape[1] > 0:
    sample_frame_to_save_np = video_for_overlay_original_res[0, 0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    try:
        pil_img = Image.fromarray(sample_frame_to_save_np)
        sample_frame_path = os.path.join(outdir, f"{video_stem}_DEBUG_sample_overlay_input_frame.png")
        pil_img.save(sample_frame_path)
        print(f"DEBUG: Saved a sample input frame for visualization to {sample_frame_path}")
    except Exception as e_save_frame:
        print(f"DEBUG: Could not save sample input frame: {e_save_frame}")
        print(traceback.format_exc())
else:
    print("DEBUG: video_for_overlay_original_res is empty, cannot save a sample frame.")
# --- END STEP 1 ---

# Initialize Visualizer
USE_GRAYSCALE_TEST = False
print(f"DEBUG: Initializing Visualizer with grayscale={USE_GRAYSCALE_TEST}")
vis_tool = Visualizer(
    save_dir=outdir,
    grayscale=USE_GRAYSCALE_TEST,
    fps=args.fps_vis,
    pad_value=0,
    linewidth=args.point_size,
    tracks_leave_trace=args.len_track
)

if final_pred_tracks_full_video.shape[2] == 0:
    print("No track points to visualize after concatenation.")
    if processed_video_frames_for_moviepy:
        final_clip_moviepy = ImageSequenceClip(processed_video_frames_for_moviepy, fps=args.fps_vis)
        final_output_file = os.path.join(outdir, f"{video_stem}_final_no_tracks_to_visualize_processed_res.mp4")
        try:
            final_clip_moviepy.write_videofile(final_output_file, codec="libx264", fps=args.fps_vis, logger='bar')
            print(f"Final video (no tracks to visualize, processed resolution) saved to: {final_output_file}")
        except Exception as e:
            print(f"Error saving final video (no tracks) with moviepy: {e}")
            print(traceback.format_exc())
        finally:
            if 'final_clip_moviepy' in locals() and hasattr(final_clip_moviepy, 'close'):
                final_clip_moviepy.close()
    sys.exit()

# Prepare video tensor for visualizer
video_for_viz_input_tensor = video_for_overlay_original_res.float()  # float 0-255

tracks_xy_for_viz = final_pred_tracks_full_video[..., :2].clone()

if W_processed != W_after_crop or H_processed != H_after_crop:
    print(f"Scaling tracks from processing resolution {W_processed}x{H_processed} to overlay resolution {W_after_crop}x{H_after_crop}.")
    scale_x = W_after_crop / W_processed if W_processed > 0 else 1.0
    scale_y = H_after_crop / H_processed if H_processed > 0 else 1.0

    tracks_xy_for_viz[..., 0] = tracks_xy_for_viz[..., 0] * scale_x
    tracks_xy_for_viz[..., 1] = tracks_xy_for_viz[..., 1] * scale_y
else:
    print(f"No track scaling needed for visualization. Overlay resolution: {W_after_crop}x{H_after_crop} (same as processing).")

print(f"DEBUG: Shape of video_for_viz_input_tensor: {video_for_viz_input_tensor.shape}, Min: {video_for_viz_input_tensor.min():.4f}, Max: {video_for_viz_input_tensor.max():.4f}")

# ---------------------- SAVE TRACKING DATA AS .NPY ----------------------
try:
    # (T, N, 2) in processed resolution
    tracks_xy_processed_np = final_pred_tracks_full_video[0, :, :, :2].detach().cpu().numpy().astype(np.float32)
    # (T, N, 2) in overlay/original resolution (after scaling above)
    tracks_xy_overlay_np = tracks_xy_for_viz[0].detach().cpu().numpy().astype(np.float32)
    # visibility -> (T, N) uint8
    vis_tensor = final_pred_visibility_full_video
    if vis_tensor.ndim == 4 and vis_tensor.shape[-1] == 1:
        vis_tensor = vis_tensor[..., 0]
    visibility_np = vis_tensor[0].detach().cpu().numpy().astype(np.uint8)

    tracking_save = {
        "tracks_xy_overlay": tracks_xy_overlay_np,          # (T, N, 2) pixels @ overlay/original resolution
        "tracks_xy_processed": tracks_xy_processed_np,      # (T, N, 2) pixels @ processed resolution
        "visibility": visibility_np,                        # (T, N) 0/1
        "frame_indices_from_original": frame_indices_to_process.cpu().numpy(),  # indices used from the original video
        "video_size_overlay_hw": (int(H_after_crop), int(W_after_crop)),
        "video_size_processed_hw": (int(H_processed), int(W_processed)),
        "args": vars(args),
    }
    tracks_npy_path = os.path.join(outdir, f"{video_stem}_tracking_data.npy")
    np.save(tracks_npy_path, tracking_save, allow_pickle=True)
    print(f"Saved tracking data (NumPy .npy) to: {tracks_npy_path}")
except Exception as e:
    print(f"Failed to save tracking data .npy: {e}")
    print(traceback.format_exc())
# ---------------------- END SAVE .NPY ----------------------

drawn_frames_tensor_uint8 = vis_tool.draw_tracks_on_video(
    video=video_for_viz_input_tensor,
    tracks=tracks_xy_for_viz,
    visibility=final_pred_visibility_full_video.cpu(),
)

if drawn_frames_tensor_uint8 is not None and drawn_frames_tensor_uint8.nelement() > 0 and drawn_frames_tensor_uint8.shape[1] > 0:
    # --- Save a sample frame of what draw_tracks_on_video returns ---
    try:
        sample_output_frame_np = drawn_frames_tensor_uint8[0, 0].permute(1, 2, 0).cpu().numpy()
        if sample_output_frame_np.dtype != np.uint8:
            print(f"DEBUG: Visualizer output frame was {sample_output_frame_np.dtype}, converting to uint8 for saving.")
            if sample_output_frame_np.max() <= 1.0 and sample_output_frame_np.min() >= 0.0 and sample_output_frame_np.dtype == np.float32:
                sample_output_frame_np = (sample_output_frame_np * 255)
            sample_output_frame_np = np.clip(sample_output_frame_np, 0, 255).astype(np.uint8)

        pil_output_img = Image.fromarray(sample_output_frame_np)
        sample_output_frame_path = os.path.join(outdir, f"{video_stem}_DEBUG_sample_visualizer_output_frame.png")
        pil_output_img.save(sample_output_frame_path)
        print(f"DEBUG: Saved a sample output frame from visualizer to {sample_output_frame_path}")
    except Exception as e_save_output_frame:
        print(f"DEBUG: Could not save sample output frame from visualizer: {e_save_output_frame}")
        print(traceback.format_exc())
    # --- End save sample output frame ---

    frames_for_moviepy_visualization = []
    for frame_tensor in drawn_frames_tensor_uint8[0]:  # uint8 expected
        frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
        if frame_np.dtype != np.uint8:
            print(f"WARN: Frame from visualizer output is {frame_np.dtype}, expected uint8. Clipping and casting.")
            if frame_np.max() <= 1.0 and frame_np.min() >= 0.0 and frame_np.dtype == np.float32:
                frame_np = (frame_np * 255)
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        frames_for_moviepy_visualization.append(frame_np)

    if frames_for_moviepy_visualization:
        final_clip = ImageSequenceClip(frames_for_moviepy_visualization, fps=args.fps_vis)
        final_output_file = os.path.join(outdir, f"{video_stem}_final_pred_overlay_original_res.mp4")

        try:
            print("-" * 50)
            print(f"DEBUG: Attempting to write video to: {final_output_file}")
            print(f"DEBUG: FPS for video: {args.fps_vis}, Type: {type(args.fps_vis)}")
            print(f"DEBUG: Codec: libx264")
            print(f"DEBUG: Logger: bar")
            if 'final_clip' in locals() and final_clip is not None:
                print(f"DEBUG: final_clip object: {final_clip}")
                print(f"DEBUG: Is final_clip.write_videofile callable: {callable(final_clip.write_videofile)}")
            else:
                print("DEBUG: final_clip object is not defined or is None.")
            print("-" * 50)

            final_clip.write_videofile(final_output_file, codec="libx264", fps=args.fps_vis, logger='bar')
            print(f"Final visualization (overlay on original resolution) saved to: {final_output_file}")
        except Exception as e:
            print(f"Error writing final video with moviepy: {e}")
            print("----------- FULL TRACEBACK -----------")
            print(traceback.format_exc())
            print("--------------------------------------")
        finally:
            if 'final_clip' in locals() and hasattr(final_clip, 'close'):
                final_clip.close()
    else:
        print("Conversion of visualized frames to list resulted in no frames (original resolution overlay).")
else:
    print("Visualization (draw_tracks_on_video for original resolution) resulted in no frames or an empty/invalid tensor.")
    if processed_video_frames_for_moviepy:
        print("Attempting to save the processed (downsampled) video without tracks as fallback.")
        final_clip_raw_moviepy = ImageSequenceClip(processed_video_frames_for_moviepy, fps=args.fps_vis)
        raw_output_file = os.path.join(outdir, f"{video_stem}_final_raw_fallback_processed_res.mp4")
        try:
            final_clip_raw_moviepy.write_videofile(raw_output_file, codec="libx264", fps=args.fps_vis, logger='bar')
            print(f"Raw processed video (no tracks, processed resolution) saved to: {raw_output_file}")
        except Exception as e_raw:
            print(f"Error saving raw video with moviepy: {e_raw}")
            print(traceback.format_exc())
        finally:
            if 'final_clip_raw_moviepy' in locals() and hasattr(final_clip_raw_moviepy, 'close'):
                final_clip_raw_moviepy.close()

print(f"Script finished. Results in '{outdir}'")
