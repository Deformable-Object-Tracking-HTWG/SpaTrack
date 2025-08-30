from performance_metrics import Performance_Metrics
from pathlib import Path
import torch
import numpy as np
import glob
import os
import time

# --------------------- Konfiguration ---------------------
video_paths = [
    "/home/TP_tracking/Documents/depthCameraResources/videos/Gymnastik_1_5s/Gymnastik_1_5s",
   # "/home/TP_tracking/Documents/depthCameraResources/videos/Gymnastik_2_5s",
    # weitere Pfade hier ...
]

base_results_dir = Path("./results")
base_results_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chunk_size = 40
grid_size = 2
fps_vis = 30
downsample = 1.0

# --------------------- Hilfsfunktionen ---------------------
def load_depth_sequence_from_npy(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {folder}")
    frames = [np.load(f) for f in files]
    depth_array = np.stack(frames, axis=0)  # (T,H,W)
    return torch.from_numpy(depth_array).float().unsqueeze(0)  # (1,T,H,W)

def create_dummy_rgb(T, H, W):
    # Dummy-RGB Tensor für SpaTracker (falls RGB benötigt wird)
    return torch.zeros((1, T, H, W, 3))  # HWC pro Frame

# --------------------- SpaTracker-Funktion ---------------------
def run_spatracker(video_processed_full, depths_full_torch_processed, video_name, output_dir):
    T_full_processed = video_processed_full.shape[1]
    H_processed = video_processed_full.shape[2]
    W_processed = video_processed_full.shape[3]

    # Performance-Monitoring
    monitor = Performance_Metrics(sampling_interval=0.25, enable_pyspy=False)
    monitor.start()

    # --------------------- Chunking ---------------------
    overlap_frames = 20  # 50% von 40
    num_chunks = (T_full_processed + chunk_size - 1) // chunk_size
    if T_full_processed == 0:
        num_chunks = 0
    print(f"{T_full_processed} Frames, {num_chunks} Chunk(s) werden verarbeitet.")

    all_pred_tracks_list = []
    all_pred_visibility_list = []
    processed_video_frames_for_moviepy = []

    prev_queries_for_next_chunk = None

    for chunk_idx in range(num_chunks):
        print(f"\n--- Chunk {chunk_idx+1}/{num_chunks} ---")

        actual_chunk_start_global = chunk_idx * chunk_size
        actual_chunk_end_global = min((chunk_idx+1) * chunk_size, T_full_processed)
        num_frames_in_actual_chunk = actual_chunk_end_global - actual_chunk_start_global
        if num_frames_in_actual_chunk <= 0:
            print(f"Skipping empty chunk {chunk_idx+1}")
            continue

        current_dynamic_overlap = overlap_frames if chunk_idx > 0 else 0
        model_feed_start_global = max(0, actual_chunk_start_global - current_dynamic_overlap)
        model_feed_end_global = actual_chunk_end_global

        chunk_video_for_model = video_processed_full[:, model_feed_start_global:model_feed_end_global].to(device)
        depth_chunk_for_model = depths_full_torch_processed[:, model_feed_start_global:model_feed_end_global].to(device)

        query_time_in_model_feed = actual_chunk_start_global - model_feed_start_global

        # --------------------- Initial Queries ---------------------
        if chunk_idx == 0:
            if grid_size > 0:
                # Grid Punkte generieren (hier Dummy-Implementierung)
                num_masked_points_to_take = min(800, grid_size*grid_size)
                current_queries_for_model_input = torch.zeros(1, num_masked_points_to_take, 3, device=device)
            else:
                current_queries_for_model_input = None
        else:
            if prev_queries_for_next_chunk is not None:
                current_queries_for_model_input = prev_queries_for_next_chunk.clone()
                current_queries_for_model_input[0, :, 0] = query_time_in_model_feed
            else:
                current_queries_for_model_input = None

        # --------------------- Hier Modell aufrufen ---------------------
        # TODO: Ersetze 'model' durch dein SpaTracker-Modell, z.B. model(video, depth, queries,...)
        pred_tracks, pred_visibility, T_Firsts = "./test_metrics.py"
        # Für die Demo hier Dummy-Tensoren
        pred_tracks = torch.zeros(1, chunk_video_for_model.shape[1], 50, 2, device=device)
        pred_visibility = torch.ones_like(pred_tracks[:, :, :, 0])

        # --------------------- Ergebnisse speichern ---------------------
        pred_slice_start = query_time_in_model_feed
        pred_slice_end = pred_slice_start + num_frames_in_actual_chunk
        all_pred_tracks_list.append(pred_tracks[:, pred_slice_start:pred_slice_end].cpu())
        all_pred_visibility_list.append(pred_visibility[:, pred_slice_start:pred_slice_end].cpu())

        # Propagiere Queries
        if pred_tracks.shape[2] > 0:
            num_points_in_pred = pred_tracks.shape[2]
            prev_queries_for_next_chunk = torch.zeros(1, num_points_in_pred, 3, device=device)
            prev_queries_for_next_chunk[0, :, 1:] = pred_tracks[0, -1, :, :2]
            prev_queries_for_next_chunk[0, :, 0] = 0
        else:
            prev_queries_for_next_chunk = None

        del chunk_video_for_model, depth_chunk_for_model, current_queries_for_model_input, pred_tracks, pred_visibility

    monitor.stop()
    monitor.output(str(output_dir))
    print(f"Ergebnisse gespeichert in: {output_dir}")

# --------------------- Hauptloop ---------------------
for video_path in video_paths:
    video_name = Path(video_path).stem
    output_dir = base_results_dir / video_name
    output_dir.mkdir(exist_ok=True)
    try:
        depths_full_torch_processed = load_depth_sequence_from_npy(video_path)
        T, H, W = depths_full_torch_processed.shape[1:4]
        video_processed_full = create_dummy_rgb(T, H, W)
        run_spatracker(video_processed_full, depths_full_torch_processed, video_name, output_dir)
    except FileNotFoundError as e:
        print(f"Fehler: {e}")
