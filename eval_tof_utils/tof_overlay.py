"""
Create an H.264 MP4 where SpatialTracker points are drawn on top of TOF depth frames.

- Coordinates are mapped from processed video space → TOF space.
- First-frame mask is warped to each frame via homography estimated from tracks that
  started inside the mask and are visible in both t=0 and t (geometric prior).
- The warped mask is slightly dilated to be lenient to small warp errors.
- A per-frame object depth band is learned from TOF depths inside the warped mask and
  smoothed with an EMA; points are also validated by a robust local median depth.
- Temporal smoothing (majority vote over recent frames) prevents flicker.

A point is green if (in warped+dilated mask) OR (local depth within depth band).
Otherwise it’s red. Optionally, if the tracker marks a point invisible and TOF has
no return locally, it’s drawn yellow.

Returned dict:
{
    'frames_written', 'size', 'vmin', 'vmax', 'fps',
    'raw_out_path', 'final_out_path'
}
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque

import numpy as np
import cv2

from .tof_to_mp4 import (
    list_npy_files,
    compute_norm_bounds,
    normalize_to_uint8,
    to_bgr,
    ensure_h264,
)

# ---------------------- helpers ----------------------

def _natural_sort(files: List[str]) -> List[str]:
    """Sort files naturally (e.g., frame1, frame2, ..., frame10)."""
    return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

def probe_video_fps(video_path: str) -> Optional[float]:
    """Try to read FPS from a (H.264) video using OpenCV. Returns None if not available."""
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1e-3 and np.isfinite(fps):
            return float(fps)
        return None
    finally:
        cap.release()

def _load_mask_resized(mask_path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    """Load binary mask (0/255) and resize to (H_proc, W_proc) using nearest neighbor; return 0/1 mask."""
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Mask image not found or unreadable: {mask_path}")
    Ht, Wt = target_hw
    m = cv2.resize(m, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)  # 0/1

def _estimate_homography(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Robust homography with RANSAC; identity if not enough inliers."""
    if p0.shape[0] >= 4 and p1.shape[0] >= 4:
        H, _ = cv2.findHomography(
            p0, p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995
        )
        if H is not None:
            return H
    return np.eye(3, dtype=np.float64)

def _robust_local_depth(depth: np.ndarray, x: int, y: int, win: int = 2) -> float:
    """
    Median depth in a (2*win+1)^2 window around (x, y), ignoring 0 (no return).
    Returns 0 if no valid samples.
    """
    h, w = depth.shape
    x0, x1 = max(0, x - win), min(w, x + win + 1)
    y0, y1 = max(0, y - win), min(h, y + win + 1)
    patch = depth[y0:y1, x0:x1]
    vals = patch[(patch > 0) & np.isfinite(patch)]
    return float(np.median(vals)) if vals.size else 0.0

def _mad(vals: np.ndarray, med: float) -> float:
    """Median absolute deviation."""
    if vals.size == 0:
        return 0.0
    return float(np.median(np.abs(vals - med)))

# ---------------------- core ----------------------

def create_tof_overlay_video(
    tracking_file: str,
    tof_data_folder: str,
    out_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    out_name: str = "tof_overlay.mp4",
    fps: float = 30.0,
    pattern: str = "*.npy",
    # depth visualization:
    colormap: str = "turbo",
    percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mask_zero: bool = True,
    global_norm: bool = True,
    global_samples: int = 50,
    # first-frame object mask (PNG from step 2)
    mask_path: Optional[str] = None,
    # drawing
    point_radius: int = 4,
    point_thickness: int = -1,
    draw_ids: bool = False,
    font_scale: float = 0.4,
    font_thickness: int = 1,
    tail_length: int = 0,
    on_color: Tuple[int,int,int] = (0,255,0),   # green (BGR)
    off_color: Tuple[int,int,int] = (0,0,255),  # red (BGR)
    unsure_color: Tuple[int,int,int] = (0,255,255),  # yellow (BGR)
    # encoding:
    raw_suffix: str = "_raw_overlay.mp4",
    h264_crf: int = 23,
    h264_preset: str = "slow",
) -> Dict[str, object]:
    """
    Build a TOF overlay video: TOF depth background + SpatialTracker points with robust correctness.

    Returns a dict with fields:
      { 'frames_written', 'size', 'vmin', 'vmax', 'fps', 'raw_out_path', 'final_out_path' }
    """

    # ---- load tracking ----
    tdata = np.load(tracking_file, allow_pickle=True).item()
    tracks_xy = tdata["tracks_xy_processed"]   # (F,P,2) in processed coords
    visibility = tdata["visibility"]           # (F,P)
    H_proc, W_proc = tdata["video_size_processed_hw"]

    # ---- TOF files ----
    tof_files = list_npy_files(tof_data_folder, pattern)
    if not tof_files:
        raise FileNotFoundError(f"No .npy found in '{tof_data_folder}' with pattern '{pattern}'.")

    # ---- output paths ----
    if out_path:
        final_out = out_path
    else:
        base_dir = out_dir if out_dir else tof_data_folder
        os.makedirs(base_dir, exist_ok=True)
        final_out = os.path.join(base_dir, out_name)
    raw_out = final_out.replace(".mp4", raw_suffix)

    # ---- TOF size ----
    first = np.load(tof_files[0], allow_pickle=False)
    if first.ndim == 3 and first.shape[-1] == 1:
        first = first[..., 0]
    if first.ndim != 2:
        raise ValueError(f"Unexpected TOF array shape {first.shape} in {tof_files[0]}.")
    H_tof, W_tof = first.shape[:2]
    frame_size = (W_tof, H_tof)

    # ---- normalization (for background visualization only) ----
    gvmin, gvmax = vmin, vmax
    if global_norm and (gvmin is None or gvmax is None):
        idxs = np.linspace(0, len(tof_files) - 1, num=min(global_samples, len(tof_files)), dtype=int)
        sample_arrays = []
        for p in [tof_files[i] for i in idxs]:
            a = np.load(p, allow_pickle=False)
            if a.ndim == 3 and a.shape[-1] == 1: a = a[..., 0]
            if a.ndim != 2: raise ValueError(f"Unexpected array shape {a.shape} in {p}.")
            sample_arrays.append(a)
        gvmin, gvmax = compute_norm_bounds(sample_arrays, vmin, vmax, percentiles, mask_zero)

    # ---- writer ----
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size, True)
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for raw overlay video (mp4v).")

    # ---- prepare first-frame mask (processed size) ----
    mask_proc: Optional[np.ndarray] = None
    if mask_path:
        mask_proc = _load_mask_resized(mask_path, (H_proc, W_proc))  # 0/1 uint8

    F, P, _ = tracks_xy.shape
    num_frames = min(len(tof_files), F)
    scale_x = W_tof / float(W_proc)
    scale_y = H_tof / float(H_proc)

    # Homography (mask warp) state
    H_prev = np.eye(3, dtype=np.float64)
    valid0 = (visibility[0] > 0)

    # Per-point membership of the t=0 point in the initial mask (used to seed homography)
    in_mask0 = np.zeros(P, dtype=bool)
    if mask_proc is not None:
        for pid in range(P):
            if not valid0[pid]:
                continue
            x0, y0 = tracks_xy[0, pid]
            ix0, iy0 = int(round(x0)), int(round(y0))
            if 0 <= ix0 < W_proc and 0 <= iy0 < H_proc:
                in_mask0[pid] = bool(mask_proc[iy0, ix0])

    # Per-point trails and decision smoothing buffer
    tails: List[List[Tuple[int,int]]] = [[] for _ in range(P)]
    hist_len = 5  # temporal smoothing window
    decide_bufs: List[Deque[bool]] = [deque(maxlen=hist_len) for _ in range(P)]

    # Depth band EMA state
    ema_med: Optional[float] = None
    ema_mad: Optional[float] = None
    ema_alpha = 0.3  # 0..1

    # ---- main loop ----
    frames_written = 0
    for t in range(num_frames):
        depth = np.load(tof_files[t], allow_pickle=False)
        if depth.ndim == 3 and depth.shape[-1] == 1: depth = depth[..., 0]
        if depth.ndim != 2: raise ValueError(f"Unexpected array shape {depth.shape} in {tof_files[t]}.")

        # Background image (visualization only)
        if not global_norm and (vmin is None or vmax is None):
            vmin_f, vmax_f = compute_norm_bounds([depth], None, None, percentiles, mask_zero)
        else:
            vmin_f, vmax_f = gvmin, gvmax
        img_u8 = normalize_to_uint8(depth, float(vmin_f), float(vmax_f), mask_zero)
        frame_bgr = to_bgr(img_u8, colormap)

        # ---- warp first-frame mask to frame t (processed coords) ----
        mask_t_bool = None
        if mask_proc is not None:
            # Use tracks that were visible at t=0, are visible at t, and started inside the mask
            both_vis = (visibility[0] > 0) & (visibility[t] > 0) & in_mask0  # (P,)

            p0_list, p1_list = [], []
            for pid in np.where(both_vis)[0]:
                x0, y0 = tracks_xy[0, pid]
                xt, yt = tracks_xy[t, pid]
                p0_list.append([x0, y0])
                p1_list.append([xt, yt])

            if len(p0_list) >= 4:
                p0 = np.array(p0_list, dtype=np.float32)
                p1 = np.array(p1_list, dtype=np.float32)
                H_t = _estimate_homography(p0, p1)
                H_prev = H_t  # keep the latest good homography
            H_comb = H_prev

            mask_t = cv2.warpPerspective(
                (mask_proc * 255).astype(np.uint8),
                H_comb,
                (W_proc, H_proc),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            # small dilation → be lenient to warp errors
            kernel = np.ones((5, 5), np.uint8)
            mask_t = cv2.dilate(mask_t, kernel, iterations=1)
            mask_t_bool = (mask_t > 127)

        # ---- learn per-frame object depth band from warped mask ----
        med_band, mad_band = None, None
        if mask_t_bool is not None:
            ys, xs = np.where(mask_t_bool)
            if xs.size:
                xs_tof = np.clip((xs * scale_x).round().astype(int), 0, W_tof - 1)
                ys_tof = np.clip((ys * scale_y).round().astype(int), 0, H_tof - 1)
                samples = depth[ys_tof, xs_tof]
                samples = samples[(samples > 0) & np.isfinite(samples)]
                if samples.size >= 50:  # need some mass
                    med = float(np.median(samples))
                    mad = _mad(samples, med)
                    med_band, mad_band = med, max(mad, 0.03)  # minimum tolerance

        # EMA smoothing / fallback
        if med_band is not None:
            ema_med = med_band if ema_med is None else (1.0 - ema_alpha) * ema_med + ema_alpha * med_band
            ema_mad = mad_band if ema_mad is None else (1.0 - ema_alpha) * ema_mad + ema_alpha * mad_band
        if ema_med is None:
            # Bootstrap from scene depth if mask didn’t provide samples yet
            nonzero = depth[(depth > 0) & np.isfinite(depth)]
            ema_med = float(np.median(nonzero)) if nonzero.size else 1.5
        if ema_mad is None:
            ema_mad = 0.08  # ≈ 8 cm baseline tolerance

        # Build depth band
        k = 2.5
        band_lo = ema_med - max(0.06, k * ema_mad)  # min total half-width ~6 cm
        band_hi = ema_med + max(0.06, k * ema_mad)

        # ---- draw points with robust correctness ----
        vis = visibility[t] > 0
        for pid in np.where(vis)[0]:
            px, py = tracks_xy[t, pid]  # processed coords
            ixp, iyp = int(round(px)), int(round(py))

            # geometric membership (processed coords)
            on_geom = False
            if mask_t_bool is not None and 0 <= ixp < W_proc and 0 <= iyp < H_proc:
                on_geom = bool(mask_t_bool[iyp, ixp])

            # depth membership (TOF coords + local median)
            x_tof = int(round(px * scale_x))
            y_tof = int(round(py * scale_y))
            in_bounds = (0 <= x_tof < W_tof and 0 <= y_tof < H_tof)

            on_depth = False
            if in_bounds:
                local_depth = _robust_local_depth(depth, x_tof, y_tof, win=2)
                if local_depth > 0:
                    on_depth = (band_lo <= local_depth <= band_hi)

            # decision (store into smoothing buffer)
            correct_now = bool(on_geom or on_depth)
            decide_bufs[pid].append(correct_now)

            # temporal smoothing: majority of last N
            recent = decide_bufs[pid]
            correct = (sum(recent) > len(recent) // 2)

            # choose color
            color = on_color if correct else off_color

            # draw (TOF coords)
            if in_bounds:
                cv2.circle(frame_bgr, (x_tof, y_tof), point_radius, color,
                           thickness=point_thickness, lineType=cv2.LINE_AA)
                if draw_ids:
                    cv2.putText(frame_bgr, str(pid), (x_tof + 6, y_tof - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                                font_thickness, cv2.LINE_AA)

                # trails in TOF coords
                if tail_length > 0:
                    trail = tails[pid]
                    trail.append((x_tof, y_tof))
                    if len(trail) > tail_length:
                        trail.pop(0)
                    if len(trail) >= 2:
                        cv2.polylines(frame_bgr, [np.array(trail, dtype=np.int32)],
                                      False, color, 1, cv2.LINE_AA)

            # Optional: visualize uncertainty if tracker says invisible and TOF has no return nearby
            if visibility[t, pid] == 0 and in_bounds:
                if _robust_local_depth(depth, x_tof, y_tof, win=1) == 0.0:
                    cv2.circle(frame_bgr, (x_tof, y_tof), point_radius, unsure_color,
                               thickness=point_thickness, lineType=cv2.LINE_AA)

        writer.write(frame_bgr)
        frames_written += 1

    writer.release()

    ensured = ensure_h264(raw_out, output_path=final_out, crf=h264_crf, preset=h264_preset)

    return {
        "frames_written": frames_written,
        "size": frame_size,
        "vmin": float(gvmin) if gvmin is not None else None,
        "vmax": float(gvmax) if gvmax is not None else None,
        "fps": fps,
        "raw_out_path": raw_out,
        "final_out_path": ensured,
    }
