from __future__ import annotations
import csv
import json
import os
from typing import Optional, Tuple, Deque, List
from collections import deque

import numpy as np
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _robust_local_depth(depth: np.ndarray, x: int, y: int, win: int = 2) -> float:
    h, w = depth.shape
    x0, x1 = max(0, x - win), min(w, x + win + 1)
    y0, y1 = max(0, y - win), min(h, y + win + 1)
    patch = depth[y0:y1, x0:x1]
    vals = patch[(patch > 0) & np.isfinite(patch)]
    return float(np.median(vals)) if vals.size else 0.0

def _mad(vals: np.ndarray, med: float) -> float:
    if vals.size == 0:
        return 0.0
    return float(np.median(np.abs(vals - med)))

def _load_and_check_tracking(tracking_file: str):
    data = np.load(tracking_file, allow_pickle=True).item()
    tracks_xy = data["tracks_xy_processed"]  # (F,P,2)
    visibility = data["visibility"]          # (F,P)
    H_proc, W_proc = data["video_size_processed_hw"]
    return tracks_xy, visibility, (H_proc, W_proc)

def _collect_tof_files(folder: str) -> list[str]:
    return sorted(
        f for f in (os.path.join(folder, x) for x in os.listdir(folder))
        if f.endswith(".npy")
    )

def _plot_correct_incorrect_over_time(
    frames: List[int],
    correct: List[int],
    visible: List[int],
    out_png: str,
    out_svg: Optional[str] = None,
) -> None:
    """Save a line chart: correct vs. incorrect points over time."""
    incorrect = [v - c for v, c in zip(visible, correct)]

    plt.figure(figsize=(12, 4))
    plt.plot(frames, correct, label="Correct points")
    plt.plot(frames, incorrect, label="Incorrect points")
    plt.xlabel("Frame")
    plt.ylabel("# Points")
    plt.title("Tracked Points Over Time (Correct vs. Incorrect)")
    plt.grid(True, linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    if out_svg:
        plt.savefig(out_svg)
    plt.close()

def analyze_results(
    tracking_file: str,
    tof_data_folder: str,
    visibility_threshold: float = 0.5,  # kept for compatibility, not used directly now
    output_dir: Optional[str] = None,
) -> None:
    """
    Depth-aware analysis consistent with overlay logic.
    Writes:
      - metrics_summary.json
      - metrics_per_frame.csv
      - accuracy_over_time.png (and .svg)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("\n📊 Analysis: loading tracking and TOF data ...")

    try:
        tracks_xy, visibility, (H_proc, W_proc) = _load_and_check_tracking(tracking_file)
        F, P, _ = tracks_xy.shape
        print(f"Tracker loaded: {F} frames, {P} points, @{W_proc}x{H_proc}.")
    except Exception as e:
        print(f"❌ Failed to load tracking file '{tracking_file}': {e}")
        return

    tof_files = _collect_tof_files(tof_data_folder)
    if not tof_files:
        print(f"❌ No .npy files found in TOF folder: {tof_data_folder}")
        return

    # Determine TOF size & scaling
    first = np.load(tof_files[0])
    if first.ndim == 3 and first.shape[-1] == 1:
        first = first[..., 0]
    H_tof, W_tof = first.shape
    sx, sy = W_tof / W_proc, H_tof / H_proc

    # Depth band EMA state (no mask used here to keep analysis independent)
    ema_med: Optional[float] = None
    ema_mad: Optional[float] = None
    ema_alpha = 0.3
    k = 2.5

    per_frame_rows = []
    total_visible = 0
    total_correct = 0

    hist_len = 5
    # per-point temporal buffers for smoothing (match overlay behavior)
    decide_bufs: List[Deque[bool]] = [deque(maxlen=hist_len) for _ in range(P)]

    frames_idx: List[int] = []
    frames_visible: List[int] = []
    frames_correct: List[int] = []

    print(f"Comparing {min(F, len(tof_files))} frames ...")
    for t in tqdm(range(min(F, len(tof_files))), desc="Frame analysis"):
        depth = np.load(tof_files[t])
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        # Estimate object depth band from points' local depth this frame (robust)
        vis_mask = (visibility[t] > 0)
        zs = []
        for pid in np.where(vis_mask)[0]:
            px, py = tracks_xy[t, pid]
            x_tof = int(round(px * sx))
            y_tof = int(round(py * sy))
            if 0 <= x_tof < W_tof and 0 <= y_tof < H_tof:
                z = _robust_local_depth(depth, x_tof, y_tof, win=2)
                if z > 0:
                    zs.append(z)
        if zs:
            zs = np.array(zs, dtype=np.float32)
            med = float(np.median(zs))
            mad = _mad(zs, med)
            ema_med = med if ema_med is None else (1 - ema_alpha) * ema_med + ema_alpha * med
            ema_mad = mad if ema_mad is None else (1 - ema_alpha) * ema_mad + ema_alpha * mad

        if ema_med is None:
            nz = depth[(depth > 0) & np.isfinite(depth)]
            ema_med = float(np.median(nz)) if nz.size else 1.5
        if ema_mad is None:
            ema_mad = 0.08

        band_lo = ema_med - max(0.06, k * ema_mad)
        band_hi = ema_med + max(0.06, k * ema_mad)

        visible_pts = int(np.sum(vis_mask))
        total_visible += visible_pts
        correct_pts = 0

        for pid in np.where(vis_mask)[0]:
            px, py = tracks_xy[t, pid]
            x_tof = int(round(px * sx))
            y_tof = int(round(py * sy))
            correct_now = False
            if 0 <= x_tof < W_tof and 0 <= y_tof < H_tof:
                z = _robust_local_depth(depth, x_tof, y_tof, win=2)
                if z > 0 and band_lo <= z <= band_hi:
                    correct_now = True

            decide_bufs[pid].append(correct_now)
            # temporal majority smoothing
            recent = decide_bufs[pid]
            if sum(recent) > len(recent) // 2:
                correct_pts += 1

        total_correct += correct_pts
        acc = (100.0 * correct_pts / max(1, visible_pts))

        per_frame_rows.append({
            "frame": t,
            "visible_points": visible_pts,
            "correct_points": correct_pts,
            "accuracy_percent": acc,
            "band_lo": band_lo,
            "band_hi": band_hi,
            "band_med": ema_med,
            "band_mad": ema_mad,
        })

        frames_idx.append(t)
        frames_visible.append(visible_pts)
        frames_correct.append(correct_pts)

    overall_acc = (100.0 * total_correct / max(1, total_visible))
    print("\n--- Summary (depth-aware) ---")
    print(f"Frames analyzed: {len(per_frame_rows)}")
    print(f"Visible points total: {total_visible}")
    print(f"Correct points (smoothed): {total_correct}")
    print(f"Average accuracy: {overall_acc:.2f}%")

    if output_dir:
        with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
            json.dump({
                "frames_analyzed": len(per_frame_rows),
                "visible_points_total": int(total_visible),
                "correct_points_total": int(total_correct),
                "average_accuracy_percent": float(overall_acc),
            }, f, indent=2)

        # Per-frame CSV
        with open(os.path.join(output_dir, "metrics_per_frame.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "frame", "visible_points", "correct_points",
                "accuracy_percent", "band_lo", "band_hi", "band_med", "band_mad"
            ])
            w.writeheader()
            w.writerows(per_frame_rows)

        # Plot: correct vs incorrect over time
        png_path = os.path.join(output_dir, "accuracy_over_time.png")
        svg_path = os.path.join(output_dir, "accuracy_over_time.svg")
        _plot_correct_incorrect_over_time(frames_idx, frames_correct, frames_visible, png_path, svg_path)
        print(f"Saved plot: {png_path}")
