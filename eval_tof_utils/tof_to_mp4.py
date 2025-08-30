# -*- coding: utf-8 -*-
"""
npy_to_mp4_module
-----------------
Turn a folder of `.npy` depth maps (e.g., TOF camera) into an MP4 and
*ensure the final file is H.264* using ffprobe/ffmpeg — just like the pattern you shared.

Key points
- Robust, TOF-friendly normalization (percentiles + ignore zeros).
- Attempts to write H.264 directly via OpenCV ('avc1'); if that's not available,
  falls back to a widely supported writer (e.g., 'mp4v') and then converts to H.264
  with ffmpeg (libx264), using ffprobe to detect the codec first.
- You can provide an output directory and file name separately, or a full `out_path`.

Dependencies
- numpy, opencv-python
- ffmpeg and ffprobe available on PATH (for codec probe/convert)
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import subprocess
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python") from e


# ---------------------- Utilities ----------------------

def natural_key(s: str):
    """Natural sort so that 'frame2' < 'frame10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_npy_files(input_dir: str, pattern: str = "*.npy") -> List[str]:
    files = sorted(glob.glob(os.path.join(input_dir, pattern)), key=natural_key)
    return [f for f in files if os.path.isfile(f)]


def apply_geometric_ops(img: np.ndarray, rotate: int = 0, flip: str = "none") -> np.ndarray:
    """Apply rotation (0/90/180/270) and optional flip ('h' or 'v')."""
    if rotate not in (0, 90, 180, 270):
        raise ValueError("rotate must be one of: 0, 90, 180, 270.")

    if rotate == 90:
        img = np.rot90(img, k=1)
    elif rotate == 180:
        img = np.rot90(img, k=2)
    elif rotate == 270:
        img = np.rot90(img, k=3)

    if flip == "h":
        img = np.fliplr(img)
    elif flip == "v":
        img = np.flipud(img)
    elif flip not in ("none", "", None):
        raise ValueError("flip must be 'h', 'v', or 'none'.")

    return img


def compute_norm_bounds(
    sample_arrays: Sequence[np.ndarray],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    mask_zero: bool = True,
) -> Tuple[float, float]:
    """
    Determine vmin/vmax either directly or via robust percentiles.
    Ignores NaN/Inf; optionally ignores zeros (typical for 'no return' in TOF).
    """
    if vmin is not None and vmax is not None:
        if not (vmax > vmin):
            raise ValueError("vmax must be > vmin.")
        return float(vmin), float(vmax)

    values: List[np.ndarray] = []
    target = 1_000_000  # collect up to ~1e6 values to limit memory
    per_array_cap = max(10_000, target // max(1, len(sample_arrays)))

    for arr in sample_arrays:
        a = arr.astype(np.float32, copy=False)
        valid = np.isfinite(a)
        if mask_zero:
            valid &= (a != 0)
        a = a[valid]
        if a.size == 0:
            continue
        if a.size > per_array_cap:
            # downsample randomly to reduce memory
            idx = np.random.default_rng(42).choice(a.size, size=per_array_cap, replace=False)
            a = a[idx]
        values.append(a)

    if not values:
        return 0.0, 1.0

    vals = np.concatenate(values, axis=0)

    if percentiles is not None:
        lo, hi = percentiles
        vmin_est = float(np.percentile(vals, lo))
        vmax_est = float(np.percentile(vals, hi))
    else:
        vmin_est = float(np.min(vals))
        vmax_est = float(np.max(vals))

    if not math.isfinite(vmin_est) or not math.isfinite(vmax_est) or vmax_est <= vmin_est:
        return 0.0, 1.0
    return vmin_est, vmax_est


def normalize_to_uint8(arr: np.ndarray, vmin: float, vmax: float, mask_zero: bool = True) -> np.ndarray:
    """Linearly scale arr to [0, 255] (uint8) using vmin/vmax, robust to zeros/NaN/Inf."""
    a = arr.astype(np.float32, copy=False)

    valid = np.isfinite(a)
    if mask_zero:
        valid &= (a != 0)

    out = np.zeros_like(a, dtype=np.uint8)
    if not np.any(valid):
        return out

    a = np.clip(a, vmin, vmax, out=a)
    scale = 255.0 / max(1e-12, (vmax - vmin))
    out_valid = ((a[valid] - vmin) * scale)
    out[valid] = np.clip(np.rint(out_valid), 0, 255).astype(np.uint8)
    return out


def to_bgr(img_u8: np.ndarray, colormap: str = "turbo") -> np.ndarray:
    """
    Map a single-channel uint8 image to 3-channel BGR for the VideoWriter.
    colormap: 'none', 'gray', 'turbo', 'viridis', 'plasma', 'magma', 'inferno', 'jet'
    """
    cmap_map = {
        "none": None,
        "gray": None,  # we convert to 3-channel gray
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "magma": cv2.COLORMAP_MAGMA,
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
    }
    c = cmap_map.get(colormap.lower(), None)
    if c is None:
        return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(img_u8, c)


def _resolve_out_path(
    out_path: Optional[str],
    out_dir: Optional[str],
    out_name: str,
    default_dir: str,
) -> str:
    """
    Resolve the final output path.
    Priority:
      1) explicit out_path (returned as-is),
      2) join(out_dir, out_name) if out_dir is given,
      3) join(default_dir, out_name).
    Ensures the parent directory exists.
    """
    if out_path:
        final = out_path
    else:
        base_dir = out_dir if out_dir else default_dir
        final = os.path.join(base_dir, out_name)

    os.makedirs(os.path.dirname(final), exist_ok=True)
    return final


# ---------------------- ffprobe/ffmpeg helpers ----------------------

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def probe_video_codec(filepath: str) -> Optional[str]:
    """Return codec_name of the first video stream, or None if not found/ffprobe missing."""
    try:
        result = _run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", filepath])
        info = json.loads(result.stdout or "{}")
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                return s.get("codec_name")
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def ensure_h264(input_path: str, output_path: Optional[str] = None, crf: int = 23, preset: str = "slow") -> Optional[str]:
    """
    Ensure the given video is H.264 using ffprobe+ffmpeg. Returns the H.264 file path or None on error.
    - If already H.264, returns the original (or renames to output_path if provided).
    - Otherwise, uses ffmpeg (libx264) to convert.
    """
    if output_path is None:
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_h264.mp4" if ext.lower() == ".mp4" else f"{input_path}_h264.mp4"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    codec = probe_video_codec(input_path)
    if codec == "h264":
        if os.path.abspath(input_path) != os.path.abspath(output_path):
            try:
                os.replace(input_path, output_path)
            except OSError:
                # fallback to copy
                import shutil
                shutil.copy2(input_path, output_path)
        return output_path

    # Not H.264 → convert
    try:
        _run([
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-movflags", "+faststart",
            output_path,
        ])
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ---------------------- Main conversion ----------------------

def convert_npy_folder_to_mp4(
    input_dir: str,
    out_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    out_name: str = "depth.mp4",
    fps: float = 30.0,
    pattern: str = "*.npy",
    colormap: str = "turbo",
    percentiles: Optional[Tuple[float, float]] = (1.0, 99.0),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mask_zero: bool = True,
    global_norm: bool = True,
    global_samples: int = 50,
    size: Optional[Tuple[int, int]] = None,  # (width, height)
    rotate: int = 0,
    flip: str = "none",
    # Writer preference: try H.264 directly; fallback to mp4v (converted after)
    prefer_fourcc: str = "avc1",
    fallback_fourcc: str = "mp4v",
    quality: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    # H.264 post-conversion settings:
    h264_crf: int = 23,
    h264_preset: str = "slow",
) -> dict:
    """
    Convert all `.npy` files in a folder to MP4 and ensure the final file is H.264.
    Writes with OpenCV first (trying H.264) and, if needed, converts with ffmpeg.

    Returns
    -------
    dict with meta info:
      {
        'frames_written', 'size', 'vmin', 'vmax',
        'writer_fourcc', 'fps',
        'raw_out_path', 'final_out_path',
        'raw_codec', 'final_codec',
        'h264_ensured': bool
      }
    """
    files = list_npy_files(input_dir, pattern)
    if not files:
        raise FileNotFoundError(f"No .npy files in '{input_dir}' matching pattern '{pattern}'.")

    # Resolve output paths
    final_out = _resolve_out_path(out_path, out_dir, out_name, default_dir=input_dir)
    tmp_out = final_out.replace(".mp4", "_raw.mp4")

    # Load first frame to determine shape
    first = np.load(files[0], allow_pickle=False)
    if first.ndim == 3 and first.shape[-1] == 1:
        first = first[..., 0]
    elif first.ndim != 2:
        raise ValueError(f"Unexpected array shape {first.shape} in {files[0]}. Expected 2D or (H,W,1).")

    first = apply_geometric_ops(first, rotate, flip)

    # Determine target size
    if size is None:
        h, w = first.shape[:2]
        frame_size = (w, h)  # (W, H)
    else:
        frame_size = (int(size[0]), int(size[1]))

    # Global normalization (if requested and not fixed vmin/vmax)
    gvmin, gvmax = vmin, vmax
    if global_norm and (gvmin is None or gvmax is None):
        if global_samples >= len(files):
            sample_paths = files
        else:
            idxs = np.linspace(0, len(files) - 1, num=global_samples, dtype=int)
            sample_paths = [files[i] for i in idxs]

        sample_arrays: List[np.ndarray] = []
        for p in sample_paths:
            a = np.load(p, allow_pickle=False)
            if a.ndim == 3 and a.shape[-1] == 1:
                a = a[..., 0]
            elif a.ndim != 2:
                raise ValueError(f"Unexpected array shape {a.shape} in {p}. Expected 2D or (H,W,1).")
            a = apply_geometric_ops(a, rotate, flip)
            sample_arrays.append(a)

        gvmin, gvmax = compute_norm_bounds(sample_arrays, vmin, vmax, percentiles, mask_zero)

    # Try to open H.264 writer first, else fallback
    def open_writer(fourcc: str) -> cv2.VideoWriter:
        writer = cv2.VideoWriter(
            tmp_out,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            frame_size,
            True,
        )
        if quality is not None:
            try:
                writer.set(cv2.VIDEOWRITER_PROP_QUALITY, int(quality))
            except Exception:
                pass
        return writer

    writer = open_writer(prefer_fourcc)
    used_fourcc = prefer_fourcc
    if not writer.isOpened():
        writer = open_writer(fallback_fourcc)
        used_fourcc = fallback_fourcc

    if not writer.isOpened():
        raise RuntimeError("Could not open any VideoWriter. Try installing an OpenCV build with H.264 support.")

    frames_written = 0
    for i, fp in enumerate(files):
        a = np.load(fp, allow_pickle=False)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        elif a.ndim != 2:
            raise ValueError(f"Unexpected array shape {a.shape} in {fp}. Expected 2D or (H,W,1).")

        a = apply_geometric_ops(a, rotate, flip)

        if not global_norm and (vmin is None or vmax is None):
            vmin_f, vmax_f = compute_norm_bounds([a], None, None, percentiles, mask_zero)
        else:
            vmin_f, vmax_f = gvmin, gvmax

        img_u8 = normalize_to_uint8(a, float(vmin_f), float(vmax_f), mask_zero)
        img_bgr = to_bgr(img_u8, colormap)

        if (img_bgr.shape[1], img_bgr.shape[0]) != frame_size:
            img_bgr = cv2.resize(img_bgr, frame_size, interpolation=cv2.INTER_LINEAR)

        writer.write(img_bgr)
        frames_written += 1

        if progress is not None:
            try:
                progress(i + 1, len(files), fp)
            except Exception:
                pass

    writer.release()

    # Probe raw output codec
    raw_codec = probe_video_codec(tmp_out)

    # Ensure final is H.264 using ffmpeg (like your example)
    ensured_path = ensure_h264(tmp_out, output_path=final_out, crf=h264_crf, preset=h264_preset)
    final_codec = probe_video_codec(ensured_path) if ensured_path else None
    h264_ok = (final_codec == "h264")

    return {
        "frames_written": frames_written,
        "size": frame_size,
        "vmin": float(gvmin) if gvmin is not None else None,
        "vmax": float(gvmax) if gvmax is not None else None,
        "writer_fourcc": used_fourcc,
        "fps": fps,
        "raw_out_path": tmp_out,
        "final_out_path": ensured_path,
        "raw_codec": raw_codec,
        "final_codec": final_codec,
        "h264_ensured": bool(h264_ok and ensured_path),
    }