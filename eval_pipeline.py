
"""
Evaluation pipeline.

Flow:
  1) Build URLs from host/token/slug and prepare a timestamped run directory.
  2) Download video and ensure H.264.
  3) Create a first-frame object mask (SAM + CLIP with gym ball bias).
  4) Run SpatialTracker.
  5) Download TOF .npy files from the public Nextcloud share.
  6) Analyze tracker vs. TOF (writes artifacts to analysis/).
  7) Create TOF overlay video (uses analysis products if needed).

Every run gets its own folder: <base-dir>/<YYYY-MM-DD_HH-MM-SS>_<slug>/
"""

from __future__ import annotations
import argparse
import datetime as dt
import os
import sys
from pathlib import Path
import gc

from eval_tof_utils.video_io import download_and_ensure_h264
from eval_tof_utils.masking import create_semantic_mask
from eval_tof_utils.tracker import run_spatial_tracker
from eval_tof_utils.tof_download import download_all_npy_from_nextcloud_folder
from eval_tof_utils.analysis import analyze_results
from eval_tof_utils.tof_overlay import create_tof_overlay_video, probe_video_fps


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end evaluation pipeline for RGB video + TOF data."
    )

    # Core inputs
    p.add_argument("--host", required=True, help="Nextcloud host, e.g. cloud.example.org")
    p.add_argument("--public-token", required=True, help="Public share token")
    p.add_argument("--slug", required=True,
                   help="Clip slug (the only changing part), e.g. Gymnastik_6_5s")

    # Optional password for the public share
    p.add_argument("--tof-share-password", default="", help="Password if the share is protected")

    # Environment
    p.add_argument("--base-dir", default="evaluation", help="Base directory for all runs")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Computation device")

    # Masking
    p.add_argument("--ref-images", required=True,
                   help="Directory with reference images used for CLIP similarity")
    p.add_argument("--sam-checkpoint", required=True, help="Path to SAM checkpoint (.pth)")
    p.add_argument("--sam-model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--center-priority", action="store_true",
                   help="Prefer segment closest to center (overrides CLIP)")

    # Tracker
    p.add_argument("--tracker-script", default="chunked_demo.py", help="SpatialTracker script path")
    p.add_argument("--chunk-size", type=int, default=30, help="Frames per chunk for tracker")

    return p.parse_args()


# ----------------------------- Helpers -----------------------------

def build_urls(host: str, token: str, slug: str) -> tuple[str, str, str]:
    """
    Build:
      - direct WebDAV video URL
      - public share folder URL (for .npy)
      - local filename
    """

    video_filename = f"{slug}.mp4"
    base_path = f"/TOF_camera/{slug}/rgbd_data/{slug}"

    video_url = f"https://{host}/public.php/dav/files/{token}{base_path}.mp4"
    # The ?path= needs URL-encoding for slashes (%2F), but downstream handles that.
    tof_share_url = f"https://{host}/s/{token}?path=%2FTOF_camera%2F{slug}%2Frgbd_data%2F{slug}"
    return video_url, tof_share_url, video_filename


def make_run_dirs(base_dir: Path, slug: str) -> dict[str, Path]:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / f"{stamp}_{slug}"

    sub = {
        "run": run_dir,
        "video": run_dir / "video",
        "mask": run_dir / "mask",
        "tracker": run_dir / "tracker",
        "tof": run_dir / "tof",
        "analysis": run_dir / "analysis",
        "overlay": run_dir / "overlay",
    }
    for p in sub.values():
        p.mkdir(parents=True, exist_ok=True)
    return sub


def fail(msg: str) -> None:
    print(f"❌ {msg}")
    sys.exit(1)


def cleanup_gpu(device: str) -> None:
    """
    Aggressively release GPU memory (safe to call without torch/cuda).
    """
    try:
        import torch  # type: ignore
        if device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    finally:
        gc.collect()


# ----------------------------- Main -----------------------------

def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    video_url, tof_share_url, video_filename = build_urls(
        host=args.host, token=args.public_token, slug=args.slug
    )

    dirs = make_run_dirs(base_dir, args.slug)

    # 1) Video (download + ensure H.264)
    h264_video = download_and_ensure_h264(
        url=video_url, folder=str(dirs["video"]), filename=video_filename
    )
    if not h264_video:
        fail("Video step failed")

    # 2) Mask (SAM + CLIP)
    mask_path = None
    try:
        mask_path = create_semantic_mask(
            video_path=h264_video,
            output_folder=str(dirs["mask"]),
            reference_images_dir=args.ref_images,
            sam_checkpoint=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.device,
            use_center_priority=args.center_priority,
            target_class="exercise ball",
            disallow_classes=["person", "shirt", "man", "t-shirt"],
            enforce_circularity=True,
        )
    finally:
        print("🧹 Releasing GPU memory after mask generation ...")
        cleanup_gpu(args.device)

    if not mask_path:
        fail("Mask creation failed")


    # 3) Tracker
    tracking_file = run_spatial_tracker(
        video_path=h264_video,
        mask_path=mask_path,
        chunk_size=args.chunk_size,
        output_folder=str(dirs["tracker"]),
        tracker_script=args.tracker_script,
    )
    if not tracking_file:
        fail("SpatialTracker failed")

    # 4) TOF download
    print("\n📥 Downloading TOF .npy files ...")
    downloaded = download_all_npy_from_nextcloud_folder(
        share_folder_url=tof_share_url,
        target_dir=str(dirs["tof"]),
        password=args.tof_share_password,
    )
    if downloaded <= 0:
        fail("No TOF .npy files downloaded")

    # 5) Analysis
    print("\n📊 Running analysis ...")
    analyze_results(
        tracking_file=tracking_file,
        tof_data_folder=str(dirs["tof"]),
        output_dir=str(dirs["analysis"]),
    )

    # 6) Overlay TOF
    print("\n🎥 Building TOF overlay video ...")
    overlay_fps = probe_video_fps(h264_video) or 30.0
    overlay_out = dirs["overlay"] / f"{Path(video_filename).stem}_tof_overlay.mp4"
    info = create_tof_overlay_video(
        tracking_file=tracking_file,
        tof_data_folder=str(dirs["tof"]),
        out_path=str(overlay_out),
        fps=overlay_fps,
        mask_path=mask_path,
        colormap="turbo",
        percentiles=(1, 99),
        mask_zero=True,
        global_norm=True,
        point_radius=4,
        point_thickness=-1,
        draw_ids=False,
        tail_length=10,
        h264_crf=23,
        h264_preset="slow",
    )
    print("✔️ Overlay:", info.get("final_out_path"))

    print(f"\n✅ Done. Run folder: {dirs['run']}")


if __name__ == "__main__":
    main()
