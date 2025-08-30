from __future__ import annotations
import os
import subprocess
from pathlib import Path
from typing import Optional


def run_spatial_tracker(
    video_path: str,
    mask_path: str,
    chunk_size: int,
    output_folder: str,
    tracker_script: str = "chunked_demo.py",
) -> Optional[str]:
    """
    Run the SpatialTracker script and return the produced tracking .npy path.
    """
    print("\n🚀 Running SpatialTracker ...")

    abs_video = os.path.abspath(video_path)
    abs_mask = os.path.abspath(mask_path)
    abs_out = os.path.abspath(output_folder)
    os.makedirs(abs_out, exist_ok=True)

    cmd = [
        "python", tracker_script,
        "--vid_name", abs_video,
        "--mask_name", abs_mask,
        "--outdir", abs_out,
        "--chunk_size", str(chunk_size),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line.rstrip())
        rc = proc.wait()
        if rc != 0:
            print(f"❌ SpatialTracker exited with code {rc}.")
            return None

        print("✔️ SpatialTracker completed.")

        # Find newest subdir and expected tracking file name
        subfolders = [p.path for p in os.scandir(abs_out) if p.is_dir()]
        if not subfolders:
            print("❌ No output subfolder found.")
            return None

        latest = max(subfolders, key=os.path.getmtime)
        video_stem = Path(abs_video).stem
        tracking_file = os.path.join(latest, f"{video_stem}_tracking_data.npy")

        if os.path.exists(tracking_file):
            print(f"✔️ Tracking data: {tracking_file}")
            return tracking_file

        print(f"❌ Tracking file not found: {tracking_file}")
        return None

    except FileNotFoundError as e:
        print(f"❌ Tracker not found: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error while running tracker: {e}")
        return None
