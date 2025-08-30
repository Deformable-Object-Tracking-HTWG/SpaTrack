import json
import os
import subprocess
from typing import Optional

import requests
from tqdm import tqdm


def is_h264(path: str) -> bool:
    """
Check if the video at `path` is encoded with H.264.
Returns True if H.264, False otherwise.
    """
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = json.loads(result.stdout)
        codec_name = next(
            (s.get("codec_name") for s in info.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        if codec_name is None:
            print("❌ No video stream found.")
            return False
        return codec_name == "h264"
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"❌ ffprobe error: {e}")
        return False


def convert_to_h264(path: str) -> Optional[str]:
    """
Convert the video at `path` to H.264 format.
Returns the new file path, or None on failure.
    """
    h264_filepath = path.replace(".mp4", "_h264.mp4")
    convert_cmd = [
        "ffmpeg",
        "-i", path,
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "23",
        "-y", h264_filepath,
    ]
    try:
        subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        print(f"✨ Conversion successful: {h264_filepath}")
        return h264_filepath
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion error: {e.stderr}")
        return None


def download_and_ensure_h264(url: str, folder: str, filename: str) -> Optional[str]:
    """
Downloads a video to `folder/filename` if missing,
ensures it is in H.264 format, and returns the path.
    """
    os.makedirs(folder, exist_ok=True)

    original_filepath = os.path.join(folder, filename)

    # download if missing
    if not os.path.exists(original_filepath):
        print(f"📥 Downloading video: {filename}")
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                with open(original_filepath, "wb") as f, tqdm(
                    desc=filename, total=total_size, unit="B", unit_scale=True, unit_divisor=1024
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            print("Video download complete.")
        except requests.RequestException as e:
            print(f"❌ Download error: {e}")
            return None
    else:
        print(f"Video already exists, skipping download: {filename}")

    # check codec and convert if needed
    if is_h264(original_filepath):
        print("✅ Video is already H.264 compatible.")
        h264_filepath = original_filepath.replace(".mp4", "_h264.mp4")
        if original_filepath != h264_filepath and os.path.exists(original_filepath):
            os.replace(original_filepath, h264_filepath)
        return h264_filepath
    else:
        print("⚠️ Video is not H.264. Converting...")
        return convert_to_h264(original_filepath)

def get_frames(path: str) -> Optional[int]:
    """
Return the number of frames in the video at `path`.
Uses ffprobe. Returns None on error.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_streams",
            path,
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = json.loads(result.stdout)

        streams = info.get("streams", [])
        if not streams:
            print("❌ No video stream found.")
            return None

        nb_frames = streams[0].get("nb_read_frames")
        if nb_frames is None:
            print("❌ Frame count not available.")
            return None

        return int(nb_frames)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ ffprobe error: {e}")
        return None

from pathlib import Path
from typing import Optional

def prepare_video_and_eval(local_path: str, eval_target: str) -> tuple[str, str]:
    """
    Ensures a usable H.264 video and matching .npy directory.
    Returns (h264_video, tof_dir).
    Raises RuntimeError if something is wrong.
    """
    mp4s = sorted(Path(local_path).glob("*.mp4"))
    h264_mp4s = [p for p in mp4s if p.stem.endswith("_h264")]
    regular_mp4s = [p for p in mp4s if not p.stem.endswith("_h264")]

    if len(h264_mp4s) == 1 and not regular_mp4s:
        h264_video = str(h264_mp4s[0])
    elif len(regular_mp4s) == 1 and not h264_mp4s:
        candidate = str(regular_mp4s[0])
        if not is_h264(candidate):
            print(f'⚠️ {h264_video} is not H.264 — converting…')
            converted = convert_to_h264(candidate)
            if not converted:
                raise RuntimeError("Conversion failed.")
            h264_video = converted
        else:
            print("✅ Video is H.264.")
            h264_video = candidate.replace(".mp4", "_h264.mp4")
            Path(candidate).rename(h264_video)
    elif len(h264_mp4s) == 1 and len(regular_mp4s) == 1:
        h264_video = str(h264_mp4s[0])
        print(f"ℹ️ Both original and converted exist. Using {h264_video}")
    else:
        raise RuntimeError(f"Expected exactly one usable video in {local_path}, found {len(mp4s)}")

    # check directory
    tof_dir = Path(eval_target)
    if not tof_dir.is_absolute():
        tof_dir = Path(local_path) / eval_target
    if not tof_dir.exists() or not tof_dir.is_dir():
        raise RuntimeError(f"Evaluation directory not found: {tof_dir}")

    frame_count = get_frames(h264_video)
    if frame_count is None:
        raise RuntimeError("Could not determine frame count.")

    npy_count = len(list(tof_dir.glob("*.npy")))
    if npy_count != frame_count:
        raise RuntimeError(
            f"Mismatch: {npy_count} .npy files vs {frame_count} video frames."
        )

    print(f"✅ All good: {npy_count} .npy files match {frame_count} frames.")
    return h264_video, str(tof_dir)
