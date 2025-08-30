import json
import os
import subprocess
from typing import Optional

import requests
from tqdm import tqdm


def download_and_ensure_h264(url: str, folder: str, filename: str) -> Optional[str]:
    """
    Downloads a video to `folder/filename` if missing, inspects the codec via ffprobe,
    and converts to H.264 if necessary. Returns the path to the H.264 mp4.
    """
    os.makedirs(folder, exist_ok=True)

    original_filepath = os.path.join(folder, filename)
    h264_filepath = original_filepath.replace(".mp4", "_h264.mp4")

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

    print(f"🔬 Probing codec: {filename}")
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", original_filepath]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        info = json.loads(result.stdout)
        codec_name = next(
            (s.get("codec_name") for s in info.get("streams", []) if s.get("codec_type") == "video"), None
        )
        if not codec_name:
            print("❌ No video stream found.")
            return None
        print(f"✔️ Detected codec: {codec_name}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"❌ ffprobe error. Ensure ffmpeg/ffprobe is installed. Error: {e}")
        return None

    if codec_name == "h264":
        print("✅ Video is already H.264 compatible.")
        if original_filepath != h264_filepath and os.path.exists(original_filepath):
            os.replace(original_filepath, h264_filepath)
        return h264_filepath

    print(f"⚠️ Incompatible codec '{codec_name}'. Converting to H.264 ...")
    convert_cmd = [
        "ffmpeg",
        "-i",
        original_filepath,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "23",
        "-y",
        h264_filepath,
    ]
    try:
        subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        print(f"✨ Conversion successful: {h264_filepath}")
        return h264_filepath
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion error: {e.stderr}")
        return None
