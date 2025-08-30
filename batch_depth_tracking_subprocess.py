from performance_metrics import Performance_Metrics
import subprocess
from pathlib import Path
import torch
import numpy as np

# Pfad zum Ergebnisordner
base_results_dir = Path("./results")
base_results_dir.mkdir(exist_ok=True)

def run_spatracker(video_path, video_name, output_dir):
    """
    Führt run_demo.sh für ein Video aus.
    """
    output_dir.mkdir(exist_ok=True)

    cmd = [
      #  "./run_demo.sh"
      "python",
      "./chunked_demo.py",
      "--root",str(video_path),
      "--vid_name", "video_name",
      "--model", "spatracker",
      "--rgbd",
      "--chunk_size", "40",
      "--downsample", "1.0",
      "--grid_size", "2",
      "--outdir", str(output_dir),
      "--fps_vis", "30"
    ]

    print(f"\n=== Starte Tracking für {video_name} ===")

    # Performance Monitoring starten
    monitor = Performance_Metrics(sampling_interval=0.25, enable_pyspy=False)
    monitor.start()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print("run_demo.sh output:\n", result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Ausführen von run_demo.sh für {video_name}:")
        print("Fehlermeldung:\n", e.stderr)

    except FileNotFoundError:
        print(f"Fehler: run_demo.sh nicht gefunden für {video_name}")

    # Monitoring stoppen und Ergebnisse speichern
    monitor.stop()
    monitor.output(str(output_dir))
    print(f"Ergebnisse gespeichert in: {output_dir}")


# --------------------- Hauptteil ---------------------
# Liste aller Videos (Ordner mit Tiefendaten)
video_paths = [
    "/home/TP_tracking/Documents/depthCameraResources/videos/Gymnastik_1_5s/Gymnastik_1_5s",
  #  "/home/TP_tracking/Documents/depthCameraResources/videos/Gymnastik_2_5s",
    # ... weitere Video-Ordner
]

for video_path in video_paths:
    video_name = Path(video_path).stem
    output_dir = base_results_dir / video_name
    run_spatracker(video_path, video_name, output_dir)
