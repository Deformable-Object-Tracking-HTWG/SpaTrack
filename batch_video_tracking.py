from performance_metrics import Performance_Metrics
import subprocess
import os
from pathlib import Path

# Liste der Videos, die getrackt werden sollen
video_paths = [
#    "/home/TP_tracking/Documents/depthCameraResources/videos/",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/Gymnastik_3_5s.mp4"

#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_1_5s360_20.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_1_5s360_50.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_1_5s360_100.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_1_5s720_20.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_1_5s720_50.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_1_5s720_100.mp4",
    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_7_10s360_20.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_7_10s360_50.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_7_10s360_100.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_7_10s720_20.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_7_10s720_50.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_7_10s720_100.mp4",
    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_10_30s360_20.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_10_30s360_50.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_10_30s360_100.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_10_30s720_20.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_10_30s720_50.mp4",
#    "/home/TP_tracking/Documents/depthCameraResources/videos/compare/Gymnastik_10_30s720_100.mp4",

]

# Basisordner für die Ergebnisse
base_results_dir = Path("./results")
base_results_dir.mkdir(exist_ok=True)

# Funktion, die ein einzelnes Video tracked
def run_video_tracking(video_path):
    video_name = Path(video_path).stem
    output_dir = base_results_dir / video_name
    output_dir.mkdir(exist_ok=True)

    print(f"\n=== Starte Tracking für {video_name} ===")

    monitor = Performance_Metrics(sampling_interval=0.25, enable_pyspy=False)
    monitor.start()

    try:
        # run_demo.sh ausführen
        result = subprocess.run(
            [
             "python",
             "./chunked_demo.py",
             "--root", str(Path(video_path).parent),
             "--vid_name", video_name,
             "--model", "spatracker",
             "--chunk_size", "40",
             "--downsample", "1.0",
             "--grid_size", "20",
             "--outdir", output_dir,
             "--fps_vis", "30"
            ],  # oder "bash ./run_demo.sh", shell=True
            check=True,
            capture_output=True,
            text=True
        )
        print("run_demo.sh output:\n", result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Ausführen von run_demo.sh für {video_name}:", e)
        print("Fehlermeldung:\n", e.stderr)
    except FileNotFoundError:
        print(f"Fehler: run_demo.sh nicht gefunden für {video_name}")

    # Monitoring stoppen und Ergebnisse speichern
    monitor.stop()
    monitor.output(str(output_dir))
    print(f"Ergebnisse gespeichert in: {output_dir}")

# Alle Videos nacheinander verarbeiten
for video_path in video_paths:
    video_name = Path(video_path).stem
    output_dir = base_results_dir / video_name
    run_video_tracking(video_path)
