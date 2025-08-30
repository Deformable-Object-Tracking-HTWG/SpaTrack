from performance_metrics import Performance_Metrics
import subprocess
from pathlib import Path

# Ergebnisordner
base_results_dir = Path("./results")
base_results_dir.mkdir(exist_ok=True)


# Parameter für den Befehl
host = "cloud.lukas-reinke.de"
public_token = "adcQibe6C3DmTxL"
slug = "Gymnastik_10_30s360"
ref_images = "/home/TP_tracking/Documents/SpaTrack/eval_tof_utils/reference_images/"
sam_checkpoint = "/home/TP_tracking/Documents/segment_anything/sam_vit_h_4b8939.pth"
sam_model_type = "vit_h"
device = "cuda"
base_dir = "evaluation"

# Befehl als Liste für subprocess

output_dir = base_results_dir / slug
output_dir.mkdir(exist_ok=True)

cmd = [
    "python", "eval_pipeline.py",
    "--host", host,
    "--public-token", public_token,
    "--slug", slug,
    "--ref-images", ref_images,
    "--sam-checkpoint", sam_checkpoint,
    "--sam-model-type", sam_model_type,
    "--device", device,
    "--base-dir", base_dir,
]
print(f"\n=== Starte Evaluation für {slug} ===")


# Performance Monitoring starten
monitor = Performance_Metrics(sampling_interval=0.25, enable_pyspy=False)
monitor.start()

try:
    # Subprocess ausführen
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("eval_pipeline.py output:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Fehler beim Ausführen von eval_pipeline.py für {slug}: {e}")
    print("Fehlermeldung:\n", e.stderr)

# Monitoring stoppen und Ergebnisse speichern
monitor.stop()
monitor.output(str(output_dir))
print(f"Ergebnisse gespeichert in: {output_dir}")
