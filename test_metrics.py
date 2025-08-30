from performance_metrics import Performance_Metrics
import subprocess

monitor = Performance_Metrics(sampling_interval=0.25, enable_pyspy=False)
monitor.start()

cmd = [
    "python", "chunked_demo.py",
    "--root", "/home/TP_tracking/Documents/subs_prep/templates/download/Gymnastik_10_30s/rgbd_data",
    "--vid_name", "Gymnastik_10_30s",
    "--model", "spatracker",
    "--rgbd",
    "--chunk_size", "20",
    "--downsample", "1.0",
#    "--gpu", "0",
    "--grid_size", "20",
    "--fps_vis", "30",
#    "--len_track", "10",
    "--outdir", "./chunked_results"
]

try:
	result = subprocess.run((cmd), check=True, capture_output=True, text=True)
	print("run_demo.sh output: \n", result.stdout)
except subprocess.CalledProcessError as e:
	print("Fehler beim ausführen von run_demo.sh:", e)
	print("Fehlermeldung:\n", e.stderr)

monitor.stop()
monitor.output('./results/20/10')
