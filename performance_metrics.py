import psutil
import pynvml
import matplotlib.pyplot as plt
import threading
import subprocess
import time
import os
from datetime import datetime
import json

class Performance_Metrics:
    def __init__(self, sampling_interval=1.0, enable_pyspy=True, pyspy_duration=30, pyspy_interval=300):
        self.sampling_interval = sampling_interval
        self.enable_pyspy = enable_pyspy
        self.pyspy_duration = pyspy_duration
        self.pyspy_interval = pyspy_interval
        self.monitoring = False
        self.start_time = None
        self.end_time = None

        # Data storage
        self.timestamps = []
        self.cpu_percent = []
        self.memory_percent = []
        self.memory_used_gb = []
        self.disk_read_mb = []
        self.disk_write_mb = []
        self.gpu_util = []
        self.gpu_memory_used = []
        self.gpu_memory_total = []

        # Monitoring thread and py-spy process
        self.monitor_thread = None
        self.pyspy_thread = None
        self.pyspy_processes = []
        self.pyspy_output_files = []

        # Initialize GPU monitoring
        self.gpu_available = self._init_gpu()

        # Initial disk I/O counters
        self.initial_disk_io = psutil.disk_io_counters()

    def _init_gpu(self):
        try:
            pynvml.nvmlInit()
            return True
        except:
            print("Warning: NVIDIA GPU not available or pynvml not installed")
            return False

    def _monitor_loop(self):
        while self.monitoring:
            timestamp = time.time() - self.start_time
            self.timestamps.append(timestamp)

            # CPU and Memory
            self.cpu_percent.append(psutil.cpu_percent())
            memory = psutil.virtual_memory()
            self.memory_percent.append(memory.percent)
            self.memory_used_gb.append(memory.used / (1024**3))

            # Disk I/O
            current_disk_io = psutil.disk_io_counters()
            if current_disk_io and self.initial_disk_io:
                read_mb = (current_disk_io.read_bytes - self.initial_disk_io.read_bytes) / (1024**2)
                write_mb = (current_disk_io.write_bytes - self.initial_disk_io.write_bytes) / (1024**2)
                self.disk_read_mb.append(read_mb)
                self.disk_write_mb.append(write_mb)
            else:
                self.disk_read_mb.append(0)
                self.disk_write_mb.append(0)

            # GPU metrics
            if self.gpu_available:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    self.gpu_util.append(gpu_util.gpu)
                    self.gpu_memory_used.append(memory_info.used / (1024**3))
                    self.gpu_memory_total.append(memory_info.total / (1024**3))
                except:
                    self.gpu_util.append(0)
                    self.gpu_memory_used.append(0)
                    self.gpu_memory_total.append(0)
            else:
                self.gpu_util.append(0)
                self.gpu_memory_used.append(0)
                self.gpu_memory_total.append(0)

            time.sleep(self.sampling_interval)

    def _pyspy_periodic_sampling(self):
        """Run py-spy recordings periodically"""
        sample_count = 0

        while self.monitoring and self.enable_pyspy:
            try:
                sample_count += 1
                output_file = f"profile_{int(self.start_time)}_sample_{sample_count}.svg"

                print(f"Starting py-spy sample {sample_count} (PID: {os.getpid()}) for {self.pyspy_duration}s")

                process = subprocess.Popen([
                    'py-spy', 'record',
                    '-o', output_file,
                    '-p', str(os.getpid()),
                    '-d', str(self.pyspy_duration)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                self.pyspy_processes.append(process)
                self.pyspy_output_files.append(output_file)

                # Wait for this recording to finish
                stdout, stderr = process.communicate(timeout=self.pyspy_duration + 10)

                if process.returncode == 0:
                    print(f"py-spy sample {sample_count} completed successfully")
                    if os.path.exists(output_file):
                        print(f"Created: {output_file}")
                    else:
                        print(f"Warning: Expected file not created: {output_file}")
                else:
                    print(f"py-spy sample {sample_count} failed with code: {process.returncode}")
                    if stderr:
                        print(f"py-spy stderr: {stderr}")

                # Wait for next interval (but break if monitoring stopped)
                for _ in range(int(self.pyspy_interval)):
                    if not self.monitoring:
                        break
                    time.sleep(1)

            except Exception as e:
                print(f"py-spy sampling error: {e}")
                time.sleep(self.pyspy_interval)

    def start(self):
        if self.monitoring:
            print("Monitoring already started")
            return

        print("Starting performance monitoring...")
        self.start_time = time.time()
        self.monitoring = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # Start py-spy periodic sampling if enabled
        if self.enable_pyspy:
            try:
                # Test if py-spy is available
                subprocess.run(['py-spy', '--help'], capture_output=True, check=True)
                print(f"Starting py-spy periodic sampling ({self.pyspy_duration}s recordings every {self.pyspy_interval}s)")

                self.pyspy_thread = threading.Thread(target=self._pyspy_periodic_sampling)
                self.pyspy_thread.daemon = True
                self.pyspy_thread.start()

            except (FileNotFoundError, subprocess.CalledProcessError):
                print("Warning: py-spy not found or not working. Install with: pip install py-spy")
                print("You may also need to run with elevated permissions (sudo)")
                self.enable_pyspy = False

    def stop(self):
        if not self.monitoring:
            print("Monitoring not started")
            return

        print("Stopping performance monitoring...")
        self.end_time = time.time()
        self.monitoring = False

        # Wait for monitoring thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

        # Wait for py-spy thread to finish and clean up processes
        if self.enable_pyspy and self.pyspy_thread:
            print("Stopping py-spy sampling...")
            self.pyspy_thread.join(timeout=5)

            # Terminate any remaining py-spy processes
            for process in self.pyspy_processes:
                if process.poll() is None:  # Still running
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()

        print("Monitoring stopped")

    def output(self, output_dir):
        if self.monitoring:
            print("Stop monitoring before generating output")
            return

        if not self.timestamps:
            print("No data collected")
            return

        os.makedirs(output_dir, exist_ok=True)

        print(f"Generating output in: {output_dir}")

        # Set style for all plots
        plt.style.use('default')
        colors = {
            'cpu': '#e74c3c',      # Red
            'memory': '#3498db',    # Blue
            'disk_read': '#2ecc71', # Green
            'disk_write': '#f39c12', # Orange
            'gpu_util': '#9b59b6',  # Purple
            'gpu_memory': '#1abc9c' # Teal
        }

        # CPU, Memory, and Disk I/O Combined Graph
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        ax1.plot(self.timestamps, self.cpu_percent, color=colors['cpu'], linewidth=2)
        ax1.set_title('CPU Usage Over Time', fontweight='bold')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.timestamps, self.memory_percent, color=colors['memory'], linewidth=2)
        ax2.set_title('Memory Usage Over Time', fontweight='bold')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.grid(True, alpha=0.3)

        ax3.plot(self.timestamps, self.disk_read_mb, color=colors['disk_read'],
                 linewidth=2, label='Read')
        ax3.plot(self.timestamps, self.disk_write_mb, color=colors['disk_write'],
                 linewidth=2, label='Write')
        ax3.set_title('Cumulative Disk I/O Over Time', fontweight='bold')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Data Transferred (MB)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'system_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Memory Usage Absolute (separate graph for GB values)
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.memory_used_gb, color=colors['memory'], linewidth=2)
        plt.title('Memory Usage Absolute Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Used (GB)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_absolute.png'), dpi=300, bbox_inches='tight')
        plt.close()





        # GPU Usage Graph (if available)
        if self.gpu_available and any(self.gpu_util):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            ax1.plot(self.timestamps, self.gpu_util, color=colors['gpu_util'], linewidth=2)
            ax1.set_title('GPU Utilization', fontweight='bold')
            ax1.set_ylabel('GPU Utilization (%)')
            ax1.grid(True, alpha=0.3)

            ax2.plot(self.timestamps, self.gpu_memory_used, color=colors['gpu_memory'], linewidth=2)
            if self.gpu_memory_total and self.gpu_memory_total[0] > 0:
                ax2.axhline(y=self.gpu_memory_total[0], color='red', linestyle='--',
                            alpha=0.7, label=f'Total Memory ({self.gpu_memory_total[0]:.1f} GB)')
                ax2.legend()
            ax2.set_title('GPU Memory Usage', fontweight='bold')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('GPU Memory Used (GB)')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gpu_usage.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Move py-spy output files
        if self.enable_pyspy and self.pyspy_output_files:
            import shutil
            pyspy_dir = os.path.join(output_dir, 'pyspy_profiles')
            os.makedirs(pyspy_dir, exist_ok=True)

            moved_files = []
            for i, output_file in enumerate(self.pyspy_output_files):
                if os.path.exists(output_file):
                    target_name = f"profile_sample_{i+1}.svg"
                    target_path = os.path.join(pyspy_dir, target_name)
                    shutil.move(output_file, target_path)
                    moved_files.append(target_name)
                    print(f"py-spy profile moved: {target_name}")

            if moved_files:
                print(f"py-spy profiles saved in: {pyspy_dir}")
                print(f"Generated {len(moved_files)} profile samples")
            else:
                print("No py-spy profile files were created")

        elif self.enable_pyspy:
            print("py-spy was enabled but no output files were generated")
        else:
            print("py-spy was disabled")

        # Generate summary text
        execution_time = self.end_time - self.start_time
        summary = self._generate_summary(execution_time)

        with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
            f.write(summary)

        print(f"Output generated successfully in: {output_dir}")

    def _generate_summary(self, execution_time):
        summary = f"""Performance Metrics Summary
            {'='*50}

            Execution Time: {execution_time:.2f} seconds
            Sampling Interval: {self.sampling_interval} seconds
            Total Samples: {len(self.timestamps)}

CPU Metrics:
            Average CPU Usage: {sum(self.cpu_percent)/len(self.cpu_percent):.1f}%
            Peak CPU Usage: {max(self.cpu_percent):.1f}%
            Min CPU Usage: {min(self.cpu_percent):.1f}%

Memory Metrics:
            Average Memory Usage: {sum(self.memory_percent)/len(self.memory_percent):.1f}%
            Peak Memory Usage: {max(self.memory_percent):.1f}%
            Peak Memory Used: {max(self.memory_used_gb):.2f} GB

Disk I/O Metrics:
            Total Data Read: {max(self.disk_read_mb):.2f} MB
            Total Data Written: {max(self.disk_write_mb):.2f} MB
            """

        if self.gpu_available and any(self.gpu_util):
            avg_gpu = sum(self.gpu_util)/len(self.gpu_util)
            peak_gpu = max(self.gpu_util)
            peak_gpu_mem = max(self.gpu_memory_used)

            summary += f"""
GPU Metrics:
                Average GPU Utilization: {avg_gpu:.1f}%
                Peak GPU Utilization: {peak_gpu:.1f}%
                Peak GPU Memory Used: {peak_gpu_mem:.2f} GB
                """
        else:
            summary += "\nGPU Metrics: Not available\n"

        return summary