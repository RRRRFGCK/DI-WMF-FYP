"""Lightweight NVIDIA board-power sampling for controlled experiments."""

from __future__ import annotations

import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass

try:
    import pynvml
except ImportError:  # pragma: no cover - exercised only on non-NVIDIA systems
    pynvml = None


_NVML_LOCK = threading.Lock()
_NVML_HANDLES = {}


@dataclass(frozen=True)
class EnergyMeasurement:
    available: bool
    sample_count: int
    duration_seconds: float
    idle_watts: float
    mean_watts: float
    gross_joules: float
    net_joules_above_idle: float
    source: str = "NVML board power usage"

    def to_dict(self) -> dict:
        return asdict(self)


def query_nvidia_power(gpu_index: int = 0) -> float:
    if pynvml is not None:
        with _NVML_LOCK:
            handle = _NVML_HANDLES.get(gpu_index)
            if handle is None:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                _NVML_HANDLES[gpu_index] = handle
        return float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        creationflags=creation_flags,
    )
    return float(completed.stdout.strip().splitlines()[0])


def measure_idle_power(
    gpu_index: int = 0, samples: int = 5, interval_seconds: float = 0.2
) -> float:
    readings = []
    for sample_index in range(samples):
        readings.append(query_nvidia_power(gpu_index))
        if sample_index + 1 < samples:
            time.sleep(interval_seconds)
    return float(statistics.median(readings))


class NvidiaPowerSampler:
    """Sample board power in a background thread over a measured code block."""

    def __init__(
        self,
        idle_watts: float,
        gpu_index: int = 0,
        interval_seconds: float = 0.1,
    ):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.idle_watts = float(idle_watts)
        self.gpu_index = gpu_index
        self.interval_seconds = interval_seconds
        self._samples: list[tuple[float, float]] = []
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                watts = query_nvidia_power(self.gpu_index)
                self._samples.append((time.perf_counter(), watts))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop_event.wait(self.interval_seconds)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Power sampler has already been started")
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> EnergyMeasurement:
        if self._thread is None or self._start_time is None:
            raise RuntimeError("Power sampler was not started")
        self._stop_event.set()
        self._thread.join(timeout=max(5.0, 2.0 * self.interval_seconds))
        stop_time = time.perf_counter()
        duration = max(stop_time - self._start_time, 0.0)
        powers = [watts for _, watts in self._samples]
        if not powers:
            return EnergyMeasurement(
                available=False,
                sample_count=0,
                duration_seconds=duration,
                idle_watts=self.idle_watts,
                mean_watts=float("nan"),
                gross_joules=float("nan"),
                net_joules_above_idle=float("nan"),
            )
        mean_watts = float(statistics.fmean(powers))
        mean_net_watts = float(
            statistics.fmean(max(0.0, watts - self.idle_watts) for watts in powers)
        )
        return EnergyMeasurement(
            available=True,
            sample_count=len(powers),
            duration_seconds=duration,
            idle_watts=self.idle_watts,
            mean_watts=mean_watts,
            gross_joules=mean_watts * duration,
            net_joules_above_idle=mean_net_watts * duration,
            source=(
                "NVML board power usage"
                if pynvml is not None
                else "nvidia-smi board power.draw"
            ),
        )

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        return False
