"""Software GPU guard (ADR-017): keep the workstation inside a power and temperature envelope.

A VRAM cap does not cap power: prefill alone drives an RTX 3060 Ti to ~180 W. Before every live model call the guard
waits while the GPU is hotter than LEXICON_GPU_TEMP_MAX_C (default 75 C) or while its rolling mean power exceeds
LEXICON_GPU_POWER_CAP_W (default 160 W, 80% of the 200 W board limit), and it enforces a duty cycle: after a call
that took t seconds, new calls are held back for (1/duty - 1) * t seconds (default duty 0.6). A hardware power limit
(`nvidia-smi -pl 160`, admin) is stronger; this guard works without admin rights.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections import deque


class GpuGuard:
    def __init__(
        self, power_cap_w: float | None = None, temp_max_c: float | None = None, duty: float | None = None
    ) -> None:
        self.power_cap = power_cap_w or float(os.environ.get("LEXICON_GPU_POWER_CAP_W", "160"))
        self.temp_max = temp_max_c or float(os.environ.get("LEXICON_GPU_TEMP_MAX_C", "75"))
        self.duty = duty or float(os.environ.get("LEXICON_GPU_DUTY", "0.6"))
        self.exe = shutil.which("nvidia-smi")
        self.power: deque[float] = deque(maxlen=10)
        self.temp = 0.0
        self.waited_s = 0.0
        self._not_before = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if self.exe and os.environ.get("LEXICON_GPU_GUARD", "1") == "1":
            threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        str(self.exe),
                        "--query-gpu=power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                ).stdout.split(",")
                self.power.append(float(out[0]))
                self.temp = float(out[1])
            except (subprocess.SubprocessError, OSError, ValueError, IndexError):
                pass
            self._stop.wait(1.0)

    @property
    def mean_power(self) -> float:
        return sum(self.power) / len(self.power) if self.power else 0.0

    def hot(self) -> bool:
        return self.temp > self.temp_max or self.mean_power > self.power_cap

    def before_call(self) -> None:
        t0 = time.time()
        while True:
            with self._lock:
                wait = self._not_before - time.time()
            if wait <= 0 and not self.hot():
                break
            time.sleep(max(min(wait, 2.0), 0.5))
        self.waited_s += time.time() - t0

    def after_call(self, seconds: float) -> None:
        with self._lock:
            self._not_before = max(self._not_before, time.time() + (1 / self.duty - 1) * seconds)

    def close(self) -> None:
        self._stop.set()


_GUARD: GpuGuard | None = None


def guard() -> GpuGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = GpuGuard()
    return _GUARD
