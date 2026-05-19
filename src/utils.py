import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import mindspore as ms
import psutil


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.seconds = time.perf_counter() - self.start


class MemoryTracker:
    def __enter__(self):
        self.process = psutil.Process()
        self.peak_mb = self.current_mb()
        return self

    def current_mb(self) -> float:
        return self.process.memory_info().rss / 1024 / 1024

    def update(self) -> float:
        value = self.current_mb()
        self.peak_mb = max(self.peak_mb, value)
        return value

    def __exit__(self, exc_type, exc, tb):
        self.update()
