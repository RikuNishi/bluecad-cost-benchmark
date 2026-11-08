"""Record wall time and, when available, CUDA peak allocation for a run."""

import time
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

from .costlog import CostRecord, append_cost_record


@contextmanager
def record_run(log_path, *, kind, model_id, items, gpu_name=None,
               gpu_hourly_rate=None, currency="JPY", seed=None, notes="",
               batch_size=None, precision=None):
    """Yield a holder whose ``record`` is populated when the block exits."""
    cuda = None
    try:
        import torch
        if torch.cuda.is_available():
            cuda = torch.cuda
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
            if gpu_name is None:
                gpu_name = cuda.get_device_name()
    except ImportError:
        pass
    holder = SimpleNamespace(record=None, notes=notes)
    start = time.perf_counter()
    try:
        yield holder
    finally:
        if cuda is not None:
            cuda.synchronize()
        elapsed = time.perf_counter() - start
        holder.record = CostRecord(
            run_id=str(uuid.uuid4()), kind=kind, model_id=model_id, items=items,
            wall_seconds=elapsed, gpu_name=gpu_name,
            gpu_hours=elapsed / 3600 if cuda is not None else None,
            peak_vram_gb=cuda.max_memory_allocated() / 1e9 if cuda is not None else None,
            gpu_hourly_rate=gpu_hourly_rate, currency=currency, seed=seed, notes=holder.notes,
            batch_size=batch_size, precision=precision,
        )
        append_cost_record(log_path, holder.record)
