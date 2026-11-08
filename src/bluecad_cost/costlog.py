"""Cost logging for experiment runs.

Every training / feature-extraction / inference run appends one JSON object
to a JSONL cost log. The fields follow the "how to measure cost" table in
RESEARCH_PLAN.ja.md. Money conversion is kept explicit:

    cost = (seconds / 3600) * gpu_hourly_rate        (per run)
    total cost for repeated use =
        training cost + index-build cost + n_items * per-item cost

Only the Python standard library is used here, on purpose.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CostRecord:
    run_id: str
    kind: str  # "train" | "extract" | "index" | "infer" | "api"
    model_id: str
    items: int  # questions / pairs / images processed in this run
    wall_seconds: float
    gpu_name: str | None = None
    gpu_hours: float | None = None  # defaults to wall_seconds/3600 for 1 GPU
    peak_vram_gb: float | None = None
    batch_size: int | None = None
    precision: str | None = None  # "bf16" | "fp32" | "fp16" | "int8" | ...
    gpu_hourly_rate: float | None = None  # in `currency`, from the actual bill
    currency: str = "JPY"
    seed: int | None = None
    notes: str = ""
    timestamp: float = field(default_factory=time.time)

    def cost(self) -> float | None:
        """Money cost of this run, or None if the rate is unknown."""
        if self.gpu_hourly_rate is None:
            return None
        hours = self.gpu_hours if self.gpu_hours is not None else self.wall_seconds / 3600.0
        return hours * self.gpu_hourly_rate

    def per_item_seconds(self) -> float:
        return self.wall_seconds / self.items if self.items else 0.0


def append_cost_record(log_path: str | Path, record: CostRecord) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_cost_records(log_path: str | Path) -> list[CostRecord]:
    path = Path(log_path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(CostRecord(**json.loads(line)))
    return records


def total_cost_for_volume(
    upfront_cost: float, per_item_cost: float, n_items: int
) -> float:
    """Total cost after processing n_items, including one-off upfront cost.

    This is the formula the plan uses to find the volume at which a model
    with training/indexing cost becomes cheaper overall:
        total = training + index-build + n_items * per-item cost
    """
    if n_items < 0:
        raise ValueError("n_items must be >= 0")
    return upfront_cost + per_item_cost * n_items
