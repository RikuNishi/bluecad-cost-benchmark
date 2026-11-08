"""Shared library code for the BlueCAD accuracy-vs-cost study.

Standard-library only, so it imports and tests on machines without a GPU.
"""

from .costlog import CostRecord, append_cost_record, read_cost_records, total_cost_for_volume

from .scoring import parse_prediction, score
from .runwrap import record_run

__all__ = [
    "parse_prediction",
    "score",
    "record_run",
    "CostRecord",
    "append_cost_record",
    "read_cost_records",
    "total_cost_for_volume",
]

__version__ = "0.1.0"
