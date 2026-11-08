import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bluecad_cost import (  # noqa: E402
    CostRecord,
    append_cost_record,
    read_cost_records,
    total_cost_for_volume,
)


class CostLogTest(unittest.TestCase):
    def test_roundtrip_and_cost(self):
        rec = CostRecord(
            run_id="test-1",
            kind="infer",
            model_id="dummy",
            items=1000,
            wall_seconds=3600.0,
            gpu_name="NVIDIA L4 24GB",
            gpu_hourly_rate=100.0,
            currency="JPY",
        )
        self.assertEqual(rec.cost(), 100.0)  # 1 GPU-hour * 100/h
        self.assertAlmostEqual(rec.per_item_seconds(), 3.6)
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "cost_log.jsonl"
            append_cost_record(log, rec)
            loaded = read_cost_records(log)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].run_id, "test-1")
        self.assertEqual(loaded[0].cost(), 100.0)

    def test_cost_unknown_without_rate(self):
        rec = CostRecord(run_id="t", kind="train", model_id="m", items=1, wall_seconds=1.0)
        self.assertIsNone(rec.cost())

    def test_total_cost_for_volume(self):
        # upfront 1000 + 0.5 per item
        self.assertEqual(total_cost_for_volume(1000.0, 0.5, 0), 1000.0)
        self.assertEqual(total_cost_for_volume(1000.0, 0.5, 2000), 2000.0)
        with self.assertRaises(ValueError):
            total_cost_for_volume(0.0, 0.0, -1)


if __name__ == "__main__":
    unittest.main()
