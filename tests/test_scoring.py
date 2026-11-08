import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bluecad_cost import parse_prediction, read_cost_records, record_run, score


class ScoringTest(unittest.TestCase):
    def test_task1_metrics(self):
        items = [dict(id=i, answer=answer, group="simple" if i < 4 else "complex",
                      meta={"tier": "easy"})
                 for i, answer in enumerate(["yes", "no", "no", "yes", "no", "yes"])]
        predictions = [{"id": 0, "raw": "Yes, matching"}, {"id": 1, "pred": "yes"},
                       {"id": 2, "pred": "no"}, {"id": 3, "raw": "uncertain"},
                       {"id": 4, "error": "timeout"}]
        result = score(items, predictions, 1)
        self.assertEqual((result["n_items"], result["n_answered"]), (6, 3))
        self.assertEqual((result["n_error"], result["n_unparseable"], result["n_missing_predictions"]), (1, 1, 1))
        self.assertAlmostEqual(result["accuracy_all"], 2 / 6)
        self.assertAlmostEqual(result["accuracy_answered"], 2 / 3)
        self.assertAlmostEqual(result["answer_rate"], 3 / 6)
        # yes: TP=1 FP=1 FN=0 => 2/3; no: TP=1 FP=0 FN=1 => 2/3.
        self.assertAlmostEqual(result["macro_f1_answered"], (2 / 3 + 2 / 3) / 2)
        self.assertEqual(result["confusion_answered"], {"yes": {"yes": 1}, "no": {"yes": 1, "no": 1}})
        self.assertEqual(result["evaluated_ids"], list(range(6)))
        self.assertEqual(result["answered_ids"], [0, 1, 2])
        self.assertEqual(result["by_group"]["complex"]["n_answered"], 0)
        self.assertIsNone(result["by_group"]["complex"]["accuracy_answered"])
        self.assertEqual(result["by_tier"]["easy"]["n_items"], 6)

    def test_task2_parsing(self):
        self.assertEqual(parse_prediction("B then ANSWER: C", 2), "C")
        self.assertEqual(parse_prediction("Candidates A or B; final D", 2), "D")
        self.assertIsNone(parse_prediction("A then Z", 2))
        self.assertIsNone(parse_prediction("nothing here", 2))

    def test_cot(self):
        self.assertEqual(parse_prediction("No earlier\nAnswer: YES, it matches.\n", 1, True), "yes")
        self.assertEqual(parse_prediction("Reason\nno.", 1, True), "no")
        self.assertIsNone(parse_prediction("yes\nTherefore the answer is no", 1, True))
        self.assertEqual(parse_prediction("yes then no", 1), "yes")

    def test_duplicates(self):
        items = [{"id": "x", "answer": "yes", "group": "simple"}]
        result = score(items, [{"id": "x", "pred": "no"}, {"id": "x", "pred": "yes"},
                               {"id": "x", "pred": "yes"}], 1)
        self.assertEqual(result["duplicate_prediction_ids"], ["x"])
        self.assertEqual(result["accuracy_all"], 0)

    def test_end_to_end_with_cost_record_and_cli(self):
        items = [{"id": "a", "answer": "C", "group": "simple", "prompt_key": "task3_retrieval"},
                 {"id": "b", "answer": "A", "group": "complex", "prompt_key": "task3_retrieval"}]
        predictions = [{"id": "a", "raw": "ANSWER: C"}, {"id": "b", "pred": "ERROR"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with record_run(root / "cost.jsonl", kind="infer", model_id="dummy", items=len(items)) as run:
                metrics = score(items, predictions, 2)
            records = read_cost_records(root / "cost.jsonl")
            self.assertEqual(metrics["accuracy_all"], 0.5)
            self.assertEqual(metrics["accuracy_answered"], 1.0)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].items, metrics["n_items"])
            self.assertEqual(run.record.run_id, records[0].run_id)
            (root / "task2_items.json").write_text(json.dumps({"config": {}, "items": items}))
            (root / "predictions.jsonl").write_text("\n".join(map(json.dumps, predictions)))
            script = Path(__file__).resolve().parents[1] / "scripts" / "score.py"
            subprocess.run([sys.executable, str(script), "--items-file", str(root / "task2_items.json"),
                            "--predictions", str(root / "predictions.jsonl"), "--out", str(root / "metrics.json")],
                           check=True, capture_output=True, text=True)
            self.assertEqual(json.loads((root / "metrics.json").read_text()), metrics)


if __name__ == "__main__":
    unittest.main()
