import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_data.py"
spec = importlib.util.spec_from_file_location("audit_data", SCRIPT)
audit_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_data)


class AuditDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.items = self.root / "items"
        self.items.mkdir()
        self.data = self.root / "data"

    def write_file(self, path, content=b"image"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def write_items(self, name, items):
        (self.items / name).write_text(json.dumps({"config": {}, "items": items}), encoding="utf-8")

    def item(self, identifier, images, answer="yes", group="simple"):
        return {"id": identifier, "images": images, "answer": answer, "group": group,
                "prompt_key": "test", "prompt_vars": {}, "meta": {}}

    def test_stats_mapping_and_groups(self):
        render = "/old/data/test/renders/simple/part1/005.png"
        drawing = "/old/data/test/png_out/simple/part1/normal/000.png"
        missing = "/old/data/test/renders/complex/part1/004.png"
        self.write_items("task1_items.json", [self.item("a", [render, drawing]),
                                             self.item("a", [missing], "no", "complex")])
        self.write_items("task2_items.json", [self.item("b", [render], "A")])
        self.write_file(self.data / "renders/part1/005.png")
        for i in range(6):
            self.write_file(self.data / f"drawings/part1/normal/view{i}.png", bytes([i]))
        report = audit_data.audit(self.items, self.data)
        stats = report["item_files"]["task1_items.json"]
        self.assertEqual(stats["item_count"], 2)
        self.assertEqual(stats["answer_distribution"], {"no": 1, "yes": 1})
        self.assertEqual(stats["group_distribution"], {"complex": 1, "simple": 1})
        self.assertEqual(stats["images_per_item_distribution"], {"1": 1, "2": 1})
        self.assertEqual(stats["duplicate_item_ids"], {"a": 2})
        self.assertEqual(stats["unique_model_ids"], ["part1"])
        self.assertEqual((stats["references"]["found"], stats["references"]["missing"]), (2, 1))
        self.assertEqual(report["references"]["found"], 3)
        self.assertEqual(len(report["references"]["drawing_directory_examples"]["part1/normal"]), 6)
        self.assertEqual(report["cross_file"]["inconsistent_groups"]["part1"], ["complex", "simple"])
        self.assertEqual(next(iter(report["cross_file"]["model_id_overlap"].values()))["model_ids"], ["part1"])
        (self.data / "drawings/part1/normal/view0.png").unlink()
        self.assertEqual(audit_data.audit(self.items, self.data)["item_files"]["task1_items.json"]["references"]["found"], 1)

    def test_completeness_and_exact_duplicates(self):
        for i in range(6):
            self.write_file(self.data / f"renders/p/{i:03}.png", f"render{i}".encode())
            for variant in ("normal", "dim"):
                self.write_file(self.data / f"drawings/p/{variant}/{i}.png", f"{variant}{i}".encode())
        self.write_file(self.data / "renders/p/transforms.json", b"{}")
        self.write_file(self.data / "renders/q/000.png", b"render0")
        report = audit_data.audit(self.items, self.data)
        self.assertNotIn("p", report["sampled_parts"]["incomplete_parts"])
        self.assertIn("q", report["sampled_parts"]["incomplete_parts"])
        self.assertEqual(len(report["exact_duplicates"]), 1)
        self.assertEqual(len(report["exact_duplicates"][0]["paths"]), 2)
        self.assertEqual(len(audit_data.audit(self.items, self.data, 1)["sampled_parts"]["model_ids"]), 1)

    def test_cli_with_absent_data_and_item_files(self):
        self.write_items("task1_items.json", [self.item("x", ["/x/renders/simple/p/000.png"])])
        out = self.root / "reports/report.json"
        subprocess.run([sys.executable, str(SCRIPT), "--items-dir", str(self.items), "--data-dir", str(self.data),
                        "--out", str(out)], check=True, capture_output=True, text=True)
        report = json.loads(out.read_text())
        self.assertEqual(report["references"]["missing"], 1)
        self.assertEqual(len(report["missing_item_files"]), 2)
        self.assertEqual(report["sampled_parts"]["model_ids"], [])


if __name__ == "__main__":
    unittest.main()
