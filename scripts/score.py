#!/usr/bin/env python3
"""Score JSONL predictions against an official-format items file."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bluecad_cost.scoring import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-file", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task", choices=("1", "2"))
    parser.add_argument("--cot", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.items_file.read_text(encoding="utf-8"))
    items = data["items"]
    task = args.task
    if task is None:
        name = args.items_file.name.lower()
        if name.startswith("task1"):
            task = "1"
        elif name.startswith("task2"):
            task = "2"
        elif items and all(item.get("prompt_key") == "task1_verification" for item in items):
            task = "1"
        else:
            parser.error("cannot infer task; specify --task")
    with args.predictions.open(encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    metrics = score(items, predictions, task, cot=args.cot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"items={metrics['n_items']} answered={metrics['n_answered']} "
          f"answer_rate={metrics['answer_rate']:.4f} accuracy_all={metrics['accuracy_all']:.4f} "
          f"accuracy_answered={metrics['accuracy_answered']}")


if __name__ == "__main__":
    main()
