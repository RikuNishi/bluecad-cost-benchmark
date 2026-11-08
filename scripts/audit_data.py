#!/usr/bin/env python3
"""Audit official item references and the extracted BlueCAD data layout."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath


ITEM_FILES = ("task1_items.json", "task1_dim_items.json", "task2_items.json")


def parse_reference(path):
    """Extract layout information without interpreting numeric drawing views."""
    parts = PurePosixPath(path).parts
    for marker in ("renders", "png_out"):
        if marker not in parts:
            continue
        tail = parts[parts.index(marker) + 1:]
        # Official paths include a group between the layout marker and model ID.
        if marker == "renders" and len(tail) >= 3:
            return "render", tail[-2], tail[-1], tail[-3]
        if marker == "png_out" and len(tail) >= 4:
            return "drawing", tail[-3], tail[-2], tail[-4]
    return None


def png_names(directory):
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".png")


def audit(items_dir, data_dir, sample_parts=200):
    items_dir, data_dir = Path(items_dir), Path(data_dir)
    files = {}
    models_by_file = {}
    groups = defaultdict(set)
    total_found = total_missing = 0
    missing_examples = []
    drawing_examples = {}
    for filename in ITEM_FILES:
        path = items_dir / filename
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            items = json.load(f)["items"]
        ids = Counter(item["id"] for item in items)
        answers = Counter(item["answer"] for item in items)
        item_groups = Counter(item["group"] for item in items)
        image_counts = Counter(str(len(item["images"])) for item in items)
        models = set()
        found = missing = 0
        examples = []
        for item in items:
            for original in item["images"]:
                ref = parse_reference(original)
                target = None
                if ref:
                    kind, model, detail, group = ref
                    models.add(model)
                    groups[model].add(group)
                    if kind == "render":
                        target = data_dir / "renders" / model / detail
                        exists = target.is_file()
                    else:
                        target = data_dir / "drawings" / model / detail
                        names = png_names(target)
                        exists = target.is_dir() and len(names) == 6
                        key = f"{model}/{detail}"
                        if key not in drawing_examples and len(drawing_examples) < 20:
                            drawing_examples[key] = names
                else:
                    exists = False
                if exists:
                    found += 1
                else:
                    missing += 1
                    example = {"item_id": item["id"], "original_path": original,
                               "mapped_path": str(target) if target is not None else None}
                    if len(examples) < 20:
                        examples.append(example)
                    if len(missing_examples) < 20:
                        missing_examples.append({"item_file": filename, **example})
        total_found += found
        total_missing += missing
        models_by_file[filename] = models
        files[filename] = {
            "item_count": len(items), "answer_distribution": dict(sorted(answers.items())),
            "group_distribution": dict(sorted(item_groups.items())),
            "images_per_item_distribution": dict(sorted(image_counts.items())),
            "duplicate_item_ids": {k: v for k, v in sorted(ids.items()) if v > 1},
            "unique_model_ids": sorted(models), "unique_model_id_count": len(models),
            "references": {"found": found, "missing": missing, "missing_examples": examples},
        }
    overlap = {}
    names = sorted(models_by_file)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            shared = sorted(models_by_file[first] & models_by_file[second])
            overlap[f"{first} | {second}"] = {"count": len(shared), "model_ids": shared}
    part_ids = set()
    for root in (data_dir / "renders", data_dir / "drawings"):
        if root.is_dir():
            part_ids.update(p.name for p in root.iterdir() if p.is_dir())
    sampled = sorted(part_ids)[:sample_parts]
    incomplete = {}
    hashes = defaultdict(list)
    for model in sampled:
        render_dir = data_dir / "renders" / model
        missing = [f"renders/{model}/{n:03d}.png" for n in range(6)
                   if not (render_dir / f"{n:03d}.png").is_file()]
        if not (render_dir / "transforms.json").is_file():
            missing.append(f"renders/{model}/transforms.json")
        for variant in ("normal", "dim"):
            directory = data_dir / "drawings" / model / variant
            count = len(png_names(directory))
            if not directory.is_dir() or count != 6:
                missing.append(f"drawings/{model}/{variant}: expected 6 PNG files, found {count}")
        if missing:
            incomplete[model] = missing
        for directory in (render_dir, data_dir / "drawings" / model):
            if directory.is_dir():
                for image in sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".png"):
                    digest = hashlib.sha256()
                    with image.open("rb") as f:
                        for chunk in iter(lambda: f.read(1024 * 1024), b""):
                            digest.update(chunk)
                    hashes[digest.hexdigest()].append(str(image))
    return {
        "item_files": files,
        "missing_item_files": [name for name in ITEM_FILES if name not in files],
        "references": {"found": total_found, "missing": total_missing,
                       "missing_examples": missing_examples, "drawing_directory_examples": drawing_examples},
        "cross_file": {"model_id_overlap": overlap,
                       "inconsistent_groups": {k: sorted(v) for k, v in sorted(groups.items()) if len(v) > 1}},
        "sampled_parts": {"requested": sample_parts, "available": len(part_ids), "model_ids": sampled,
                          "incomplete_parts": incomplete},
        "exact_duplicates": [{"sha256": digest, "paths": paths} for digest, paths in sorted(hashes.items()) if len(paths) > 1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-parts", type=int, default=200)
    args = parser.parse_args()
    if args.sample_parts < 0:
        parser.error("--sample-parts must be nonnegative")
    report = audit(args.items_dir, args.data_dir, args.sample_parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    refs = report["references"]
    parts = report["sampled_parts"]
    print(f"Audited {len(report['item_files'])} item files: {refs['found']} references found, {refs['missing']} missing.")
    print(f"Sampled {len(parts['model_ids'])} parts; {len(parts['incomplete_parts'])} incomplete; "
          f"{len(report['exact_duplicates'])} exact duplicate groups. Report: {args.out}")


if __name__ == "__main__":
    main()
