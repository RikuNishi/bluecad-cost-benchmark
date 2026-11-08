"""Score official task predictions, including unanswered questions."""

import re
from collections import Counter


def parse_prediction(raw, task, cot=False):
    """Parse a model response using the official task-specific conventions."""
    task = str(task)
    if task not in ("1", "2"):
        raise ValueError("task must be 1 or 2")
    if not isinstance(raw, str):
        return None
    if task == "1":
        if cot:
            # Official CoT rule: the verdict must open the final non-empty line,
            # allowing an "Answer:"/"Final answer:" prefix and markdown quoting.
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            match = re.match(r"^(?:(?:final\s+)?answer\s*[:\-]?\s*)?[*_\"'(]*(yes|no)\b",
                             lines[-1], re.I) if lines else None
        else:
            match = re.search(r"\b(yes|no)\b", raw, re.I)
        return match.group(1).lower() if match else None
    match = re.search(r"\bANSWER\s*:\s*\(?([A-D])\)?\b", raw, re.I)
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b[A-Z]\b", raw.upper())
    return letters[-1] if letters and letters[-1] in "ABCD" else None


def _headline(entries):
    n = len(entries)
    answered = [(item, pred) for item, pred, reason in entries if reason is None]
    correct = sum(item["answer"] == pred for item, pred in answered)
    confusion = {}
    for item, pred in answered:
        row = confusion.setdefault(item["answer"], {})
        row[pred] = row.get(pred, 0) + 1
    labels = sorted({item["answer"] for item, _, _ in entries})
    f1s = []
    for label in labels:
        tp = sum(item["answer"] == label and pred == label for item, pred in answered)
        fp = sum(item["answer"] != label and pred == label for item, pred in answered)
        fn = sum(item["answer"] == label and pred != label for item, pred in answered)
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    reasons = Counter(reason for _, _, reason in entries if reason is not None)
    return {
        "n_items": n, "n_answered": len(answered),
        "answer_rate": len(answered) / n if n else 0.0,
        "n_error": reasons["error"], "n_unparseable": reasons["unparseable"],
        "n_missing_predictions": reasons["missing"],
        "accuracy_all": correct / n if n else 0.0,
        "accuracy_answered": correct / len(answered) if answered else None,
        "macro_f1_answered": sum(f1s) / len(f1s) if answered and f1s else None,
        "confusion_answered": confusion,
    }


def score(items, predictions, task, cot=False):
    """Score items; the first prediction for each ID takes precedence."""
    if str(task) not in ("1", "2"):
        raise ValueError("task must be 1 or 2")
    indexed, duplicates = {}, []
    for record in predictions:
        item_id = record["id"]
        if item_id in indexed:
            if item_id not in duplicates:
                duplicates.append(item_id)
        else:
            indexed[item_id] = record
    entries = []
    for item in items:
        record = indexed.get(item["id"])
        pred, reason = None, None
        if record is None:
            reason = "missing"
        elif record.get("error") or str(record.get("pred", "")).upper() == "ERROR":
            reason = "error"
        elif str(record.get("pred", "")).upper() == "UNPARSEABLE":
            reason = "unparseable"
        else:
            value = record.get("pred") if record.get("pred") is not None else record.get("raw")
            pred = parse_prediction(value, task, cot=cot)
            if pred is None:
                reason = "unparseable"
        entries.append((item, pred, reason))
    metrics = _headline(entries)
    metrics["duplicate_prediction_ids"] = duplicates
    metrics["by_group"] = {group: _headline([e for e in entries if e[0].get("group") == group])
                           for group in sorted({e[0]["group"] for e in entries if e[0].get("group") is not None})}
    tiers = {e[0].get("meta", {}).get("tier") for e in entries if e[0].get("meta", {}).get("tier") is not None}
    metrics["by_tier"] = {tier: _headline([e for e in entries if e[0].get("meta", {}).get("tier") == tier])
                          for tier in sorted(tiers, key=str)}
    metrics["evaluated_ids"] = [item["id"] for item, _, _ in entries]
    metrics["answered_ids"] = [item["id"] for item, _, reason in entries if reason is None]
    return metrics
