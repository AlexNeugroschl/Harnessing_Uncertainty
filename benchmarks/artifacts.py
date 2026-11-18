"""Atomic JSON and interruption-tolerant JSONL for experiment artifacts."""
import json
from pathlib import Path
import warnings


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    lines, rows = raw.splitlines(), []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith("\n"):
                warnings.warn(f"Ignoring interrupted final JSONL record in {path}")
                break
            raise
    return rows
