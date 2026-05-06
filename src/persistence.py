"""JSON history file helpers for stateful nodes (recipes, predictions).

The Action commits these files back to the repo at the end of each run, giving
us durable cross-run memory without an external database.
"""
import json
from pathlib import Path
from typing import Any


def load_history(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return []


def save_history(
    path: str | Path,
    items: list[dict[str, Any]],
    max_entries: int = 200,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    trimmed = items[-max_entries:]
    p.write_text(json.dumps(trimmed, indent=2, default=str))


def append_history(
    path: str | Path,
    entry: dict[str, Any],
    max_entries: int = 200,
) -> None:
    items = load_history(path)
    items.append(entry)
    save_history(path, items, max_entries)
