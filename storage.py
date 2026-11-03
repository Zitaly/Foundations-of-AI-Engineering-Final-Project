"""Save and load PawPal+ data as JSON, so pets, tasks, guidelines, and history survive restarts.

Writes go to a temporary file that then replaces the real one, so a crash mid-save can't corrupt it.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from pawpal_system import Owner, Pet, Schedule, Task, fmt

SCHEMA_VERSION = 1
DEFAULT_PATH = Path(os.environ.get("PAWPAL_DATA", Path(__file__).parent / "data" / "pawpal.json"))


class StorageError(ValueError):
    """The saved data couldn't be read."""


def _time(value: str | None) -> time | None:
    return datetime.strptime(value, "%H:%M").time() if value else None


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def task_to_dict(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "duration_minutes": task.duration_minutes,
        "priority": task.priority.name.lower(),
        "preferred_time": fmt(task.preferred_time) if task.preferred_time else None,
        "frequency": task.frequency,
        "start_date": task.start_date.isoformat(),
        "last_completed": task.last_completed.isoformat() if task.last_completed else None,
        "not_before": fmt(task.not_before) if task.not_before else None,
        "done_by": fmt(task.done_by) if task.done_by else None,
        "after": task.after,
        "after_gap_minutes": task.after_gap_minutes,
    }


def owner_to_dict(owner: Owner) -> dict[str, Any]:
    """Everything needed to rebuild `owner`. Schedules keep only what was skipped and why, for the AI's history."""
    return {
        "version": SCHEMA_VERSION,
        "owner": {
            "name": owner.name,
            "day_start": fmt(owner.day_start),
            "available_minutes": owner.available_minutes,
            "prefer_high_priority_first": owner.prefer_high_priority_first,
            "max_tasks_per_day": owner.max_tasks_per_day,
            "overdue_boost_per_day": owner.overdue_boost_per_day,
            "bump_lower_priority": owner.bump_lower_priority,
            "best_fit_packing": owner.best_fit_packing,
        },
        "pets": [
            {
                "name": pet.name,
                "species": pet.species,
                "age": pet.age,
                "notes": pet.notes,
                "guidelines": list(pet.guidelines),
                "tasks": [task_to_dict(task) for task in pet.tasks],
            }
            for pet in owner.pets
        ],
        "schedules": {
            day.isoformat(): [{"task_id": task.id, "reason": schedule.reasons[task]} for task in schedule.skipped]
            for day, schedule in sorted(owner.schedules.items())
        },
    }


def owner_from_dict(data: dict[str, Any]) -> Owner:
    """Rebuild an Owner saved by owner_to_dict. Raises StorageError if the data is malformed or invalid."""
    if not isinstance(data, dict) or data.get("version") != SCHEMA_VERSION:
        raise StorageError(f"expected PawPal+ data version {SCHEMA_VERSION}")
    try:
        o = data["owner"]
        owner = Owner(
            o["name"],
            available_minutes=o.get("available_minutes", 120),
            day_start=_time(o.get("day_start")) or time(8, 0),
            prefer_high_priority_first=o.get("prefer_high_priority_first", True),
            max_tasks_per_day=o.get("max_tasks_per_day"),
            overdue_boost_per_day=o.get("overdue_boost_per_day", 0.5),
            bump_lower_priority=o.get("bump_lower_priority", True),
            best_fit_packing=o.get("best_fit_packing", True),
        )
        tasks_by_id = {}
        for p in data.get("pets", []):
            pet = Pet(p["name"], p["species"], age=p.get("age", 0), notes=p.get("notes", ""), guidelines=list(p.get("guidelines", [])))
            owner.add_pet(pet)
            for t in p.get("tasks", []):
                task = Task(
                    t["title"],
                    t["duration_minutes"],
                    t["priority"],
                    preferred_time=_time(t.get("preferred_time")),
                    frequency=t.get("frequency", "daily"),
                    start_date=_date(t["start_date"]),
                    last_completed=_date(t.get("last_completed")),
                    not_before=_time(t.get("not_before")),
                    done_by=_time(t.get("done_by")),
                    after=t.get("after", ""),
                    after_gap_minutes=t.get("after_gap_minutes", 0),
                    id=t["id"],
                )
                pet.add_task(task)
                tasks_by_id[task.id] = task
        for day, entries in data.get("schedules", {}).items():
            skipped = [(tasks_by_id[e["task_id"]], e["reason"]) for e in entries if e["task_id"] in tasks_by_id]
            owner.schedules[_date(day)] = Schedule(
                _date(day), owner, skipped=[task for task, _ in skipped], reasons=dict(skipped)
            )
    except (KeyError, TypeError, ValueError) as e:
        raise StorageError(f"saved data is invalid: {e}") from None
    return owner


def dumps(owner: Owner) -> str:
    return json.dumps(owner_to_dict(owner), indent=2)


def loads(text: str) -> Owner:
    try:
        return owner_from_dict(json.loads(text))
    except json.JSONDecodeError as e:
        raise StorageError(f"not valid JSON: {e}") from None


def save(owner: Owner, path: Path = DEFAULT_PATH) -> None:
    """Write `owner` to `path` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(dumps(owner), encoding="utf-8")
    os.replace(temp, path)


def load(path: Path = DEFAULT_PATH) -> Owner | None:
    """The saved Owner, or None if nothing has been saved yet. Raises StorageError if the file is unreadable."""
    if not path.exists():
        return None
    return loads(path.read_text(encoding="utf-8"))


def quarantine(path: Path = DEFAULT_PATH) -> Path:
    """Move an unreadable save file aside (so it isn't overwritten) and return its new path."""
    backup = path.with_name(f"{path.stem}.unreadable-{datetime.now():%Y%m%d-%H%M%S}{path.suffix}")
    os.replace(path, backup)
    return backup
