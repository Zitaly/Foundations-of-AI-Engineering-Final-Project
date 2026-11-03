"""Tests for JSON saving/loading and schedule exports."""

import json
from datetime import date, time

import pytest

import storage
from export import schedule_rows, schedule_to_ics
from pawpal_system import Owner, Pet, Priority, Task
from rag import build_documents

DAY = date(2026, 10, 5)


def make_owner() -> Owner:
    owner = Owner("Jordan", available_minutes=30, day_start=time(7, 0), max_tasks_per_day=5, overdue_boost_per_day=1.0, best_fit_packing=False)
    mochi = Pet("Mochi", "dog", age=3, notes="Pulls on the leash", guidelines=["At least 60 minutes of walks a day"])
    owner.add_pet(mochi)
    walk = Task("Walk", 30, "high", preferred_time=time(7, 0), start_date=DAY)
    mochi.add_task(walk)
    mochi.add_task(Task("Brush, coat", 15, "critical", frequency="weekly", start_date=DAY))
    mochi.add_task(Task("Meds", 5, "high", not_before=time(7, 0), done_by=time(9, 0), after="Brush, coat", after_gap_minutes=10, start_date=DAY))
    mochi.complete_task(walk, DAY)
    return owner


def test_round_trip_keeps_every_field(tmp_path):
    owner = make_owner()
    owner.create_schedule(DAY)
    path = tmp_path / "pawpal.json"

    storage.save(owner, path)
    loaded = storage.load(path)

    assert storage.owner_to_dict(loaded) == storage.owner_to_dict(owner)
    assert (loaded.day_start, loaded.max_tasks_per_day, loaded.overdue_boost_per_day) == (time(7, 0), 5, 1.0)
    mochi = loaded.pets[0]
    assert mochi.notes == "Pulls on the leash" and mochi.guidelines == ["At least 60 minutes of walks a day"]
    assert [t.title for t in mochi.tasks] == ["Walk", "Brush, coat", "Meds", "Walk"]
    assert (mochi.tasks[2].not_before, mochi.tasks[2].done_by, mochi.tasks[2].after, mochi.tasks[2].after_gap_minutes) == (time(7, 0), time(9, 0), "Brush, coat", 10)
    assert all(t.pet is mochi for t in mochi.tasks)  # back-references are rebuilt
    assert mochi.tasks[0].last_completed == DAY
    assert mochi.tasks[1].priority == Priority.CRITICAL


def test_skip_history_survives_reload_for_the_ai():
    owner = make_owner()
    schedule = owner.create_schedule(date(2026, 10, 6))  # Walk (next occurrence) and Brush clash in 30 min
    assert schedule.skipped

    loaded = storage.loads(storage.dumps(owner))

    texts = [d.text for d in build_documents(loaded, date(2026, 10, 7)) if d.source == "history"]
    assert any("was skipped" in t for t in texts)


def test_load_returns_none_when_nothing_is_saved(tmp_path):
    assert storage.load(tmp_path / "missing.json") is None


@pytest.mark.parametrize("text", [
    "not json",
    json.dumps({"version": 99}),
    json.dumps({"version": 1, "owner": {}}),
    json.dumps({"version": 1, "owner": {"name": "J"}, "pets": [{"name": "M", "species": "dog",
               "tasks": [{"id": "1", "title": "Walk", "duration_minutes": -5, "priority": "high", "start_date": "2026-10-05"}]}]}),
])
def test_unreadable_data_raises_storage_error(text):
    with pytest.raises(storage.StorageError):
        storage.loads(text)


def test_quarantine_moves_bad_file_aside(tmp_path):
    path = tmp_path / "pawpal.json"
    path.write_text("garbage")

    backup = storage.quarantine(path)

    assert not path.exists() and backup.read_text() == "garbage"


def test_save_replaces_file_without_leaving_temp_files(tmp_path):
    path = tmp_path / "pawpal.json"
    storage.save(make_owner(), path)
    storage.save(Owner("Sam"), path)

    assert storage.load(path).name == "Sam"
    assert [p.name for p in tmp_path.iterdir()] == ["pawpal.json"]


def test_ics_export_has_one_event_per_planned_task():
    owner = make_owner()
    owner.available_minutes = 120
    schedule = owner.create_schedule(date(2026, 10, 6))

    ics = schedule_to_ics(schedule)

    assert ics.startswith("BEGIN:VCALENDAR\r\n") and ics.endswith("END:VCALENDAR\r\n")
    assert ics.count("BEGIN:VEVENT") == len(schedule.planned) == 3
    assert "DTSTART:20261006T070000" in ics
    assert "SUMMARY:Mochi: Brush\\, coat" in ics  # commas escaped


def test_schedule_rows_match_plan():
    owner = make_owner()
    owner.available_minutes = 120
    rows = schedule_rows(owner.create_schedule(date(2026, 10, 6)))
    assert [(r["Start"], r["End"], r["Task"]) for r in rows] == [
        ("07:00", "07:30", "Walk"), ("07:30", "07:45", "Brush, coat"), ("07:55", "08:00", "Meds"),  # Meds 10 min after Brush
    ]
