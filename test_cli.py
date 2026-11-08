"""Tests for the terminal front end, driven by scripted keyboard input."""

import io
from datetime import date, time

import pytest
from rich.console import Console

import storage
from cli import PawPalCLI
from pawpal_system import Owner, Pet, Priority, Task
from test_ai import FakeLLM, act

DAY = date(2026, 10, 5)


@pytest.fixture
def keyboard(monkeypatch):
    """Call with the answers to type; running out of answers acts like Ctrl+D."""
    def type_in(*answers):
        queue = list(answers)

        def fake_input(prompt=""):
            if not queue:
                raise EOFError
            return queue.pop(0)

        monkeypatch.setattr("builtins.input", fake_input)
    return type_in


def make_cli(tmp_path, owner=None, llm=None):
    console = Console(file=io.StringIO(), width=140, color_system=None)
    return PawPalCLI(owner or Owner("Jordan"), llm, console, tmp_path / "pawpal.json", DAY)


def output(cli) -> str:
    return cli.console.file.getvalue()


def test_add_a_pet_and_task_then_see_the_schedule_and_it_is_saved(tmp_path, keyboard):
    keyboard(
        "3", "a", "Mochi", "dog", "3", "Pulls on the leash", "b",  # Pets > add
        "2", "a", "Morning walk", "30", "high", "daily", "8:30", "n", "b",  # Tasks > add
        "1", "b",  # Schedule
        "q",
    )
    cli = make_cli(tmp_path)
    cli.run()

    saved = storage.load(tmp_path / "pawpal.json")
    walk = saved.pets[0].tasks[0]
    assert (saved.pets[0].notes, walk.title, walk.duration_minutes, walk.priority, walk.preferred_time) == (
        "Pulls on the leash", "Morning walk", 30, Priority.HIGH, time(8, 30))
    assert "08:30-09:00" in output(cli) and "fixed at preferred time 08:30" in output(cli)


def test_bad_input_is_asked_again(tmp_path, keyboard):
    owner = Owner("Jordan")
    owner.add_pet(Pet("Mochi", "dog"))
    keyboard("2", "a", "", "Fetch", "999", "20", "medium", "daily", "25 o'clock", "", "y", "", "9:00", "", "b", "q")
    cli = make_cli(tmp_path, owner)
    cli.run()

    fetch = owner.pets[0].tasks[0]
    assert (fetch.duration_minutes, fetch.preferred_time, fetch.done_by) == (20, None, time(9, 0))
    text = output(cli)
    assert "Please enter something" in text and "from 1 to 240" in text and "time must be HH:MM" in text


def test_describe_a_task_with_the_ai_and_add_it(tmp_path, keyboard):
    owner = Owner("Jordan")
    owner.add_pet(Pet("Mochi", "dog"))
    llm = FakeLLM({"pet_name": "Mochi", "title": "Evening walk", "duration_minutes": 25, "priority": "high",
                   "preferred_time": None, "not_before": "17:00", "done_by": None, "frequency": "daily"})
    keyboard("2", "d", "walk Mochi after 5pm for 25 minutes, important", "y", "b", "q")
    cli = make_cli(tmp_path, owner, llm)
    cli.run()

    task = owner.pets[0].tasks[0]
    assert (task.title, task.not_before, task.priority) == ("Evening walk", time(17, 0), Priority.HIGH)
    assert "Answered by the AI." in output(cli)
    assert cli.quota.used == 1


def test_without_ai_the_built_in_rules_draft_the_task(tmp_path, keyboard):
    owner = Owner("Jordan")
    owner.add_pet(Pet("Biscuit", "cat"))
    keyboard("2", "d", "give Biscuit her meds at 8am, never skip", "y", "b", "q")
    cli = make_cli(tmp_path, owner)
    cli.run()

    task = owner.pets[0].tasks[0]
    assert (task.title, task.priority, task.preferred_time) == ("Medication", Priority.CRITICAL, time(8, 0))
    assert "built-in rules handled this" in output(cli)


def test_plan_my_day_shows_the_proposal_and_applies_it_on_approval(tmp_path, keyboard):
    owner = Owner("Jordan", available_minutes=120)
    mochi = Pet("Mochi", "dog")
    owner.add_pet(mochi)
    mochi.add_task(Task("Fetch", 20, "medium", start_date=DAY))
    llm = FakeLLM(
        act(("update_task", {"pet_name": "Mochi", "title": "Fetch", "priority": "high"})),
        act(("finish", {"summary": "Raised Fetch to high [history Oct 05]."})),
    )
    keyboard("4", "p", "Fetch matters most today", "s", "a", "b", "q")
    cli = make_cli(tmp_path, owner, llm)
    cli.run()

    assert mochi.tasks[0].priority == Priority.HIGH
    text = output(cli)
    assert "Raised Fetch to high [history Oct 05]." in text  # bracketed text isn't swallowed as markup
    assert "Change Mochi: Fetch: priority → high." in text
    assert "Plan applied (1 change(s))." in text


def test_rejecting_a_plan_changes_nothing(tmp_path, keyboard):
    owner = Owner("Jordan", available_minutes=30)
    mochi = Pet("Mochi", "dog")
    owner.add_pet(mochi)
    mochi.add_task(Task("Walk", 30, "high", start_date=DAY))
    mochi.add_task(Task("Brush coat", 20, "low", start_date=DAY))
    keyboard("4", "p", "I only have 20 minutes, skip the coat", "r", "b", "q")
    cli = make_cli(tmp_path, owner)
    cli.run()

    assert owner.available_minutes == 30 and mochi.tasks[1].start_date == DAY
    assert "Nothing was changed" in output(cli)


def test_guideline_check_and_settings(tmp_path, keyboard):
    owner = Owner("Jordan")
    mochi = Pet("Mochi", "dog", guidelines=["At least 60 minutes of walks a day"])
    owner.add_pet(mochi)
    mochi.add_task(Task("Walk", 30, "high", start_date=DAY))
    keyboard("4", "c", "b", "5", "w", "7:00", "90", "o", "d", "2026-10-06", "b", "q")
    cli = make_cli(tmp_path, owner)
    cli.run()

    assert "[Mochi guideline 1] At least 60 minutes of walks a day" in output(cli)
    assert (owner.day_start, owner.available_minutes, owner.prefer_high_priority_first) == (time(7, 0), 90, False)
    assert cli.day == date(2026, 10, 6)


def test_ctrl_d_saves_and_exits(tmp_path, keyboard):
    keyboard("5", "n", "Sam")  # then input runs out
    cli = make_cli(tmp_path)
    cli.run()

    assert storage.load(tmp_path / "pawpal.json").name == "Sam"
    assert "Bye!" in output(cli)
