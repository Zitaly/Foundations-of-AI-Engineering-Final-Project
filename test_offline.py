"""Tests for the rule-based fallbacks used when no model is available."""

from datetime import date, time

import pytest

from ai_assistant import apply_changes, with_fallback
from llm import LLMError
from offline import (
    check_guidelines_rules,
    explain_schedule_rules,
    goal_phrases,
    parse_task_rules,
    plan_rules,
    same_task,
    suggest_tasks_rules,
    unchecked_guidelines,
)
from pawpal_system import Owner, Pet, Priority, Task
from rag import Retriever, build_documents

DAY = date(2026, 10, 5)


def make_owner() -> Owner:
    owner = Owner("Jordan", available_minutes=120, day_start=time(8, 0))
    owner.add_pet(Pet("Mochi", "dog", age=3))
    owner.add_pet(Pet("Biscuit", "cat", age=5))
    return owner


# ---- with_fallback ----

class FailingLLM:
    def generate_json(self, system, prompt):
        raise LLMError("quota exceeded")

    generate_text = generate_json


def test_with_fallback_uses_rules_when_ai_is_off_or_fails():
    assert with_fallback(None, lambda llm: "ai", lambda: "rules") == ("rules", "AI is off, so PawPal+'s built-in rules handled this.")

    result, note = with_fallback(FailingLLM(), lambda llm: llm.generate_text("", ""), lambda: "rules")
    assert result == "rules" and "quota exceeded" in note

    assert with_fallback(object(), lambda llm: "ai", lambda: "rules") == ("ai", None)


# ---- Typing a task ----

@pytest.mark.parametrize("text, expected", [
    ("Walk Mochi for 30 minutes every morning at 7:30, it's important",
     ("Mochi", "Morning walk", 30, "high", time(7, 30), "daily")),
    ("give biscuit her meds at 8am, never skip", ("Biscuit", "Medication", 5, "critical", time(8, 0), "daily")),
    ("brush Mochi's coat weekly if there's time", ("Mochi", "Brush coat", 15, "low", None, "weekly")),
    ("Biscuit dinner at 6:30 pm", ("Biscuit", "Dinner", 10, "high", time(18, 30), "daily")),
    ("Mochi zoomies for half an hour", ("Mochi", "Zoomies", 30, "medium", None, "daily")),
    ("Mochi swim 1.5 hours", ("Mochi", "Swim", 90, "medium", None, "daily")),
])
def test_rules_parse_common_descriptions(text, expected):
    d = parse_task_rules(text, make_owner())
    assert (d.pet_name, d.title, d.duration_minutes, d.priority, d.preferred_time, d.frequency) == expected


def test_rules_need_a_pet_name_when_there_are_several_pets():
    with pytest.raises(ValueError, match="which pet"):
        parse_task_rules("walk for 20 minutes", make_owner())


def test_rules_use_the_only_pet_when_none_is_named():
    owner = Owner("Sam")
    owner.add_pet(Pet("Solo", "cat"))
    assert parse_task_rules("clean the litter", owner).pet_name == "Solo"


# ---- Matching phrases to tasks ----

@pytest.mark.parametrize("phrase, title, expected", [
    ("walks", "Morning walk", True),
    ("medication and feeding", "Breakfast", True),
    ("food", "Dinner", True),
    ("brush teeth", "Brush coat", False),
    ("the walk", "Play session", False),
])
def test_same_task(phrase, title, expected):
    assert same_task(phrase, title) is expected


# ---- Task ideas ----

def test_rule_ideas_skip_existing_tasks_and_put_guideline_matches_first():
    owner = make_owner()
    mochi = owner.pets[0]
    mochi.add_task(Task("Breakfast", 10, "high", start_date=DAY))
    mochi.guidelines = ["Teeth brushed every day"]

    ideas = suggest_tasks_rules(owner, mochi, Retriever(build_documents(owner, DAY)), DAY)

    assert ideas[0].title == "Brush teeth" and "[Mochi guideline 1]" in ideas[0].rationale
    assert "Breakfast" not in [i.title for i in ideas]
    assert len(ideas) == 3


def test_rule_ideas_include_senior_check_for_old_pets():
    owner = Owner("Sam")
    old = Pet("Rex", "dog", age=12)
    owner.add_pet(old)
    ideas = suggest_tasks_rules(owner, old, Retriever(build_documents(owner, DAY)), DAY, count=20)
    assert "Mobility check" in [i.title for i in ideas]


# ---- Explaining ----

def test_rule_explanation_mentions_skips_and_suggests_more_time():
    owner = make_owner()
    owner.available_minutes = 30
    owner.pets[0].add_task(Task("Walk", 30, "high", start_date=DAY))
    owner.pets[1].add_task(Task("Play", 20, "low", start_date=DAY))

    text = explain_schedule_rules(owner.create_schedule(DAY))

    assert "1 task is planned between 08:00 and 08:30" in text
    assert "Didn't fit: Biscuit: Play" in text
    assert "About 20 more available minutes" in text


# ---- Checking guidelines ----

def test_rule_guideline_check_finds_amount_time_and_frequency_problems():
    owner = make_owner()
    mochi, biscuit = owner.pets
    mochi.add_task(Task("Morning walk", 30, "high", start_date=DAY))
    mochi.add_task(Task("Dinner", 10, "high", preferred_time=time(9, 30), start_date=DAY))
    biscuit.add_task(Task("Medication", 5, "critical", preferred_time=time(9, 0), start_date=DAY))
    mochi.guidelines = ["At least 60 minutes of walks a day", "No food after 09:15", "Be gentle"]
    biscuit.guidelines = ["Medication must be given before 09:00", "Brush coat twice a week"]

    issues = {i.ref: i for i in check_guidelines_rules(owner, DAY)}

    assert set(issues) == {"Mochi guideline 1", "Mochi guideline 2", "Biscuit guideline 1", "Biscuit guideline 2"}
    assert "Only 30 min" in issues["Mochi guideline 1"].issue
    assert "after 09:15" in issues["Mochi guideline 2"].issue
    assert "not before 09:00" in issues["Biscuit guideline 1"].issue
    assert unchecked_guidelines(owner) == ["Mochi guideline 3"]


def test_rule_guideline_check_passes_when_guidelines_are_met():
    owner = make_owner()
    mochi = owner.pets[0]
    mochi.add_task(Task("Morning walk", 30, "high", start_date=DAY))
    mochi.add_task(Task("Evening walk", 30, "high", start_date=DAY))
    mochi.guidelines = ["At least 1 hour of walking a day"]

    assert check_guidelines_rules(owner, DAY) == []


def test_rule_guideline_check_does_not_store_a_schedule():
    owner = make_owner()
    owner.pets[0].guidelines = ["At least 60 minutes of walks a day"]
    check_guidelines_rules(owner, DAY)
    assert owner.schedules == {}


# ---- Planning ----

def test_goal_phrases_split_keep_and_skip_instructions():
    keep, drop = goal_phrases("I have 20 minutes. Make sure meds, feeding and walks happen, don't skip teeth. Skip dinner but keep play.")
    assert [p.strip() for p in keep] == ["meds", "feeding and walks happen", "teeth", "play"]
    assert [p.strip() for p in drop] == ["dinner"]


def test_rule_plan_sets_time_raises_kept_tasks_and_defers_skipped_ones():
    owner = make_owner()
    mochi = owner.pets[0]
    mochi.add_task(Task("Walk", 30, "high", start_date=DAY))
    mochi.add_task(Task("Brush teeth", 5, "low", start_date=DAY))
    mochi.add_task(Task("Brush coat", 20, "low", start_date=DAY))

    proposal = plan_rules(owner, DAY, "I only have 40 minutes. Don't skip the teeth, skip the coat.")

    assert [c["tool"] for c in proposal.changes] == ["set_available_time", "defer_task", "update_task"]
    assert {t.title for _, t in proposal.schedule.planned} == {"Walk", "Brush teeth"}
    assert owner.available_minutes == 120  # real data untouched until approved

    apply_changes(owner, proposal.changes, DAY)
    assert owner.available_minutes == 40
    assert mochi.tasks[1].priority == Priority.HIGH


def test_rule_plan_raises_kept_task_to_critical_when_high_is_not_enough():
    owner = make_owner()
    mochi = owner.pets[0]
    mochi.add_task(Task("Walk", 30, "high", preferred_time=time(8, 0), start_date=DAY))  # fixed, so placed first
    mochi.add_task(Task("Medication", 5, "medium", start_date=DAY))

    proposal = plan_rules(owner, DAY, "I only have 30 minutes, make sure medication happens")

    assert "Medication" in {t.title for _, t in proposal.schedule.planned}
    assert proposal.changes[-1]["args"]["priority"] == "critical"


def test_rule_plan_explains_when_it_understood_nothing():
    owner = make_owner()
    owner.pets[0].add_task(Task("Walk", 30, "high", start_date=DAY))
    proposal = plan_rules(owner, DAY, "have a nice day")
    assert proposal.changes == [] and "didn't find" in proposal.summary


@pytest.mark.parametrize("text, expected", [
    ("give Mochi meds before 9am", (None, None, time(9, 0))),
    ("walk Mochi after 5pm for 30 min", (None, time(17, 0), None)),
    ("Mochi dinner at 6:30 pm", (time(18, 30), None, None)),
    ("Mochi brush between 10:00 and 11:00, by noon", (time(10, 0), None, time(12, 0))),
])
def test_rules_read_fixed_times_and_windows(text, expected):
    d = parse_task_rules(text, make_owner())
    assert (d.preferred_time, d.not_before, d.done_by) == expected
