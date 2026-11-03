"""Tests for the AI features, using a scripted fake model so no API key is needed."""

from datetime import date, time, timedelta

import pytest

from ai_assistant import (
    PlanningAgent,
    apply_changes,
    draft_from_json,
    explain_schedule,
    parse_task,
    suggest_tasks,
)
from llm import LLMError, parse_json
from pawpal_system import Owner, Pet, Priority, Task
from rag import Retriever, build_documents, check_guidelines

DAY = date(2026, 10, 5)  # a Monday


class FakeLLM:
    """Returns scripted replies in order and records every prompt it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def _next(self, system, prompt):
        self.prompts.append((system, prompt))
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return self.replies.pop(0)

    generate_json = _next
    generate_text = _next


def make_owner() -> Owner:
    owner = Owner("Jordan", available_minutes=120, day_start=time(8, 0))
    mochi, biscuit = Pet("Mochi", "dog", age=3), Pet("Biscuit", "cat", age=5)
    owner.add_pet(mochi)
    owner.add_pet(biscuit)
    mochi.add_task(Task("Morning walk", 30, "high", preferred_time=time(8, 30), start_date=DAY))
    mochi.add_task(Task("Fetch", 20, "medium", start_date=DAY))
    biscuit.add_task(Task("Medication", 5, "high", preferred_time=time(9, 0), start_date=DAY))
    return owner


def retriever_for(owner: Owner, day: date = DAY) -> Retriever:
    return Retriever(build_documents(owner, day))


# ---- JSON handling ----

def test_parse_json_strips_code_fence():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_rejects_garbage():
    with pytest.raises(LLMError):
        parse_json("not json")


# ---- Typing a task ----

def test_parse_task_builds_draft():
    owner = make_owner()
    llm = FakeLLM({"pet_name": "mochi", "title": "Evening walk", "duration_minutes": 30,
                   "priority": "High", "preferred_time": "18:00", "frequency": "daily"})

    draft = parse_task(llm, "walk mochi 30 min at 6pm, important", owner)

    assert draft.pet_name == "Mochi"  # matched case-insensitively
    assert draft.priority == "high"
    assert draft.preferred_time == time(18, 0)
    assert "Mochi (dog)" in llm.prompts[0][1]


def test_parse_task_does_not_add_task_before_review():
    owner = make_owner()
    before = len(owner.all_tasks())
    parse_task(FakeLLM({"pet_name": "Mochi", "title": "Brush", "duration_minutes": 10}), "brush mochi", owner)
    assert len(owner.all_tasks()) == before


def test_draft_rejects_unknown_pet():
    with pytest.raises(ValueError, match="unknown pet"):
        draft_from_json({"pet_name": "Rex", "title": "Walk", "duration_minutes": 10}, make_owner())


@pytest.mark.parametrize("bad", [
    {"duration_minutes": 0},
    {"duration_minutes": "lots"},
    {"priority": "urgent"},
    {"frequency": "monthly"},
    {"preferred_time": "6pm"},
    {"title": ""},
])
def test_draft_rejects_invalid_fields(bad):
    data = {"pet_name": "Mochi", "title": "Walk", "duration_minutes": 10, **bad}
    with pytest.raises(ValueError):
        draft_from_json(data, make_owner())


def test_draft_defaults_to_only_pet():
    owner = Owner("Sam")
    owner.add_pet(Pet("Solo", "cat"))
    assert draft_from_json({"title": "Feed", "duration_minutes": 5}, owner).pet_name == "Solo"


# ---- Retrieval ----

def test_retriever_finds_matching_guideline_for_pet():
    owner = make_owner()
    owner.pets[0].guidelines = ["At least 60 minutes of walks a day", "No food after 20:00"]
    owner.pets[1].guidelines = ["Brush coat twice a week"]

    hits = retriever_for(owner).search("how many minutes of walking", pet_name="mochi")

    assert hits[0].ref == "Mochi guideline 1"
    assert all(doc.pet_name == "Mochi" for doc in hits)


def test_retriever_returns_nothing_for_unrelated_query():
    owner = make_owner()
    owner.pets[0].guidelines = ["At least 60 minutes of walks a day"]
    assert retriever_for(owner).search("vaccination") == []


def test_history_includes_completions_overdue_and_skips():
    owner = make_owner()
    mochi = owner.pets[0]
    mochi.complete_task(mochi.tasks[0], DAY)
    owner.available_minutes = 10
    owner.create_schedule(DAY)  # Fetch (20 min) won't fit

    texts = [d.text for d in build_documents(owner, DAY + timedelta(days=1)) if d.source == "history"]

    assert any("Morning walk" in t and "completed" in t for t in texts)
    assert any("Fetch" in t and "overdue" in t for t in texts)
    assert any("Fetch" in t and "skipped" in t for t in texts)


# ---- Guideline check ----

def test_check_guidelines_keeps_valid_issues_and_drops_made_up_ones():
    owner = make_owner()
    owner.pets[0].guidelines = ["At least 60 minutes of walks a day"]
    llm = FakeLLM({"issues": [
        {"guideline": 1, "issue": "Only 30 min of walks", "suggestion": "Add an evening walk"},
        {"guideline": 7, "issue": "Made up", "suggestion": ""},
        {"guideline": 1, "issue": ""},
    ]})

    issues = check_guidelines(llm, owner, retriever_for(owner), DAY)

    assert len(issues) == 1
    assert issues[0].ref == "Mochi guideline 1"
    assert issues[0].guideline == "At least 60 minutes of walks a day"
    assert len(llm.prompts) == 1  # Biscuit has no guidelines, so no call for it


# ---- Task ideas ----

def test_suggest_tasks_filters_duplicates_and_invalid_ideas():
    owner = make_owner()
    mochi = owner.pets[0]
    llm = FakeLLM({"ideas": [
        {"title": "Morning walk", "duration_minutes": 30},  # already exists
        {"title": "Brush teeth", "duration_minutes": 5, "priority": "medium", "rationale": "Dental health"},
        {"title": "Nap", "duration_minutes": -5},  # invalid
        {"pet_name": "Biscuit", "title": "Nail trim", "duration_minutes": 10, "frequency": "weekly"},
    ]})

    ideas = suggest_tasks(llm, owner, mochi, retriever_for(owner), DAY)

    assert [i.title for i in ideas] == ["Brush teeth", "Nail trim"]
    assert all(i.pet_name == "Mochi" for i in ideas)  # forced to the chosen pet
    assert ideas[0].rationale == "Dental health"


# ---- Explain ----

def test_explain_schedule_sends_the_real_schedule():
    owner = make_owner()
    schedule = owner.create_schedule(DAY)
    llm = FakeLLM("Your day looks good.")

    assert explain_schedule(llm, schedule, retriever_for(owner)) == "Your day looks good."
    assert schedule.explain() in llm.prompts[0][1]


# ---- Planning agent ----

def test_agent_works_on_a_copy_until_approved():
    owner = make_owner()
    llm = FakeLLM(
        {"thought": "see the plan", "tool": "generate_schedule", "args": {}},
        {"thought": "fetch matters", "tool": "update_task", "args": {"pet_name": "Mochi", "title": "Fetch", "priority": "high"}},
        {"thought": "check", "tool": "generate_schedule", "args": {}},
        {"tool": "finish", "args": {"summary": "Raised Fetch to high."}},
    )

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner)).run("make fetch important")

    fetch = owner.pets[0].tasks[1]
    assert fetch.priority == Priority.MEDIUM  # real data untouched
    assert proposal.summary == "Raised Fetch to high."
    assert [s.tool for s in proposal.steps] == ["generate_schedule", "update_task", "generate_schedule"]
    assert proposal.steps[0].result.startswith("Plan for Jordan")

    apply_changes(owner, proposal.changes, DAY)
    assert fetch.priority == Priority.HIGH


def test_agent_sees_tool_errors_and_can_recover():
    owner = make_owner()
    llm = FakeLLM(
        {"tool": "update_task", "args": {"pet_name": "Mochi", "title": "Fetch", "priority": "urgent"}},
        {"tool": "fly", "args": {}},
        {"tool": "finish", "args": {"summary": "done"}},
    )

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner)).run("goal")

    assert proposal.steps[0].result.startswith("error:")
    assert "unknown tool" in proposal.steps[1].result
    assert proposal.changes == []
    assert "error: priority" in llm.prompts[1][1]  # the error was fed back to the model


def test_agent_stops_at_max_steps_with_warning():
    owner = make_owner()
    llm = FakeLLM(*[{"tool": "list_tasks", "args": {}}] * 3)

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner), max_steps=3).run("goal")

    assert len(proposal.steps) == 3
    assert any("without finishing" in w for w in proposal.warnings)


def test_agent_flags_deferred_and_skipped_high_priority_tasks():
    owner = make_owner()
    llm = FakeLLM(
        {"tool": "set_available_time", "args": {"day_start": "08:00", "available_minutes": 20}},
        {"tool": "defer_task", "args": {"pet_name": "Biscuit", "title": "Medication"}},
        {"tool": "finish", "args": {"summary": "short day"}},
    )

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner)).run("I only have 20 minutes")

    assert any("Defer Biscuit: Medication" in w for w in proposal.warnings)
    assert any("High-priority task Mochi: Morning walk" in w for w in proposal.warnings)
    assert owner.available_minutes == 120


def test_apply_changes_is_all_or_nothing():
    owner = make_owner()
    changes = [
        {"tool": "update_task", "args": {"pet_name": "Mochi", "title": "Fetch", "priority": "high"}},
        {"tool": "defer_task", "args": {"pet_name": "Mochi", "title": "Swim"}},  # no such task
    ]

    with pytest.raises(ValueError):
        apply_changes(owner, changes, DAY)
    assert owner.pets[0].tasks[1].priority == Priority.MEDIUM


def test_agent_add_task_starts_on_plan_day():
    owner = make_owner()
    later = DAY + timedelta(days=3)
    llm = FakeLLM(
        {"tool": "add_task", "args": {"pet_name": "Biscuit", "title": "Play", "duration_minutes": 15}},
        {"tool": "finish", "args": {"summary": "added play"}},
    )

    proposal = PlanningAgent(llm, owner, later, retriever_for(owner, later)).run("more play")
    apply_changes(owner, proposal.changes, later)

    play = owner.pets[1].tasks[-1]
    assert play.title == "Play" and play.start_date == later
