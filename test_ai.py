"""Tests for the AI features, using a scripted fake model so no API key is needed."""

from collections import OrderedDict, deque
from datetime import date, time, timedelta
from types import SimpleNamespace

import pytest

from ai_assistant import (
    PlanningAgent,
    apply_changes,
    draft_from_json,
    explain_schedule,
    parse_task,
    suggest_tasks,
)
from llm import GeminiLLM, LLMError, parse_json
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
        {"pet": "mochi", "guideline": 1, "issue": "Only 30 min of walks", "suggestion": "Add an evening walk"},
        {"pet": "Mochi", "guideline": 7, "issue": "Made up", "suggestion": ""},
        {"pet": "Mochi", "guideline": 1, "issue": ""},
        {"pet": "Biscuit", "guideline": 1, "issue": "Biscuit has no guidelines"},
        {"pet": "Rex", "guideline": 1, "issue": "No such pet"},
    ]})

    issues = check_guidelines(llm, owner, retriever_for(owner), DAY)

    assert len(issues) == 1
    assert issues[0].ref == "Mochi guideline 1"
    assert issues[0].guideline == "At least 60 minutes of walks a day"


def test_check_guidelines_uses_one_request_for_all_pets():
    owner = make_owner()
    owner.pets[0].guidelines = ["At least 60 minutes of walks a day"]
    owner.pets[1].guidelines = ["Medication before 08:00"]
    llm = FakeLLM({"issues": [{"pet": "Biscuit", "guideline": 1, "issue": "Medication is at 09:00"}]})

    issues = check_guidelines(llm, owner, retriever_for(owner), DAY)

    assert len(llm.prompts) == 1
    assert "## Mochi" in llm.prompts[0][1] and "## Biscuit" in llm.prompts[0][1]
    assert issues[0].ref == "Biscuit guideline 1"


def test_check_guidelines_skips_request_without_guidelines():
    owner = make_owner()
    llm = FakeLLM()
    assert check_guidelines(llm, owner, retriever_for(owner), DAY) == []
    assert llm.prompts == []


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

def act(*actions, thought=""):
    """One agent reply carrying several actions."""
    return {"thought": thought, "actions": [{"tool": tool, "args": args} for tool, args in actions]}


def test_agent_works_on_a_copy_until_approved():
    owner = make_owner()
    llm = FakeLLM(
        act(("update_task", {"pet_name": "Mochi", "title": "Fetch", "priority": "high"}), thought="fetch matters"),
        act(("finish", {"summary": "Raised Fetch to high."})),
    )

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner)).run("make fetch important")

    fetch = owner.pets[0].tasks[1]
    assert fetch.priority == Priority.MEDIUM  # real data untouched
    assert proposal.summary == "Raised Fetch to high."
    assert proposal.turns == 2
    assert [s.tool for s in proposal.steps] == ["update_task"]

    apply_changes(owner, proposal.changes, DAY)
    assert fetch.priority == Priority.HIGH


def test_agent_prompt_has_context_so_no_lookup_calls_are_needed():
    owner = make_owner()
    owner.pets[1].guidelines = ["Medication before 08:00"]
    llm = FakeLLM(act(("finish", {"summary": "Already fine."})))

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner)).run("check my day")

    prompt = llm.prompts[0][1]
    assert "[Biscuit guideline 1] Medication before 08:00" in prompt
    assert "Biscuit: Medication: 5 min" in prompt  # tasks due
    assert owner.create_schedule(DAY).explain() in prompt  # current schedule
    assert proposal.turns == 1 and proposal.changes == []


def test_agent_runs_several_actions_in_one_turn_and_sees_the_new_schedule():
    owner = make_owner()
    llm = FakeLLM(
        act(
            ("update_task", {"pet_name": "Mochi", "title": "Fetch", "priority": "high"}),
            ("update_task", {"pet_name": "Biscuit", "title": "Medication", "preferred_time": "08:00"}),
        ),
        act(("finish", {"summary": "done"})),
    )

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner)).run("goal")

    assert len(proposal.changes) == 2
    assert "Biscuit: Medication (5 min) [high] - fixed at preferred time 08:00" in llm.prompts[1][1]


def test_agent_finish_waits_until_it_has_seen_its_changes():
    owner = make_owner()
    llm = FakeLLM(
        act(("update_task", {"pet_name": "Mochi", "title": "Fetch", "priority": "high"}), ("finish", {"summary": "early"})),
        act(("finish", {"summary": "checked"})),
    )

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner)).run("goal")

    assert proposal.summary == "checked"
    assert proposal.turns == 2
    assert "not yet" in proposal.steps[-1].result


def test_agent_accepts_single_tool_replies():
    owner = make_owner()
    llm = FakeLLM({"tool": "finish", "args": {"summary": "ok"}})
    assert PlanningAgent(llm, owner, DAY, retriever_for(owner)).run("goal").summary == "ok"


def test_agent_sees_tool_errors_and_can_recover():
    owner = make_owner()
    llm = FakeLLM(
        act(("update_task", {"pet_name": "Mochi", "title": "Fetch", "priority": "urgent"}), ("fly", {})),
        {"thought": "nothing to do"},
        act(("finish", {"summary": "done"})),
    )

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner)).run("goal")

    assert proposal.steps[0].result.startswith("error:")
    assert "unknown tool" in proposal.steps[1].result
    assert "no actions" in proposal.steps[2].result
    assert proposal.changes == []
    assert "error: priority" in llm.prompts[1][1]  # the error was fed back to the model


def test_agent_stops_at_max_turns_with_warning():
    owner = make_owner()
    llm = FakeLLM(*[act(("search_guidelines", {"query": "walk"}))] * 3)

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner), max_turns=3).run("goal")

    assert proposal.turns == 3
    assert any("without finishing" in w for w in proposal.warnings)


def test_agent_flags_deferred_and_skipped_high_priority_tasks():
    owner = make_owner()
    llm = FakeLLM(
        act(
            ("set_available_time", {"day_start": "08:00", "available_minutes": 20}),
            ("defer_task", {"pet_name": "Biscuit", "title": "Medication"}),
        ),
        act(("finish", {"summary": "short day"})),
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
        act(("add_task", {"pet_name": "Biscuit", "title": "Play", "duration_minutes": 15})),
        act(("finish", {"summary": "added play"})),
    )

    proposal = PlanningAgent(llm, owner, later, retriever_for(owner, later)).run("more play")
    apply_changes(owner, proposal.changes, later)

    play = owner.pets[1].tasks[-1]
    assert play.title == "Play" and play.start_date == later


# ---- Gemini client: cache, pacing, fallback (no network) ----

class FakeAPIError(Exception):
    def __init__(self, code, message=""):
        super().__init__(message)
        self.code = code


class FakeModels:
    """Stands in for client.models; `outcomes` maps model name -> list of reply texts or exceptions."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        outcome = self.outcomes[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def make_gemini(outcomes, rpm=15) -> GeminiLLM:
    llm = GeminiLLM.__new__(GeminiLLM)  # skip __init__, which needs a real key
    llm.model, llm.models = "primary", ["primary", "backup"]
    llm.temperature, llm.rpm = 0.2, rpm
    llm.request_count = llm.cache_hits = 0
    llm._cache, llm._busy_until = OrderedDict(), {}
    llm._sent = {m: deque() for m in llm.models}
    llm._client = SimpleNamespace(models=FakeModels(outcomes))
    return llm


def test_gemini_caches_identical_requests():
    llm = make_gemini({"primary": ['{"a": 1}']})
    assert llm.generate_json("sys", "prompt") == {"a": 1}
    assert llm.generate_json("sys", "prompt") == {"a": 1}
    assert llm.request_count == 1 and llm.cache_hits == 1


def test_gemini_falls_back_when_model_is_busy_and_remembers_it():
    llm = make_gemini({"primary": [FakeAPIError(429, "Please retry in 30s.")], "backup": ["one", "two"]})
    assert llm.generate_text("sys", "first") == "one"
    assert llm.generate_text("sys", "second") == "two"
    assert llm._client.models.calls == ["primary", "backup", "backup"]  # primary skipped while cooling down


def test_gemini_paces_requests_to_rpm_limit():
    llm = make_gemini({"primary": ["a", "b"], "backup": ["c"]}, rpm=2)
    assert [llm.generate_text("sys", p) for p in ("1", "2", "3")] == ["a", "b", "c"]  # third goes to backup


def test_gemini_raises_other_errors_without_fallback():
    llm = make_gemini({"primary": [FakeAPIError(400, "bad request")], "backup": ["unused"]})
    with pytest.raises(LLMError, match="bad request"):
        llm.generate_text("sys", "prompt")
