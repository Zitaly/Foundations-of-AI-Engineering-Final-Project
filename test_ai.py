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
    parse_time,
    suggest_tasks,
)
from llm import MAX_WAIT_SECONDS, APIError, FallbackLLM, GeminiLLM, GroqLLM, LLMError, parse_json
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
    {"preferred_time": "25:00"},
    {"preferred_time": "around six"},
    {"title": ""},
])
def test_draft_rejects_invalid_fields(bad):
    data = {"pet_name": "Mochi", "title": "Walk", "duration_minutes": 10, **bad}
    with pytest.raises(ValueError):
        draft_from_json(data, make_owner())


@pytest.mark.parametrize("text, expected", [
    ("18:30", time(18, 30)), ("6pm", time(18, 0)), ("6:30 p.m.", time(18, 30)),
    ("7 AM", time(7, 0)), ("noon", time(12, 0)), ("flexible", None), (None, None),
])
def test_parse_time_accepts_common_formats(text, expected):
    assert parse_time(text) == expected


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


class FailAfter(FakeLLM):
    """Answers with the scripted replies, then raises LLMError like a model that hit its quota."""

    def _next(self, system, prompt):
        if not self.replies:
            raise LLMError("quota exceeded")
        return super()._next(system, prompt)

    generate_json = _next


def test_agent_keeps_partial_plan_when_model_fails_mid_run():
    owner = make_owner()
    llm = FailAfter(act(("update_task", {"pet_name": "Mochi", "title": "Fetch", "priority": "high"})))

    proposal = PlanningAgent(llm, owner, DAY, retriever_for(owner)).run("goal")

    assert len(proposal.changes) == 1
    assert proposal.turns == 1
    assert any("stopped responding after 1 turn" in w for w in proposal.warnings)
    assert not any("without finishing" in w for w in proposal.warnings)


def test_agent_failure_on_first_turn_is_raised_for_fallback():
    owner = make_owner()
    with pytest.raises(LLMError):
        PlanningAgent(FailAfter(), owner, DAY, retriever_for(owner)).run("goal")


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
    llm.temperature, llm.rpm, llm.max_wait = 0.2, rpm, MAX_WAIT_SECONDS
    llm.request_count = llm.cache_hits = 0
    llm._cache, llm._busy_until, llm._cache_path = OrderedDict(), {}, None
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


def test_gemini_skips_a_retired_model_for_the_rest_of_the_session():
    llm = make_gemini({"primary": [FakeAPIError(404, "model not found")], "backup": ["one", "two"]})
    assert llm.generate_text("sys", "first") == "one"
    assert llm.generate_text("sys", "second") == "two"
    assert llm._client.models.calls == ["primary", "backup", "backup"]


class ReadTimeout(Exception):
    """Named like httpx's timeout, which has no status code."""


def test_gemini_falls_back_on_timeout():
    llm = make_gemini({"primary": [ReadTimeout("timed out")], "backup": ["ok"]})
    assert llm.generate_text("sys", "prompt") == "ok"


def test_gemini_tries_next_model_after_empty_reply():
    llm = make_gemini({"primary": [""], "backup": ["ok"]})
    assert llm.generate_text("sys", "prompt") == "ok"


def test_gemini_reports_every_model_blocked():
    llm = make_gemini({"primary": [""], "backup": [""]})
    with pytest.raises(LLMError, match="empty or unusable response"):
        llm.generate_text("sys", "prompt")


def test_gemini_asks_again_after_invalid_json_and_does_not_cache_it():
    llm = make_gemini({"primary": ["not json", '{"a": 1}', '{"a": 2}']})
    assert llm.generate_json("sys", "prompt") == {"a": 1}
    assert llm.request_count == 2
    assert ("application/json", "sys", "prompt") not in llm._cache  # the broken reply isn't served again
    assert llm.generate_json("sys", "prompt") == {"a": 2}


# ---- Groq client and provider chain (no network) ----

class ScriptedGroq(GroqLLM):
    """GroqLLM whose HTTP call returns scripted replies (dicts) or raises scripted errors, recording each body."""

    def __init__(self, *outcomes, cache_dir=None):
        super().__init__(api_key="test-key", cache_dir=cache_dir)
        self.outcomes, self.bodies = list(outcomes), []

    def _post(self, body):
        self.bodies.append(body)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def reply(text):
    return {"choices": [{"message": {"content": text}}]}


def test_groq_sends_json_mode_and_low_reasoning_effort():
    groq = ScriptedGroq(reply('{"a": 1}'))

    assert groq.generate_json("Reply with JSON.", "prompt") == {"a": 1}

    body = groq.bodies[0]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == "low"
    assert body["messages"][0] == {"role": "system", "content": "Reply with JSON."}


def test_groq_moves_to_next_model_on_rate_limit_using_retry_after():
    groq = ScriptedGroq(APIError(429, "rate limited", {"retry-after": "7"}), reply("hi"))

    assert groq.generate_text("sys", "prompt") == "hi"
    assert [b["model"] for b in groq.bodies] == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    assert "reasoning_effort" in groq.bodies[1] and "response_format" not in groq.bodies[1]


def test_groq_treats_failed_json_validation_as_unusable_and_tries_next_model():
    groq = ScriptedGroq(APIError(400, '{"error": {"code": "json_validate_failed"}}'), reply('{"ok": true}'))
    assert groq.generate_json("sys", "prompt") == {"ok": True}


def test_groq_strips_leaked_reasoning():
    assert ScriptedGroq(reply("<think>hmm</think>Hello")).generate_text("sys", "p") == "Hello"


def test_groq_needs_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="No Groq API key"):
        GroqLLM(cache_dir=None)


def test_fallback_llm_uses_next_provider_and_records_who_answered():
    gemini = make_gemini({"primary": [FakeAPIError(400, "bad request")], "backup": []})
    groq = ScriptedGroq(reply("from groq"))
    chain = FallbackLLM([gemini, groq])

    assert chain.generate_text("sys", "prompt") == "from groq"
    assert chain.last_used == "Groq (openai/gpt-oss-20b)"
    assert chain.name == "Gemini → Groq"


def test_fallback_llm_raises_when_every_provider_fails():
    chain = FallbackLLM([ScriptedGroq(APIError(401, "invalid key"))])
    with pytest.raises(LLMError, match="invalid key"):
        chain.generate_text("sys", "prompt")


def test_reply_cache_survives_a_restart(tmp_path):
    first = ScriptedGroq(reply("cached answer"), cache_dir=tmp_path)
    assert first.generate_text("sys", "prompt") == "cached answer"

    second = ScriptedGroq(cache_dir=tmp_path)  # no scripted replies: any request would fail
    assert second.generate_text("sys", "prompt") == "cached answer"
    assert second.cache_hits == 1 and second.request_count == 0


def test_damaged_cache_file_is_ignored(tmp_path):
    (tmp_path / "groq.json").write_text("not json")
    assert ScriptedGroq(reply("fresh"), cache_dir=tmp_path).generate_text("sys", "p") == "fresh"


# ---- Agent tools for timing rules ----

def test_agent_can_set_window_and_order():
    owner = make_owner()
    changes = [
        {"tool": "update_task", "args": {"pet_name": "Mochi", "title": "Fetch", "not_before": "09:00", "done_by": "10:00"}},
        {"tool": "update_task", "args": {"pet_name": "Mochi", "title": "Fetch", "after": "Morning walk", "after_gap_minutes": 10}},
    ]

    descriptions = apply_changes(owner, changes, DAY)

    fetch = owner.pets[0].tasks[1]
    assert (fetch.not_before, fetch.done_by, fetch.after, fetch.after_gap_minutes) == (time(9, 0), time(10, 0), "Morning walk", 10)
    assert "not_before → 09:00" in descriptions[0]


def test_provider_failure_does_not_send_a_json_repair_request():
    from llm import HostedLLM

    class Busy(HostedLLM):
        provider = "Busy"

        def __init__(self):
            super().__init__(["m"], rpm=100, temperature=0, cache_dir=None)
            self.max_wait = 0
            self.prompts = []

        def _call(self, model, system, prompt, mime_type):
            self.prompts.append(prompt)
            raise APIError(429, "quota exceeded")

    llm = Busy()
    with pytest.raises(LLMError, match="No Busy model could answer"):
        llm.generate_json("s", "p")
    assert llm.prompts == ["p"]  # no "reply again with only the JSON" request after a failure
