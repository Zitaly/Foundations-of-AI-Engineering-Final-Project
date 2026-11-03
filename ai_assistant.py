"""AI features for PawPal+: natural-language task entry, task ideas, schedule explanations, and a planning agent.

The model never edits the owner's data directly. Every task it drafts goes through Task's own validation,
and the app only applies a draft or a plan after the user approves it.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, TypeVar

from llm import LLM, LLMError
from pawpal_system import Owner, Priority, Schedule, Task, fmt
from rag import Retriever, describe_task, format_hits

T = TypeVar("T")

TIME_FORMATS = ("%H:%M", "%I:%M %p", "%I %p")  # 18:30, 6:30 PM, 6 PM
NAMED_TIMES = {"noon": time(12, 0), "midnight": time(0, 0)}


def parse_time(value: Any) -> time | None:
    """Parse "18:30", "6:30pm", "6 pm", or "noon" into a time; empty, null, or "flexible" means no fixed time."""
    text = "" if value is None else str(value).strip().lower()
    if text in ("", "null", "none", "flexible"):
        return None
    if text in NAMED_TIMES:
        return NAMED_TIMES[text]
    normalized = re.sub(r"\s*([ap])\.?m\.?$", r" \1m", text).upper()  # "6:30p.m." -> "6:30 PM"
    for pattern in TIME_FORMATS:
        try:
            return datetime.strptime(normalized, pattern).time()
        except ValueError:
            pass
    raise ValueError(f"time must be HH:MM, got {value!r}")


def with_fallback(llm: LLM | None, ai_call: Callable[[LLM], T], rules_call: Callable[[], T]) -> tuple[T, str | None]:
    """Run `ai_call(llm)`, or the rule-based `rules_call()` when there's no model or the model fails.

    Returns (result, note), where note tells the user why the built-in rules were used (None if the AI answered).
    """
    if llm is None:
        return rules_call(), "AI is off, so PawPal+'s built-in rules handled this."
    try:
        return ai_call(llm), None
    except LLMError as e:
        return rules_call(), f"The AI couldn't answer ({e}), so PawPal+'s built-in rules handled this."


@dataclass
class TaskDraft:
    """A task proposed by the AI, waiting for the user to confirm it."""

    pet_name: str
    title: str
    duration_minutes: int
    priority: str = "medium"
    preferred_time: time | None = None
    frequency: str = "daily"
    rationale: str = ""  # why the AI suggested it (task ideas only)
    not_before: time | None = None
    done_by: time | None = None

    def to_task(self, start_date: date | None = None) -> Task:
        """Build the real Task, which raises ValueError if any field is invalid."""
        return Task(
            self.title,
            self.duration_minutes,
            self.priority,
            preferred_time=self.preferred_time,
            frequency=self.frequency,
            start_date=start_date or date.today(),
            not_before=self.not_before,
            done_by=self.done_by,
        )


def draft_from_json(data: Any, owner: Owner) -> TaskDraft:
    """Turn model output into a TaskDraft, raising ValueError if it names an unknown pet or fails Task validation."""
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object describing one task")
    pets = {p.name.lower(): p.name for p in owner.pets}
    pet_name = str(data.get("pet_name") or "").strip()
    if not pet_name and len(owner.pets) == 1:
        pet_name = owner.pets[0].name
    if pet_name.lower() not in pets:
        raise ValueError(f"unknown pet {pet_name!r}; pets are {', '.join(p.name for p in owner.pets)}")
    try:
        duration = int(data.get("duration_minutes"))
    except (TypeError, ValueError):
        raise ValueError("duration_minutes must be a whole number") from None
    draft = TaskDraft(
        pets[pet_name.lower()],
        str(data.get("title") or "").strip(),
        duration,
        str(data.get("priority") or "medium").lower(),
        parse_time(data.get("preferred_time")),
        str(data.get("frequency") or "daily").lower(),
        str(data.get("rationale") or "").strip(),
        parse_time(data.get("not_before")),
        parse_time(data.get("done_by")),
    )
    if not draft.title:
        raise ValueError("the task needs a title")
    draft.to_task()  # run Task's validation now, so a bad draft fails before the user reviews it
    return draft


TASK_FIELDS = """Task fields:
- "pet_name": one of the owner's pets
- "title": short task name, e.g. "Morning walk"
- "duration_minutes": whole number; estimate a typical duration if none is given
- "priority": "low", "medium", "high", or "critical" (critical tasks, like medication, are never bumped for other tasks)
- "preferred_time": "HH:MM" (24-hour) only if an exact clock time is given ("at 7:30"), otherwise null
- "not_before": "HH:MM" if it must start after a time ("after 17:00"), otherwise null
- "done_by": "HH:MM" if it must be finished by a time ("before 9am", "by noon"), otherwise null
- "frequency": "daily" or "weekly\""""

PARSE_SYSTEM = f"""You turn a pet owner's description of a care task into structured fields.
{TASK_FIELDS}
Use "critical" for words like "critical", "never skip", or "essential", "high" for "important" or "must",
"low" for "if there's time", and "medium" otherwise.
Reply with one JSON object containing exactly these fields."""


def parse_task(llm: LLM, text: str, owner: Owner) -> TaskDraft:
    """Draft a task from a plain-English description. The user reviews the draft before it is added."""
    if not owner.pets:
        raise ValueError("add a pet before adding tasks")
    pets = ", ".join(f"{p.name} ({p.species})" for p in owner.pets)
    return draft_from_json(llm.generate_json(PARSE_SYSTEM, f"Pets: {pets}\n\nDescription: {text}"), owner)


IDEAS_SYSTEM = f"""You suggest care tasks a pet is missing, based on its species, age, notes, the owner's guidelines, and recent history.
Do not repeat tasks the pet already has. Prefer tasks that address a guideline or a gap in the history.
{TASK_FIELDS}
- "rationale": one sentence on why this pet needs it; cite a source in [brackets] if you used one
Reply with JSON: {{"ideas": [<task>, ...]}}"""


def suggest_tasks(llm: LLM, owner: Owner, pet, retriever: Retriever, day: date, count: int = 3) -> list[TaskDraft]:
    """Suggest up to `count` new tasks for `pet`. Invalid ideas and ones the pet already has are dropped."""
    current = [t for t in pet.tasks if not t.is_completed(day)]
    query = " ".join([pet.species, pet.notes, *pet.guidelines, *(t.title for t in current)])
    prompt = (
        f"Pet: {pet.name}, a {pet.age}-year-old {pet.species}. Notes: {pet.notes or 'none'}\n\n"
        f"Current tasks:\n" + ("\n".join(f"- {describe_task(t, day)}" for t in current) or "- (none)")
        + f"\n\nGuidelines and history:\n{format_hits(retriever.search(query, pet_name=pet.name, k=8))}\n\n"
        f"Suggest up to {count} tasks."
    )
    reply = llm.generate_json(IDEAS_SYSTEM, prompt)
    seen = {t.title.lower() for t in current}
    ideas = []
    for item in reply.get("ideas", []) if isinstance(reply, dict) else []:
        try:
            draft = draft_from_json({**item, "pet_name": pet.name}, owner) if isinstance(item, dict) else None
        except ValueError:
            continue
        if draft and draft.title.lower() not in seen:
            seen.add(draft.title.lower())
            ideas.append(draft)
    return ideas[:count]


EXPLAIN_SYSTEM = """You explain a pet care schedule to its owner in friendly, plain language.
Use only the facts provided. Never change times or invent tasks.
Mention anything that was skipped or moved and why, and give at most two concrete suggestions.
Keep it under 150 words."""


def explain_schedule(llm: LLM, schedule: Schedule, retriever: Retriever) -> str:
    """Rewrite schedule.explain() for the owner, adding relevant history (e.g. a task skipped on earlier days)."""
    flagged = [t for t in schedule.reasons if t in schedule.skipped or t.is_overdue(schedule.day) or t in schedule.conflicts]
    history = retriever.search(" ".join(t.label for t in flagged), source="history", k=5) if flagged else []
    prompt = f"Schedule:\n{schedule.explain()}\n\nRelated history:\n{format_hits(history)}"
    return llm.generate_text(EXPLAIN_SYSTEM, prompt)


# ---- Planning agent ----

def find_task(owner: Owner, pet_name: Any, title: Any, day: date) -> Task:
    """Return the task with this title that is due on `day` for this pet (names compared case-insensitively)."""
    pet = next((p for p in owner.pets if p.name.lower() == str(pet_name).strip().lower()), None)
    if pet is None:
        raise ValueError(f"unknown pet {pet_name!r}")
    due = pet.due_tasks(day)
    task = next((t for t in due if t.title.lower() == str(title).strip().lower()), None)
    if task is None:
        titles = ", ".join(t.title for t in due) or "none"
        raise ValueError(f"{pet.name} has no task titled {title!r} due on {day}; due tasks: {titles}")
    return task


def apply_change(owner: Owner, change: dict, day: date) -> str:
    """Apply one agent change ({"tool": ..., "args": {...}}) to `owner` and describe it. Raises ValueError if invalid."""
    tool, args = change["tool"], change["args"]
    if tool == "set_available_time":
        start = parse_time(args.get("day_start")) or owner.day_start
        minutes = int(args.get("available_minutes", owner.available_minutes))
        owner.set_window(start, minutes)
        return f"Available time set to {fmt(start)}-{fmt(owner.day_end)} ({minutes} min). This applies to every day."
    if tool == "add_task":
        draft = draft_from_json(args, owner)
        pet = next(p for p in owner.pets if p.name == draft.pet_name)
        pet.add_task(draft.to_task(day))
        return f"Add {draft.pet_name}: {draft.title} ({draft.duration_minutes} min, {draft.priority})."
    if tool == "update_task":
        task = find_task(owner, args.get("pet_name"), args.get("title"), day)
        updates = {}
        if "priority" in args:
            updates["priority"] = args["priority"]
        for name in ("preferred_time", "not_before", "done_by"):
            if name in args:
                updates[name] = parse_time(args[name])
        for name in ("duration_minutes", "after_gap_minutes"):
            if name in args:
                updates[name] = int(args[name])
        if "after" in args:
            updates["after"] = str(args["after"] or "")
        if not updates:
            raise ValueError(
                "update_task needs priority, preferred_time, duration_minutes, not_before, done_by, after, or after_gap_minutes"
            )
        task.update(**updates)  # validated on a copy first, so invalid values are rejected without a partial change
        details = ", ".join(
            f"{name} → {fmt(value) if isinstance(value, time) else 'none' if value in (None, '') else getattr(value, 'name', value)}"
            for name, value in ((n, getattr(task, n)) for n in updates)
        )
        return f"Change {task.label}: {details.lower()}."
    if tool == "defer_task":
        task = find_task(owner, args.get("pet_name"), args.get("title"), day)
        task.start_date = day + timedelta(days=1)
        return f"Defer {task.label} to {task.start_date:%a %b %d}."
    raise ValueError(f"unknown change {tool!r}")


def apply_changes(owner: Owner, changes: list[dict], day: date) -> list[str]:
    """Apply an approved plan's changes to the real owner. All are tried on a copy first, so it's all or nothing."""
    trial = copy.deepcopy(owner)
    for change in changes:
        apply_change(trial, change, day)
    return [apply_change(owner, change, day) for change in changes]


def plan_warnings(schedule: Schedule, changes: list[dict]) -> list[str]:
    """Automated checks on a proposed plan, shown to the user next to it."""
    warnings = [f"{c['description']} It won't be in this day's plan." for c in changes if c["tool"] == "defer_task"]
    warnings += [
        f"{t.priority.name.title()}-priority task {t.label} isn't in the plan: {schedule.reasons[t]}."
        for t in schedule.skipped
        if t.priority >= Priority.HIGH
    ]
    if schedule.has_conflicts():
        warnings.append("Some planned tasks overlap.")
    return warnings


@dataclass
class AgentStep:
    """One action the agent took and what it got back."""

    turn: int  # which model call it came from
    thought: str
    tool: str
    args: dict
    result: str


@dataclass
class PlanProposal:
    """A plan the agent built on a copy of the owner's data, waiting for the user's approval."""

    goal: str
    day: date
    changes: list[dict]  # {"tool", "args", "description"}, replayed by apply_changes on approval
    schedule: Schedule
    summary: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    turns: int = 0  # model calls used


AGENT_SYSTEM = """You are PawPal+'s planning agent. You prepare a proposed plan for one day; the owner reviews it,
and nothing you do is final until they approve it.
Each prompt shows the owner's guidelines, relevant history, the tasks due, and the current schedule, refreshed after every change.
Reply with ONE JSON object: {"thought": "<short reasoning>", "actions": [{"tool": "<name>", "args": {...}}, ...]}
You may send several actions in one reply; they run in order.
Tools:
- set_available_time {"day_start": "HH:MM", "available_minutes": number}
- update_task {"pet_name", "title", and any of "priority" ("low"/"medium"/"high"/"critical"), "preferred_time" ("HH:MM" or "flexible"),
  "duration_minutes", "not_before" / "done_by" ("HH:MM" or null: the window the task must fit in),
  "after" (title of the same pet's task that must come first, or ""), "after_gap_minutes"}
- add_task {"pet_name", "title", "duration_minutes", "priority", "preferred_time", "frequency" ("daily"/"weekly")}
- defer_task {"pet_name", "title"}: move a due task to the next day
- search_guidelines {"query": "...", "pet_name": optional}: search older task history
- finish {"summary": "<what you changed and why, written to the owner>"}
Rules:
- The scheduler decides the times; you only change settings. A task that doesn't fit bumps strictly
  lower-priority tasks, and overdue tasks gain priority each day, so raising a priority is often enough.
- finish only takes effect in a reply with no changes, so you see the schedule your changes produce before finishing.
- Follow the owner's guidelines and cite them like [Mochi guideline 1] in the summary.
- Make the fewest changes that meet the goal. If the goal is already met, reply with only finish."""

CHANGE_TOOLS = {"set_available_time", "add_task", "update_task", "defer_task"}
MAX_ACTIONS_PER_TURN = 8


class PlanningAgent:
    """Plans a day by calling PawPal+'s own methods as tools, on a copy of the owner's data.

    To stay within free-tier rate limits, every prompt already contains the due tasks, the current schedule,
    the guidelines, and relevant history, and one reply can carry several actions. A typical plan takes two
    model calls: one to make changes and one to finish after seeing the result.
    """

    def __init__(self, llm: LLM, owner: Owner, day: date, retriever: Retriever, max_turns: int = 4):
        self.llm, self.owner, self.day, self.retriever, self.max_turns = llm, owner, day, retriever, max_turns

    def run(self, goal: str) -> PlanProposal:
        """Run the plan → act → observe loop until the model finishes or max_turns is reached.

        If the model fails on the first turn, the LLMError is raised so the caller can fall back. If it fails
        later, the changes made so far are returned as a proposal with a warning.
        """
        work = copy.deepcopy(self.owner)
        history = format_hits(self.retriever.search(
            " ".join([goal, *(t.title for t in work.all_tasks())]), source="history", k=5
        ))
        steps, changes, summary, warnings = [], [], "", []
        for turn in range(1, self.max_turns + 1):
            try:
                reply = self.llm.generate_json(AGENT_SYSTEM, self._prompt(work, goal, history, steps))
            except LLMError as e:
                if turn == 1:
                    raise
                turn -= 1  # this call produced nothing
                warnings.append(f"The AI stopped responding after {turn} turn(s) ({e}). This is its unfinished plan.")
                break
            reply = reply if isinstance(reply, dict) else {}
            thought = str(reply.get("thought") or "")
            actions = reply.get("actions") if isinstance(reply.get("actions"), list) else [reply] if "tool" in reply else []
            if not actions:
                steps.append(AgentStep(turn, thought, "", {}, "error: reply had no actions"))
                continue
            changed, finish = False, None
            for action in actions[:MAX_ACTIONS_PER_TURN]:
                action = action if isinstance(action, dict) else {}
                tool, args = str(action.get("tool") or ""), action.get("args") or {}
                if not isinstance(args, dict):
                    steps.append(AgentStep(turn, thought, tool, {}, "error: args must be an object"))
                elif tool == "finish":
                    finish = args
                else:
                    before = len(changes)
                    steps.append(AgentStep(turn, thought, tool, args, self._call_tool(work, tool, args, changes)))
                    changed = changed or len(changes) > before
            if finish is not None and changed:
                steps.append(AgentStep(turn, thought, "finish", finish, "not yet: review the schedule your changes produced, then finish"))
            elif finish is not None:
                summary = str(finish.get("summary") or "").strip()
                break
        else:
            warnings.append(f"The agent stopped after {self.max_turns} turns without finishing. Review its changes carefully.")
        schedule = work.create_schedule(self.day)
        return PlanProposal(
            goal, self.day, changes, schedule, summary, steps,
            warnings + plan_warnings(schedule, changes), turns=turn,
        )

    def _prompt(self, work: Owner, goal: str, history: str, steps: list[AgentStep]) -> str:
        """Everything the agent needs for its next reply, so it never spends a call just to look things up."""
        guidelines = "\n".join(
            f"- [{p.name} guideline {i}] {g}" for p in work.pets for i, g in enumerate(p.guidelines, 1)
        ) or "- (none)"
        due = "\n".join(f"- {describe_task(t, self.day)}" for p in work.pets for t in p.due_tasks(self.day)) or "- (none)"
        lines = [
            f"Plan day: {self.day:%A %b %d %Y}",
            f"Pets: {', '.join(f'{p.name} ({p.species}, {p.age})' for p in work.pets)}",
            f"\nGoal from the owner:\n{goal}",
            f"\nOwner's guidelines:\n{guidelines}",
            f"\nRelevant history:\n{history}",
            f"\nTasks due:\n{due}",
            f"\nCurrent schedule:\n{work.create_schedule(self.day).explain()}",
        ]
        if steps:
            lines.append("\nYour previous actions:")
            lines += [f"Turn {s.turn}: {s.tool} {json.dumps(s.args)} -> {s.result}" for s in steps]
        lines.append("\nReply with your next actions.")
        return "\n".join(lines)

    def _call_tool(self, work: Owner, tool: str, args: dict, changes: list[dict]) -> str:
        """Run one tool against the working copy. Errors are returned as text so the agent can correct itself."""
        try:
            if tool == "search_guidelines":
                return format_hits(self.retriever.search(str(args.get("query") or ""), pet_name=args.get("pet_name") or None, k=5))
            if tool in CHANGE_TOOLS:
                change = {"tool": tool, "args": args}
                change["description"] = apply_change(work, change, self.day)
                changes.append(change)
                return f"done: {change['description']}"
            return f"error: unknown tool {tool!r}"
        except (TypeError, ValueError) as e:
            return f"error: {e}"
