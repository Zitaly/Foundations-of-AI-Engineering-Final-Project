"""Rule-based versions of the AI features, used when there's no model or the model fails.

They understand far less than the model, but they're instant, free, and deterministic, and their
output goes through the same review steps as the AI's (drafts are confirmed, plans are approved).
"""

from __future__ import annotations

import copy
import re
from datetime import date, time

from ai_assistant import PlanProposal, TaskDraft, apply_change, draft_from_json, parse_time, plan_warnings
from pawpal_system import Owner, Pet, Priority, Schedule, Task, add_minutes, fmt
from rag import GuidelineIssue, Retriever, tokenize

# Words that name the same care activity, so "food" in a guideline matches a "Breakfast" task.
CONCEPTS = {
    "walk": {"walk", "walked", "walking", "stroll", "hike", "jog", "run", "exercise"},
    "food": {"food", "feed", "feeding", "fed", "meal", "breakfast", "lunch", "dinner", "supper", "eat", "eating", "kibble"},
    "medication": {"medication", "medicine", "meds", "med", "pill", "dose", "insulin", "drop", "tablet"},
    "water": {"water", "drink"},
    "brush": {"brush", "brushed", "brushing"},  # its own concept, so "brush teeth" and "brush coat" stay different
    "groom": {"groom", "groomed", "grooming", "comb", "coat", "fur"},
    "teeth": {"teeth", "tooth", "dental"},
    "play": {"play", "played", "playtime", "fetch", "toy", "toys", "game", "enrichment"},
    "litter": {"litter"},
    "bath": {"bath", "bathe", "bathed", "wash"},
    "nail": {"nail", "claw", "trim", "trimmed"},
    "train": {"train", "trained", "training", "trick"},
    "clean": {"clean", "cleaned", "cleaning", "cage", "tank", "habitat", "enclosure"},
}
# Words that carry no meaning about which task a phrase is about.
FILLER = {
    "make", "sure", "still", "happen", "happens", "get", "gets", "done", "given", "always", "need", "needs",
    "only", "today", "my", "his", "her", "their", "i", "we", "you", "time", "minute", "min", "hour", "day",
    "week", "daily", "weekly", "least", "most", "no", "not", "never", "than", "later", "after", "before", "but",
}


def concepts(text: str) -> set[str]:
    """Care activities named in `text`, e.g. {"food", "medication"}. Other meaningful words count as themselves."""
    found = set()
    for word in tokenize(text):
        if word in FILLER or word.isdigit():
            continue
        found.add(next((name for name, words in CONCEPTS.items() if word in words), word))
    return found


def activities(text: str) -> set[str]:
    """Only the known care activities named in `text`."""
    return concepts(text) & CONCEPTS.keys()


def same_task(phrase: str, title: str) -> bool:
    """True if one names a narrower version of the other: "walks" ~ "Morning walk", "medication and food" ~ "Breakfast"."""
    a, b = concepts(phrase), concepts(title)
    return bool(a and b) and (a <= b or b <= a or bool(activities(title)) and activities(title) <= a)


def matching_tasks(tasks: list[Task], phrase: str) -> list[Task]:
    """Tasks that `phrase` refers to."""
    return [t for t in tasks if same_task(phrase, t.title)]


# ---- Typing a task ----

# (concept, title, typical minutes, default priority), most specific first.
ACTIVITIES = [
    ("medication", "Medication", 5, "critical"),
    ("teeth", "Brush teeth", 5, "medium"),
    ("litter", "Clean litter box", 10, "high"),
    ("food", "Feeding", 10, "high"),
    ("water", "Fresh water", 5, "high"),
    ("walk", "Walk", 30, "medium"),
    ("groom", "Brush coat", 15, "low"),
    ("nail", "Nail trim", 15, "low"),
    ("bath", "Bath", 30, "low"),
    ("play", "Play session", 15, "medium"),
    ("train", "Training", 15, "medium"),
    ("clean", "Clean habitat", 20, "medium"),
]
MEALS = ("breakfast", "lunch", "dinner", "supper")
TIMES_OF_DAY = ("morning", "afternoon", "evening", "night")

CLOCK = re.compile(r"\b(\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?|\d{1,2}:\d{2}|noon|midnight)", re.I)
DURATION = re.compile(r"\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m)\b", re.I)
WEEKLY = re.compile(r"\b(weekly|every week|each week|once a week|twice a week|a week|per week|on (mon|tues|wednes|thurs|fri|satur|sun)days?)\b", re.I)
CRITICAL_WORDS = re.compile(r"\b(critical|never skip|essential|vital|must not (be )?(skipped|missed))\b", re.I)
HIGH_WORDS = re.compile(r"\b(important|must|high priority|urgent|has to|have to|need to)\b", re.I)
LOW_WORDS = re.compile(r"\b(if (there'?s|there is|i have) time|optional|low priority|when possible|if possible|nice to have)\b", re.I)


def find_minutes(text: str) -> int | None:
    """Minutes mentioned in `text`: "30 min", "1.5 hours", "half an hour", "an hour"."""
    if re.search(r"\bhalf an? hour\b", text, re.I):
        return 30
    if match := DURATION.search(text):
        amount, unit = float(match.group(1)), match.group(2).lower()
        return round(amount * 60) if unit.startswith("h") else round(amount)
    if re.search(r"\b(an|one) hour\b", text, re.I):
        return 60
    return None


def find_times(text: str) -> tuple[time | None, time | None, time | None]:
    """(fixed time, not before, done by) from phrases like "at 7:30", "after 5pm", "before 9am", "by noon"."""
    at = not_before = done_by = None
    for match in CLOCK.finditer(text):
        word = re.search(r"(\w+)\s*$", text[:match.start()])
        prefix = word.group(1).lower() if word else ""
        value = parse_time(match.group(1))
        if prefix in ("before", "by", "until", "than"):
            done_by = done_by or value
        elif prefix in ("after", "from"):
            not_before = not_before or value
        else:
            at = at or value
    return at, not_before, done_by


def parse_task_rules(text: str, owner: Owner) -> TaskDraft:
    """Draft a task from a plain-English description using keyword rules. Raises ValueError if it can't."""
    if not owner.pets:
        raise ValueError("add a pet before adding tasks")
    pet = next((p for p in owner.pets if re.search(rf"\b{re.escape(p.name)}\b", text, re.I)), None)
    if pet is None and len(owner.pets) == 1:
        pet = owner.pets[0]
    if pet is None:
        raise ValueError(f"say which pet this is for ({', '.join(p.name for p in owner.pets)})")

    lowered = text.lower()
    found = concepts(text)
    activity = next((a for a in ACTIVITIES if a[0] in found), None)
    meal = next((m for m in MEALS if m in lowered), None)
    part_of_day = next((p for p in TIMES_OF_DAY if p in lowered), None)
    if meal:
        title = meal.capitalize()
    elif activity:
        title = f"{part_of_day.capitalize()} {activity[1].lower()}" if part_of_day else activity[1]
    else:
        leftover = re.sub(r"\b(half an?|an|one) hour\b", " ", text, flags=re.I)
        leftover = CLOCK.sub(" ", DURATION.sub(" ", WEEKLY.sub(" ", leftover)))
        leftover = re.sub(rf"\b{re.escape(pet.name)}('s)?\b", " ", leftover, flags=re.I)
        skip = FILLER | {"for", "at", "every", "it's", "a", "the", "and", "by", "until", "from"}
        words = [w for w in re.findall(r"[A-Za-z']+", leftover) if w.lower() not in skip]
        if not words:
            raise ValueError("couldn't tell what the task is; try e.g. \"Walk Mochi for 30 minutes at 7:30\"")
        title = " ".join(words[:4]).capitalize()

    if CRITICAL_WORDS.search(text):
        priority = "critical"
    elif HIGH_WORDS.search(text):
        priority = "high"
    elif LOW_WORDS.search(text):
        priority = "low"
    else:
        priority = activity[3] if activity else "medium"

    at, not_before, done_by = find_times(text)
    return draft_from_json({
        "pet_name": pet.name,
        "title": title,
        "duration_minutes": find_minutes(text) or (activity[2] if activity else 15),
        "priority": priority,
        "preferred_time": fmt(at) if at else None,
        "not_before": fmt(not_before) if not_before else None,
        "done_by": fmt(done_by) if done_by else None,
        "frequency": "weekly" if WEEKLY.search(text) else "daily",
    }, owner)


# ---- Task ideas ----

# (title, minutes, priority, frequency, why) per species.
CARE_LIBRARY = {
    "dog": [
        ("Breakfast", 10, "high", "daily", "Regular meals keep energy and digestion steady."),
        ("Dinner", 10, "high", "daily", "Regular meals keep energy and digestion steady."),
        ("Morning walk", 30, "medium", "daily", "Dogs need daily exercise and time outside."),
        ("Evening walk", 30, "medium", "daily", "A second walk helps with exercise and toilet breaks."),
        ("Brush teeth", 5, "medium", "daily", "Daily brushing helps prevent dental disease."),
        ("Play session", 15, "medium", "daily", "Play burns energy and builds your bond."),
        ("Brush coat", 15, "low", "weekly", "Weekly brushing reduces shedding and mats."),
        ("Nail trim", 15, "low", "weekly", "Short nails prevent discomfort and joint strain."),
    ],
    "cat": [
        ("Breakfast", 10, "high", "daily", "Regular meals keep energy and digestion steady."),
        ("Dinner", 10, "high", "daily", "Regular meals keep energy and digestion steady."),
        ("Clean litter box", 10, "high", "daily", "Cats may avoid a dirty litter box."),
        ("Fresh water", 5, "high", "daily", "Cats drink more when their water is fresh."),
        ("Play session", 15, "medium", "daily", "Indoor cats need play for exercise and enrichment."),
        ("Brush coat", 10, "low", "weekly", "Brushing reduces hairballs and mats."),
        ("Nail trim", 10, "low", "weekly", "Trimmed claws protect furniture and paws."),
    ],
    "other": [
        ("Feeding", 10, "high", "daily", "Most pets need food on a steady schedule."),
        ("Fresh water", 5, "high", "daily", "Clean water every day prevents illness."),
        ("Health check", 5, "medium", "daily", "A quick daily look catches problems early."),
        ("Clean habitat", 20, "medium", "weekly", "A clean habitat prevents disease and odor."),
    ],
}
SENIOR_AGE = {"dog": 8, "cat": 10}
SENIOR_IDEA = ("Mobility check", 10, "medium", "weekly", "Older pets benefit from regular weight and joint checks.")


def covered(title: str, existing: list[str]) -> bool:
    """True if a task with this title, or a broader or narrower version of it, already exists."""
    words = set(tokenize(title))
    return any(words <= set(tokenize(e)) or set(tokenize(e)) <= words for e in existing)


def suggest_tasks_rules(owner: Owner, pet: Pet, retriever: Retriever, day: date, count: int = 3) -> list[TaskDraft]:
    """Suggest standard care tasks the pet doesn't have yet; ideas that match one of its guidelines come first."""
    current = [t.title for t in pet.tasks if not t.is_completed(day)]
    library = list(CARE_LIBRARY.get(pet.species, CARE_LIBRARY["other"]))
    if pet.age >= SENIOR_AGE.get(pet.species, 99):
        library.append(SENIOR_IDEA)
    ideas = []
    for title, minutes, priority, frequency, why in library:
        if covered(title, current):
            continue
        hits = [
            hit for hit in retriever.search(title, pet_name=pet.name, source="guideline", k=3)
            if activities(title) and activities(title) <= activities(hit.text)
        ]
        rationale = f"{why} Matches [{hits[0].ref}]." if hits else why
        draft = draft_from_json({
            "pet_name": pet.name, "title": title, "duration_minutes": minutes,
            "priority": priority, "frequency": frequency, "rationale": rationale,
        }, owner)
        ideas.append((not hits, draft))  # guideline matches sort first; otherwise library order
    ideas.sort(key=lambda pair: pair[0])
    return [draft for _, draft in ideas[:count]]


# ---- Explaining a schedule ----

def explain_schedule_rules(schedule: Schedule) -> str:
    """A short plain-language summary of the schedule, with at most two concrete suggestions."""
    owner, day = schedule.owner, schedule.day
    if not schedule.planned and not schedule.skipped:
        return f"No tasks are due on {day:%A, %b %d}."
    parts = []
    if schedule.planned:
        first_start, (last_start, last_task) = schedule.planned[0][0], schedule.planned[-1]
        end = add_minutes(last_start, last_task.duration_minutes)
        count = len(schedule.planned)
        parts.append(
            f"{count} task{'s are' if count != 1 else ' is'} planned between {fmt(first_start)} and {fmt(end)}, "
            f"using {schedule.total_minutes()} of your {owner.available_minutes} available minutes."
        )
    overdue = [t.label for _, t in schedule.planned if t.is_overdue(day)]
    if overdue:
        parts.append(f"Overdue and given extra priority: {', '.join(overdue)}.")
    if schedule.conflicts:
        parts.append("Time clashes: " + "; ".join(schedule.conflict_messages()) + ".")
    if schedule.skipped:
        parts.append("Didn't fit: " + "; ".join(f"{t.label} ({schedule.reasons[t]})" for t in schedule.skipped) + ".")

    suggestions = []
    outside = [t for t in schedule.skipped if "outside available time" in schedule.reasons[t]]
    if outside:
        suggestions.append(
            f"Move {outside[0].label}'s fixed time inside {fmt(owner.day_start)}-{fmt(owner.day_end)}, or widen your available time."
        )
    room_needed = sum(t.duration_minutes for t in schedule.skipped if t not in outside)
    free = owner.available_minutes - schedule.total_minutes()
    if room_needed > free:
        suggestions.append(f"About {room_needed - free} more available minutes would fit the skipped tasks.")
    if critical := [t.label for t in schedule.skipped if t.priority == Priority.CRITICAL]:
        suggestions.insert(0, f"Make room for critical task {critical[0]} first.")
    if suggestions:
        parts.append("Suggestions: " + " ".join(suggestions[:2]))
    return "\n\n".join(parts)


# ---- Checking guidelines ----

TIME_TEXT = r"(\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?|noon|midnight)"
AMOUNT = re.compile(r"at least (\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?)\s+(?:of\s+)?(.+?)(?:\s+(?:a|per|each|every)\s+day|\s+daily)?\.?$", re.I)
BEFORE = re.compile(rf"^(.+?)\s+(?:before|by|no later than)\s+{TIME_TEXT}\.?$", re.I)
NOT_AFTER = re.compile(rf"^(?:no\s+(.+?)|(.+?)\s+(?:not|never))\s+after\s+{TIME_TEXT}\.?$", re.I)
AFTER = re.compile(rf"^(.+?)\s+after\s+{TIME_TEXT}\.?$", re.I)
FREQUENCY = re.compile(r"^(.+?)\s+(daily|every day|once a day|twice a day|weekly|once a week|twice a week|(\d+) times (?:a|per) week)\.?$", re.I)
WEEKLY_COUNTS = {"daily": 7, "every day": 7, "once a day": 7, "twice a day": 14, "weekly": 1, "once a week": 1, "twice a week": 2}


def _clean_subject(subject: str) -> str:
    """Drop helper verbs, e.g. "Medication must be given" -> "Medication"."""
    return re.sub(r"\s+(must|should|has to|have to|needs? to)?\s*(be\s+)?(given|done|fed|served|happen)?\s*$", "", subject.strip(), flags=re.I) or subject


def _check_one(guideline: str, tasks: list[Task], starts: dict[Task, time]) -> tuple[str, str] | None | bool:
    """(issue, suggestion) if `guideline` is broken, None if it's met, or False if no rule understands it."""
    text = guideline.strip()
    if match := AMOUNT.search(text):
        needed = float(match.group(1)) * (60 if match.group(2).lower().startswith("h") else 1)
        subject = match.group(3)
        daily = [t for t in matching_tasks(tasks, subject) if t.frequency == "daily"]
        total = sum(t.duration_minutes for t in daily)
        if total < needed:
            return f"Only {total} min of {subject} a day is planned, not {needed:g}.", f"Add about {needed - total:g} more minutes of {subject}."
        return None
    if (match := NOT_AFTER.search(text)) and concepts(match.group(1) or match.group(2)):
        subject, limit = match.group(1) or match.group(2), parse_time(match.group(3))
        late = [t for t in matching_tasks(tasks, subject) if (s := starts.get(t, t.preferred_time)) and s >= limit]
        if late:
            return f"{late[0].label} is at {fmt(starts.get(late[0], late[0].preferred_time))}, after {fmt(limit)}.", f"Move it before {fmt(limit)}."
        return None
    for pattern, wants_before in ((BEFORE, True), (AFTER, False)):
        if (match := pattern.search(text)) and concepts(_clean_subject(match.group(1))):
            subject, limit = _clean_subject(match.group(1)), parse_time(match.group(2))
            found = matching_tasks(tasks, subject)
            if not found:
                word = "before" if wants_before else "after"
                return f"No task covers {subject.lower()}.", f"Add a {subject.lower()} task fixed {word} {fmt(limit)}."
            for task in found:
                start = starts.get(task, task.preferred_time)
                if start is not None and (start >= limit if wants_before else start < limit):
                    word = "before" if wants_before else "at or after"
                    setting = "done-by" if wants_before else "not-before"
                    return f"{task.label} is at {fmt(start)}, not {word} {fmt(limit)}.", f"Give {task.title} a {setting} time of {fmt(limit)}."
            return None
    if (match := FREQUENCY.search(text)) and concepts(_clean_subject(match.group(1))):
        subject, phrase = _clean_subject(match.group(1)), match.group(2).lower()
        needed = int(match.group(3)) if match.group(3) else WEEKLY_COUNTS[phrase]
        per_week = sum(7 if t.frequency == "daily" else 1 for t in matching_tasks(tasks, subject))
        if per_week < needed:
            return f"{subject.capitalize()} happens about {per_week} time(s) a week, but the guideline asks for {phrase}.", f"Add or repeat a {subject.lower()} task to reach {phrase}."
        return None
    return False


def check_guidelines_rules(owner: Owner, day: date) -> list[GuidelineIssue]:
    """Check the guideline shapes the rules understand (amounts, before/after times, frequencies)."""
    schedule = Schedule(day, owner)  # not stored in owner.schedules, so checking doesn't add history
    schedule.generate()
    starts = {task: start for start, task in schedule.planned}
    issues = []
    for pet in owner.pets:
        tasks = [t for t in pet.tasks if not t.is_completed(day)]
        for number, guideline in enumerate(pet.guidelines, 1):
            result = _check_one(guideline, tasks, starts)
            if result:
                issues.append(GuidelineIssue(pet.name, guideline, result[0], result[1], f"{pet.name} guideline {number}"))
    return issues


def unchecked_guidelines(owner: Owner) -> list[str]:
    """Citations of guidelines the rules couldn't interpret, so the user knows they weren't checked."""
    return [
        f"{pet.name} guideline {number}"
        for pet in owner.pets
        for number, guideline in enumerate(pet.guidelines, 1)
        if _check_one(guideline, list(pet.tasks), {}) is False
    ]


# ---- Planning a day ----

KEEP = re.compile(r"\b(?:make sure|ensure|don['’]?t skip|do not skip|keep|prioriti[sz]e)\b", re.I)
DROP = re.compile(r"\b(?:skip|defer|drop|postpone|push back)\b", re.I)
START = re.compile(rf"\b(?:start|starting|from|begin|beginning)\w*\s+(?:at\s+)?{TIME_TEXT}", re.I)


def goal_phrases(goal: str) -> tuple[list[str], list[str]]:
    """Split a goal into phrases naming tasks to keep and tasks to defer.

    A clause after a comma continues the previous instruction, so "make sure meds, food and walks happen"
    keeps all three; a new sentence or "but" starts fresh.
    """
    keep, drop = [], []
    for part in re.split(r"[.;!?]|\bbut\b", goal):
        target = None
        for clause in part.split(","):
            if match := KEEP.search(clause):  # checked first, so "don't skip X" keeps X
                target, clause = keep, clause[match.end():]
            elif match := DROP.search(clause):
                target, clause = drop, clause[match.end():]
            if target is not None:
                target.append(clause)
    return keep, drop


def plan_rules(owner: Owner, day: date, goal: str) -> PlanProposal:
    """Plan a day from a goal using rules: a time budget, a start time, tasks to keep, and tasks to defer."""
    work = copy.deepcopy(owner)
    changes, notes = [], []

    def change(tool: str, args: dict) -> None:
        entry = {"tool": tool, "args": args}
        try:
            entry["description"] = apply_change(work, entry, day)
            changes.append(entry)
        except ValueError as e:
            notes.append(f"Skipped a change ({tool}): {e}")

    minutes = find_minutes(goal)
    start_match = START.search(goal)
    start = parse_time(start_match.group(1)) if start_match else None
    if minutes is not None or start is not None:
        change("set_available_time", {
            "day_start": fmt(start or work.day_start),
            "available_minutes": minutes if minutes is not None else work.available_minutes,
        })

    due = [t for p in work.pets for t in p.due_tasks(day)]
    keep_phrases, drop_phrases = goal_phrases(goal)
    deferred = {id(t) for phrase in drop_phrases for t in matching_tasks(due, phrase)}
    for task in [t for t in due if id(t) in deferred]:
        change("defer_task", {"pet_name": task.pet_name, "title": task.title})

    keep = [t for t in due if id(t) not in deferred and any(same_task(phrase, t.title) for phrase in keep_phrases)]
    for task in keep:
        if task.priority < Priority.HIGH:
            change("update_task", {"pet_name": task.pet_name, "title": task.title, "priority": "high"})
    still_skipped = set(work.create_schedule(day).skipped)
    for task in keep:
        if task in still_skipped and task.priority < Priority.CRITICAL:
            change("update_task", {"pet_name": task.pet_name, "title": task.title, "priority": "critical"})

    if changes:
        summary = "Built with PawPal+'s rules: " + " ".join(c["description"] for c in changes)
    elif minutes is None and start is None and not keep and not deferred:
        summary = (
            "The built-in rules didn't find a time budget, start time, or task names in your goal, so nothing changed. "
            "Try e.g. \"I only have 60 minutes. Make sure medication happens, skip brushing.\""
        )
    else:
        summary = "Your current settings already meet this goal, so nothing needs to change."
    schedule = work.create_schedule(day)
    return PlanProposal(goal, day, changes, schedule, summary, [], notes + plan_warnings(schedule, changes), turns=0)
