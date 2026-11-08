"""PawPal+ core classes (design in diagrams/uml.mmd)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from enum import IntEnum
from uuid import uuid4

# How many days apart each recurring task is due.
FREQUENCY_DAYS = {"daily": 1, "weekly": 7}

MINUTES_PER_DAY = 24 * 60
PACKING_NODE_LIMIT = 20_000  # search steps best-fit packing may take before keeping the best arrangement found

log = logging.getLogger("pawpal.scheduler")


class Priority(IntEnum):
    """Task importance; higher values are scheduled first."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4  # e.g. medication: never bumped, and overdue tasks never age past HIGH to compete with it


PRIORITY_NAMES = [p.name.lower() for p in Priority]  # "low", "medium", "high", "critical"


def new_id() -> str:
    """A unique id that identifies a task across saves and reloads."""
    return uuid4().hex


def add_minutes(start: time, minutes: int) -> time:
    """Return `start` shifted by `minutes` (datetime.time can't do this directly)."""
    return (datetime.combine(date.min, start) + timedelta(minutes=minutes)).time()


def minutes_between(start: time, end: time) -> int:
    """Return the number of minutes from `start` to `end` on the same day."""
    return int((datetime.combine(date.min, end) - datetime.combine(date.min, start)).total_seconds() // 60)


def fmt(t: time) -> str:
    """Format a time as HH:MM."""
    return t.strftime("%H:%M")


@dataclass(eq=False)  # identity equality: two identical-looking tasks are still different tasks
class Task:
    """One occurrence of a pet care activity, e.g. a walk, feeding, or medication.

    Completing it (via Pet.complete_task) creates the next occurrence as a new Task;
    an occurrence that isn't completed stays due (overdue) until it is.
    """

    title: str
    duration_minutes: int
    priority: Priority = Priority.MEDIUM
    preferred_time: time | None = None  # fixed start time; None means flexible
    frequency: str = "daily"  # a key of FREQUENCY_DAYS
    start_date: date = field(default_factory=date.today)  # the day this occurrence is due
    last_completed: date | None = None  # the day this occurrence was done
    not_before: time | None = None  # earliest allowed start
    done_by: time | None = None  # latest allowed finish
    after: str = ""  # title of another task of the same pet that must finish first (if both are planned)
    after_gap_minutes: int = 0  # minimum minutes between that task's end and this task's start
    pet_name: str = ""  # set by Pet.add_task
    id: str = field(default_factory=new_id)  # each occurrence gets its own id
    pet: Pet | None = field(default=None, init=False, repr=False)  # the pet holding this task; set by Pet.add_task

    def __post_init__(self) -> None:
        """Convert a string or int priority to Priority and validate duration and frequency."""
        try:
            if isinstance(self.priority, str):  # accepts "low"/.../"critical", e.g. from the Streamlit selectbox
                self.priority = Priority[self.priority.upper()]
            else:  # accepts 1-4
                self.priority = Priority(self.priority)
        except (KeyError, ValueError):
            raise ValueError(f"priority must be one of {', '.join(PRIORITY_NAMES)}, got {self.priority!r}") from None
        if self.duration_minutes <= 0:
            raise ValueError(f"duration_minutes must be positive, got {self.duration_minutes}")
        if self.frequency not in FREQUENCY_DAYS:
            raise ValueError(f"frequency must be one of {list(FREQUENCY_DAYS)}, got {self.frequency!r}")
        self._check_timing()

    def _check_timing(self) -> None:
        """Validate the time window, fixed time, and ordering rule against each other."""
        if self.not_before and self.done_by and minutes_between(self.not_before, self.done_by) < self.duration_minutes:
            raise ValueError(
                f"the window {fmt(self.not_before)}-{fmt(self.done_by)} is shorter than the task's {self.duration_minutes} min"
            )
        if self.preferred_time:
            end = minutes_between(time(0, 0), self.preferred_time) + self.duration_minutes
            if self.not_before and self.preferred_time < self.not_before:
                raise ValueError(f"fixed time {fmt(self.preferred_time)} is before its not-before time {fmt(self.not_before)}")
            if self.done_by and end > minutes_between(time(0, 0), self.done_by):
                raise ValueError(f"fixed time {fmt(self.preferred_time)} ends after its done-by time {fmt(self.done_by)}")
        self.after = (self.after or "").strip()
        if self.after.lower() == self.title.strip().lower():
            raise ValueError("a task can't come after itself")
        if self.after_gap_minutes < 0:
            raise ValueError(f"after_gap_minutes can't be negative, got {self.after_gap_minutes}")

    def update(self, **changes) -> None:
        """Change fields after validating them on a copy, so an invalid change leaves this task untouched."""
        checked = replace(self, **changes)
        for name in changes:
            setattr(self, name, getattr(checked, name))

    @property
    def label(self) -> str:
        """Display name, e.g. "Mochi: Morning walk"."""
        return f"{self.pet_name}: {self.title}" if self.pet_name else self.title

    @property
    def window_label(self) -> str:
        """The time window, e.g. "08:00-09:00", "not before 08:00", "done by 09:00", or "" if there is none."""
        if self.not_before and self.done_by:
            return f"{fmt(self.not_before)}-{fmt(self.done_by)}"
        if self.not_before:
            return f"not before {fmt(self.not_before)}"
        return f"done by {fmt(self.done_by)}" if self.done_by else ""

    @property
    def timing_label(self) -> str:
        """Short description of the window and ordering rule, e.g. "08:00-09:00, after Breakfast +30 min"."""
        parts = [self.window_label] if self.window_label else []
        if self.after:
            parts.append(f"after {self.after}" + (f" +{self.after_gap_minutes} min" if self.after_gap_minutes else ""))
        return ", ".join(parts)

    def mark_complete(self, day: date) -> None:
        """Record that this occurrence was done on `day`. Use Pet.complete_task to also create the next occurrence."""
        self.last_completed = day

    def last_occurrence(self, day: date) -> date | None:
        """Return the most recent date on or before `day` that this task recurs on, or None if it hasn't started."""
        if day < self.start_date:
            return None
        return day - timedelta(days=(day - self.start_date).days % FREQUENCY_DAYS[self.frequency])

    def next_occurrence(self, day: date) -> Task:
        """Return a new, not-yet-completed Task for the first occurrence after the one current on `day`.

        The cadence stays anchored to start_date, so a weekly task done late is still next due on its usual weekday.
        """
        current = self.last_occurrence(day) or self.start_date
        return replace(
            self, start_date=current + timedelta(days=FREQUENCY_DAYS[self.frequency]), last_completed=None, id=new_id()
        )

    def is_completed(self, day: date) -> bool:
        """Return True if this occurrence was done on or before `day`."""
        return self.last_completed is not None and self.last_completed <= day

    def is_due(self, day: date) -> bool:
        """Return True if this occurrence has started and isn't done; missed occurrences carry over."""
        return day >= self.start_date and not self.is_completed(day)

    def is_overdue(self, day: date) -> bool:
        """Return True if this occurrence is still due on `day` but was due on an earlier day."""
        return self.is_due(day) and self.start_date < day


@dataclass(eq=False)
class Pet:
    """A pet and the care tasks it needs."""

    name: str
    species: str
    age: int = 0
    notes: str = ""
    tasks: list[Task] = field(default_factory=list)
    guidelines: list[str] = field(default_factory=list)  # owner-written care rules the AI checks tasks against

    def add_task(self, task: Task) -> None:
        """Add a care task for this pet, moving it from its previous pet if it had one."""
        if task.pet is not None and task.pet is not self:
            task.pet.remove_task(task)
        task.pet = self
        task.pet_name = self.name
        if task not in self.tasks:
            self.tasks.append(task)

    def remove_task(self, task: Task) -> None:
        """Remove this exact task object from the pet. Raises ValueError if it isn't there."""
        self.tasks.remove(task)
        task.pet = None

    def complete_task(self, task: Task, day: date) -> Task:
        """Mark `task` done on `day`, add its next occurrence as a new task, and return that new task."""
        if task not in self.tasks:
            raise ValueError(f"{task.label!r} isn't one of {self.name}'s tasks")
        if task.last_completed is not None:
            raise ValueError(f"{task.label!r} was already completed on {task.last_completed}")
        task.mark_complete(day)
        next_task = task.next_occurrence(day)
        self.add_task(next_task)
        return next_task

    def due_tasks(self, day: date) -> list[Task]:
        """Return this pet's tasks that are due on `day`."""
        return [task for task in self.tasks if task.is_due(day)]

    @property
    def task_count(self) -> int:
        """Number of care tasks assigned to this pet."""
        return len(self.tasks)


@dataclass
class Owner:
    """A pet owner, their pets, and their daily constraints."""

    name: str
    available_minutes: int = 120
    day_start: time = time(8, 0)
    prefer_high_priority_first: bool = True  # False: shortest tasks first, to fit as many as possible
    max_tasks_per_day: int | None = None  # None means no limit
    overdue_boost_per_day: float = 0.5  # priority levels an overdue task gains per day late (capped at HIGH); 0 turns it off
    bump_lower_priority: bool = True  # a task that doesn't fit may displace strictly lower-priority tasks (priority-first only)
    best_fit_packing: bool = True  # search for the arrangement of flexible tasks that fits the most important ones
    pets: list[Pet] = field(default_factory=list)
    schedules: dict[date, Schedule] = field(default_factory=dict)

    def __setattr__(self, name: str, value) -> None:
        """Validate the available window whenever day_start or available_minutes is set, including in __init__."""
        if name in ("day_start", "available_minutes"):
            window = {"day_start": getattr(self, "day_start", None), "available_minutes": getattr(self, "available_minutes", None)}
            window[name] = value
            self._check_window(**window)
        super().__setattr__(name, value)

    @staticmethod
    def _check_window(day_start: time | None, available_minutes: int | None) -> None:
        """Raise ValueError if the available time is negative or doesn't end before midnight. None skips a check."""
        if available_minutes is not None and available_minutes < 0:
            raise ValueError(f"available_minutes can't be negative, got {available_minutes}")
        if day_start is not None and available_minutes is not None:
            if minutes_between(time(0, 0), day_start) + available_minutes >= MINUTES_PER_DAY:
                raise ValueError("available time must end before midnight")

    def set_window(self, day_start: time, available_minutes: int) -> None:
        """Change both window settings at once, so a valid new pair isn't rejected halfway through."""
        self._check_window(day_start, available_minutes)
        super().__setattr__("day_start", day_start)
        super().__setattr__("available_minutes", available_minutes)

    @property
    def day_end(self) -> time:
        """Time when the owner's available window closes."""
        return add_minutes(self.day_start, self.available_minutes)

    def add_pet(self, pet: Pet) -> None:
        """Add a pet to this owner. Pet names must be unique so task labels stay unambiguous."""
        if pet in self.pets:
            return
        if any(existing.name == pet.name for existing in self.pets):
            raise ValueError(f"{self.name} already has a pet named {pet.name!r}")
        self.pets.append(pet)

    def remove_pet(self, pet: Pet) -> None:
        """Remove this exact pet object from the owner. Raises ValueError if it isn't there."""
        self.pets.remove(pet)

    def all_tasks(self) -> list[Task]:
        """Return every task across all of this owner's pets."""
        return [task for pet in self.pets for task in pet.tasks]

    def filter_tasks(self, pet_name: str | None = None, completed: bool | None = None, day: date | None = None) -> list[Task]:
        """Return tasks matching every filter given (None skips that filter). `completed` is judged as of `day`, default today."""
        day = day or date.today()
        return [
            task
            for task in self.all_tasks()
            if (pet_name is None or task.pet_name == pet_name)
            and (completed is None or task.is_completed(day) == completed)
        ]

    def create_schedule(self, day: date) -> Schedule:
        """Create a Schedule for `day`, call generate(), store it (replacing any old one for that day), and return it."""
        schedule = Schedule(day, self)
        schedule.generate()
        self.schedules[day] = schedule
        return schedule


@dataclass
class Schedule:
    """A daily plan that orders an owner's tasks within their available time.

    Placement rule: tasks are ranked by effective priority (priority plus an overdue boost), with a
    task's predecessor moved ahead of it; tasks with a preferred_time are fixed at that time; a fixed
    task that clashes with a higher-ranked one (or breaks an ordering rule) moves to the nearest free
    slot; flexible tasks then fill the remaining gaps, either first-fit or arranged so the most
    important ones fit (best-fit packing). Every placement respects the task's not_before/done_by
    window and "after" rule. Finally, a task that found no room may bump strictly lower-priority tasks.
    """

    day: date
    owner: Owner = field(repr=False, compare=False)  # avoids Owner <-> Schedule recursion
    planned: list[tuple[time, Task]] = field(default_factory=list)  # (start time, task) pairs, sorted by time
    skipped: list[Task] = field(default_factory=list)
    reasons: dict[Task, str] = field(default_factory=dict)  # why each task was planned or skipped
    conflicts: dict[Task, list[Task]] = field(default_factory=dict)  # moved fixed task -> tasks at its preferred time
    _no_room: set[Task] = field(default_factory=set, init=False, repr=False, compare=False)  # skipped only for lack of space

    def generate(self) -> None:
        """Fill `planned`, `skipped`, and `reasons`, then apply the daily task limit to what actually got placed."""
        tasks = self._sort_tasks(self._collect_due_tasks())
        self._place(tasks)

        limit = self.owner.max_tasks_per_day
        if limit is not None and len(self.planned) > limit:
            # Keep the highest-ranked placed tasks and re-place only those, so the cut tasks free up their time.
            placed = {task for _, task in self.planned}
            ranked = [task for task in tasks if task in placed]
            unplaceable = [(task, self.reasons[task]) for task in self.skipped]
            self._place(ranked[:limit])
            for task, reason in unplaceable:
                self._skip(task, reason)
            for task in ranked[limit:]:
                self._skip(task, f"over the daily limit of {limit} tasks")

        for task, reason in self.reasons.items():
            if task.is_overdue(self.day):
                self.reasons[task] = f"{reason} (overdue since {task.start_date:%a %b %d})"

        if log.isEnabledFor(logging.DEBUG):
            log.debug("Schedule %s: %d planned, %d skipped, %d moved", self.day, len(self.planned), len(self.skipped), len(self.conflicts))
            for task in self.skipped:
                log.debug("Skipped %s: %s", task.label, self.reasons[task])

    def _place(self, tasks: list[Task]) -> None:
        """Clear any previous plan and place `tasks` (already ranked): fixed ones first, then flexible ones."""
        self.planned.clear()
        self.skipped.clear()
        self.reasons.clear()
        self.conflicts.clear()
        self._no_room.clear()
        flexible = self._place_fixed_tasks(tasks)
        self._fill_flexible_tasks(flexible)
        if self.owner.bump_lower_priority and self.owner.prefer_high_priority_first:  # shortest-first wants the most tasks
            self._bump_lower_priority(tasks)
        self.sort_by_time()

    def effective_priority(self, task: Task) -> float:
        """Priority plus owner.overdue_boost_per_day for each day overdue, capped at HIGH (CRITICAL stays CRITICAL)."""
        days_late = (self.day - task.start_date).days if task.is_overdue(self.day) else 0
        boosted = task.priority + days_late * self.owner.overdue_boost_per_day
        return min(boosted, max(task.priority, Priority.HIGH))

    def _priority_text(self, task: Task) -> str:
        """E.g. "high priority", or "low priority, raised to high after 4 days overdue"."""
        text = f"{task.priority.name.lower()} priority"
        raised = Priority(int(self.effective_priority(task)))
        if raised > task.priority:
            text += f", raised to {raised.name.lower()} after {(self.day - task.start_date).days} days overdue"
        return text

    def _collect_due_tasks(self) -> list[Task]:
        """Step 1: gather tasks from all pets that are due on `day`."""
        return [task for pet in self.owner.pets for task in pet.due_tasks(self.day)]

    def _sort_tasks(self, tasks: list[Task]) -> list[Task]:
        """Step 2: order by effective priority (respecting owner preferences), then by duration; overdue tasks win ties."""
        if self.owner.prefer_high_priority_first:
            ranked = sorted(tasks, key=lambda t: (-self.effective_priority(t), not t.is_overdue(self.day), t.duration_minutes))
        else:
            ranked = sorted(tasks, key=lambda t: (t.duration_minutes, -self.effective_priority(t), not t.is_overdue(self.day)))
        return self._predecessors_first(ranked)

    def _predecessor(self, task: Task, tasks: list[Task]) -> Task | None:
        """The task in `tasks` that `task` must come after: same pet, title matching task.after."""
        if not task.after:
            return None
        return next((t for t in tasks if t.pet_name == task.pet_name and t.title.lower() == task.after.lower()), None)

    def _predecessors_first(self, ranked: list[Task]) -> list[Task]:
        """Move each task's predecessor just ahead of it, so ordering rules rarely force a move later."""
        result = list(ranked)
        for _ in range(len(result)):  # enough passes for any chain; a cycle just stops changing order
            moved = False
            for task in list(result):
                before = self._predecessor(task, result)
                if before is not None and result.index(before) > result.index(task):
                    result.remove(before)
                    result.insert(result.index(task), before)
                    moved = True
            if not moved:
                break
        return result

    def _offset(self, t: time) -> int:
        """Minutes from day_start to `t` (negative if earlier)."""
        return minutes_between(self.owner.day_start, t)

    def _bounds(self, task: Task) -> tuple[int, int]:
        """The [earliest start, latest end] offsets `task` may use, from its window and planned order partners."""
        low, high = 0, self.owner.available_minutes
        if task.not_before:
            low = max(low, self._offset(task.not_before))
        if task.done_by:
            high = min(high, self._offset(task.done_by))
        for start, other in self.planned:
            if other is task or other.pet_name != task.pet_name:
                continue
            if task.after and other.title.lower() == task.after.lower():  # must start after `other` ends
                low = max(low, self._offset(start) + other.duration_minutes + task.after_gap_minutes)
            if other.after and other.after.lower() == task.title.lower():  # must end before `other` starts
                high = min(high, self._offset(start) - other.after_gap_minutes)
        return low, high

    def _order_partners(self, task: Task) -> list[Task]:
        """Planned tasks that `task` has an ordering rule with."""
        return [
            other for _, other in self.planned
            if other is not task and other.pet_name == task.pet_name
            and (task.after.lower() == other.title.lower() or other.after.lower() == task.title.lower())
        ]

    def _task_gaps(self, task: Task) -> list[list[int]]:
        """Free intervals clipped to the task's bounds (some may be shorter than the task)."""
        low, high = self._bounds(task)
        return [[max(s, low), min(e, high)] for s, e in self._free_gaps() if min(e, high) > max(s, low)]

    def _place_fixed_tasks(self, tasks: list[Task]) -> list[Task]:
        """Step 3: place tasks that have a preferred_time; skip ones outside the owner's window.

        A task that clashes with an earlier-ranked fixed task, or would break an ordering rule, is moved
        to the nearest free slot. Returns the remaining flexible tasks.
        """
        window_start, window_end = self.owner.day_start, self.owner.day_end
        flexible, clashing = [], []
        for task in tasks:
            if task.preferred_time is None:
                flexible.append(task)
                continue
            start = task.preferred_time
            end = add_minutes(start, task.duration_minutes)
            if start < window_start or end > window_end or end < start:
                self._skip(task, f"fixed at {fmt(start)}, outside available time {fmt(window_start)}-{fmt(window_end)}")
                continue
            low, high = self._bounds(task)
            out_of_order = self._offset(start) < low or self._offset(start) + task.duration_minutes > high
            if out_of_order or any(self._overlaps(start, task, other_start, other) for other_start, other in self.planned):
                clashing.append(task)  # relocate after every non-clashing fixed task has its slot
                continue
            self._plan(start, task, f"fixed at preferred time {fmt(start)}")
        for task in clashing:
            self._relocate(task)
        return flexible

    def _relocate(self, task: Task) -> None:
        """Move a clashing fixed task to the free slot whose start is nearest its preferred time, or skip it."""
        overlapping = [
            other for other_start, other in self.planned if self._overlaps(task.preferred_time, task, other_start, other)
        ]
        if overlapping:
            self.conflicts[task] = overlapping
            clashes = ", ".join(other.label for other in overlapping)
        else:  # in a free slot, but on the wrong side of an ordering rule
            clashes = "the order with " + ", ".join(other.label for other in self._order_partners(task))
        start = self._slot_for(task)
        if start is None:
            self._skip(task, f"fixed at {fmt(task.preferred_time)}, clashes with {clashes} and no other free slot fits")
            self._no_room.add(task)
            return
        self._plan(start, task, f"moved from preferred time {fmt(task.preferred_time)} to avoid {clashes}")

    def _slot_for(self, task: Task) -> time | None:
        """Where `task` fits in the current free time: the start nearest its preferred time, else the earliest gap."""
        gaps = [gap for gap in self._task_gaps(task) if gap[1] - gap[0] >= task.duration_minutes]
        if not gaps:
            return None
        if task.preferred_time is None:
            return add_minutes(self.owner.day_start, gaps[0][0])
        preferred = self._offset(task.preferred_time)
        # Within each gap that fits, the closest possible start is the preferred offset clamped into the gap.
        candidates = [min(max(preferred, gap_start), gap_end - task.duration_minutes) for gap_start, gap_end in gaps]
        return add_minutes(self.owner.day_start, min(candidates, key=lambda c: abs(c - preferred)))

    def next_available_slot(self, duration_minutes: int, after: time | None = None) -> time | None:
        """Earliest start at or after `after` (default day_start) where `duration_minutes` fits around the plan."""
        earliest = max(minutes_between(self.owner.day_start, after), 0) if after else 0
        for gap_start, gap_end in self._free_gaps():
            start = max(gap_start, earliest)
            if gap_end - start >= duration_minutes:
                return add_minutes(self.owner.day_start, start)
        return None

    def _fill_flexible_tasks(self, tasks: list[Task]) -> None:
        """Step 4: place flexible tasks in free gaps (first-fit, or best-fit packing); skip ones that don't fit."""
        if self.owner.best_fit_packing:
            self._pack(tasks)
            return
        for task in tasks:
            start = self._slot_for(task)
            if start is None:
                self._skip_no_room(task)
            else:
                self._plan(start, task, self._placed_reason(task, "placed in earliest free slot"))

    def _pack(self, tasks: list[Task]) -> None:
        """Choose where the flexible tasks go so the plan holds the most value, by branch and bound.

        Each task (in rank order) is tried at the start of every free gap it fits, or left out. The first
        arrangement tried is plain first-fit, so the result is never worse; a node budget caps the search.
        Value: with priority-first ordering, one task at a higher level outweighs any number below it;
        with shortest-first ordering, every task counts the same (fit as many as possible).
        """
        if not tasks:
            return
        if self.owner.prefer_high_priority_first:
            values = [(len(tasks) + 1) ** round(2 * self.effective_priority(t)) for t in tasks]
        else:
            values = [1] * len(tasks)
        remaining = [sum(values[i:]) for i in range(len(tasks) + 1)]
        base = list(self.planned)
        best: dict = {"value": -1, "planned": base, "first_fit": None}
        budget = [PACKING_NODE_LIMIT]

        def search(i: int, value: int) -> None:
            if budget[0] <= 0 or value + remaining[i] <= best["value"]:
                return
            budget[0] -= 1
            if i == len(tasks):
                best["value"], best["planned"] = value, list(self.planned)
                if best["first_fit"] is None:  # the first complete arrangement is the plain first-fit one
                    best["first_fit"] = {t: s for s, t in self.planned}
                return
            task = tasks[i]
            for gap_start, gap_end in self._task_gaps(task):
                if gap_end - gap_start >= task.duration_minutes:
                    self.planned.append((add_minutes(self.owner.day_start, gap_start), task))
                    search(i + 1, value + values[i])
                    self.planned.pop()
            search(i + 1, value)

        search(0, 0)
        self.planned[:] = best["planned"]
        starts = {t: s for s, t in self.planned}
        first_fit = best["first_fit"] if best["first_fit"] is not None else {}
        for task in tasks:
            if task not in starts:
                self._skip_no_room(task)
            elif first_fit.get(task) == starts[task]:
                self.reasons[task] = self._placed_reason(task, "placed in earliest free slot")
            else:
                self.reasons[task] = self._placed_reason(task, f"placed at {fmt(starts[task])} so more important tasks fit")

    def _placed_reason(self, task: Task, how: str) -> str:
        """E.g. "medium priority, placed in earliest free slot, after Mochi: Breakfast, within 08:00-09:00"."""
        parts = [self._priority_text(task), how]
        before = self._predecessor(task, [t for _, t in self.planned])
        if before is not None:
            parts.append(f"after {before.label}" + (f" +{task.after_gap_minutes} min" if task.after_gap_minutes else ""))
        if task.window_label:
            parts.append(f"within its window ({task.window_label})")
        return ", ".join(parts)

    def _skip_no_room(self, task: Task) -> None:
        """Skip a task that has no free slot big enough, saying how much room there is."""
        largest = max((e - s for s, e in self._task_gaps(task)), default=0)
        low, high = self._bounds(task)
        if (low, high) == (0, self.owner.available_minutes):
            reason = f"needs {task.duration_minutes} min, largest free slot is {largest} min"
        else:
            window = f"{fmt(add_minutes(self.owner.day_start, low))}-{fmt(add_minutes(self.owner.day_start, max(high, low)))}"
            reason = f"needs {task.duration_minutes} min within {window}, largest free slot there is {largest} min"
        self._skip(task, reason)
        self._no_room.add(task)

    def _bump_lower_priority(self, ranked: list[Task]) -> None:
        """Step 5: let each task that found no room displace strictly lower-priority planned tasks.

        Victims are removed lowest priority first until the task fits; each is then put back wherever it
        still fits (its old slot if possible), or skipped. Victims never bump others, so this always ends.
        """
        for task in [t for t in ranked if t in self._no_room]:
            rank = self.effective_priority(task)
            victims = sorted(
                (entry for entry in self.planned if self.effective_priority(entry[1]) < rank),
                key=lambda entry: (self.effective_priority(entry[1]), -ranked.index(entry[1])),
            )
            removed = []
            for entry in victims:
                self.planned.remove(entry)
                removed.append(entry)
                if self._slot_for(task) is not None:
                    break
            else:
                self.planned.extend(removed)  # even removing every lower-priority task doesn't make room
                continue

            labels = ", ".join(victim.label for _, victim in removed)
            log.debug("%s bumped %s", task.label, labels)
            self.skipped.remove(task)
            self._no_room.discard(task)
            self.conflicts.pop(task, None)
            self._plan(self._slot_for(task), task, f"{self._priority_text(task)}, made room by bumping {labels}")
            for old_start, victim in sorted(removed, key=lambda entry: ranked.index(entry[1])):
                start = self._slot_for(victim)
                if start == old_start:
                    self.planned.append((start, victim))  # it wasn't in the way after all; keep its reason
                elif start is not None and start == victim.preferred_time:
                    self.conflicts.pop(victim, None)  # back at its own preferred time
                    self._plan(start, victim, f"fixed at preferred time {fmt(start)}")
                else:
                    if victim.preferred_time is not None:
                        self.conflicts[victim] = [task]
                    if start is None:
                        self._skip(victim, f"bumped by higher-priority {task.label}")
                    else:
                        self._plan(start, victim, f"moved to make room for higher-priority {task.label}")

    def _free_gaps(self) -> list[list[int]]:
        """Return free [start, end] intervals (minute offsets from day_start) not used by planned tasks."""
        busy = sorted(
            (minutes_between(self.owner.day_start, start), minutes_between(self.owner.day_start, start) + task.duration_minutes)
            for start, task in self.planned
        )
        gaps, cursor = [], 0
        for busy_start, busy_end in busy:
            if busy_start > cursor:
                gaps.append([cursor, busy_start])
            cursor = max(cursor, busy_end)
        if cursor < self.owner.available_minutes:
            gaps.append([cursor, self.owner.available_minutes])
        return gaps

    @staticmethod
    def _overlaps(start_a: time, task_a: Task, start_b: time, task_b: Task) -> bool:
        """Return True if the two scheduled tasks share any time."""
        return start_a < add_minutes(start_b, task_b.duration_minutes) and start_b < add_minutes(start_a, task_a.duration_minutes)

    def sort_by_time(self) -> None:
        """Order `planned` chronologically, keyed on each start time's HH:MM string (zero-padded, so it sorts correctly)."""
        self.planned.sort(key=lambda entry: fmt(entry[0]))

    def _plan(self, start: time, task: Task, reason: str) -> None:
        """Add `task` to the plan at `start` and record why."""
        self.planned.append((start, task))
        self.reasons[task] = reason

    def _skip(self, task: Task, reason: str) -> None:
        """Mark `task` as skipped and record why."""
        self.skipped.append(task)
        self.reasons[task] = reason

    def total_minutes(self) -> int:
        """Return the total duration of all planned tasks."""
        return sum(task.duration_minutes for _, task in self.planned)

    def has_conflicts(self) -> bool:
        """Return True if any planned tasks overlap in time.

        Clashes found during placement are resolved and listed in `conflicts`, so this is a safety check that should stay False.
        """
        latest_end = None
        for start, task in sorted(self.planned, key=lambda entry: entry[0]):
            if latest_end is not None and start < latest_end:
                return True
            end = add_minutes(start, task.duration_minutes)
            latest_end = end if latest_end is None else max(latest_end, end)
        return False

    def conflict_messages(self) -> list[str]:
        """Describe each detected time conflict and how it was resolved."""
        starts = {task: start for start, task in self.planned}
        return [
            f"{task.label} at {fmt(task.preferred_time)} overlaps {', '.join(other.label for other in others)} - "
            + (f"moved to {fmt(starts[task])}" if task in starts else "skipped")
            for task, others in self.conflicts.items()
        ]

    def explain(self) -> str:
        """Return a human-readable explanation built from `planned`, `skipped`, and `reasons`."""
        owner = self.owner
        lines = [
            f"Plan for {owner.name} on {self.day:%A, %b %d %Y} "
            f"(available {fmt(owner.day_start)}-{fmt(owner.day_end)}, {owner.available_minutes} min):"
        ]
        if not self.planned and not self.skipped:
            lines.append("  No tasks are due today.")
            return "\n".join(lines)

        for start, task in self.planned:
            end = add_minutes(start, task.duration_minutes)
            lines.append(
                f"  {fmt(start)}-{fmt(end)}  {task.label} ({task.duration_minutes} min) "
                f"[{task.priority.name.lower()}] - {self.reasons[task]}"
            )
        lines.append(f"Total: {self.total_minutes()} of {owner.available_minutes} min used.")

        if self.skipped:
            lines.append("Skipped:")
            lines.extend(f"  {task.label} - {self.reasons[task]}" for task in self.skipped)
        if self.conflicts:
            lines.append("Conflicts detected:")
            lines.extend(f"  {message}" for message in self.conflict_messages())
        if self.has_conflicts():
            lines.append("Warning: some fixed-time tasks overlap. Adjust their preferred times.")
        return "\n".join(lines)
