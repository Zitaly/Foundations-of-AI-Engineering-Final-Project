"""PawPal+ core classes (design in diagrams/uml.mmd)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta
from enum import IntEnum

# How many days apart each recurring task is due.
FREQUENCY_DAYS = {"daily": 1, "weekly": 7}

MINUTES_PER_DAY = 24 * 60


class Priority(IntEnum):
    """Task importance; higher values are scheduled first."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


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
    pet_name: str = ""  # set by Pet.add_task
    pet: Pet | None = field(default=None, init=False, repr=False)  # the pet holding this task; set by Pet.add_task

    def __post_init__(self) -> None:
        """Convert a string or int priority to Priority and validate duration and frequency."""
        try:
            if isinstance(self.priority, str):  # accepts "low"/"medium"/"high", e.g. from the Streamlit selectbox
                self.priority = Priority[self.priority.upper()]
            else:  # accepts 1/2/3
                self.priority = Priority(self.priority)
        except (KeyError, ValueError):
            raise ValueError(f"priority must be low, medium, or high, got {self.priority!r}") from None
        if self.duration_minutes <= 0:
            raise ValueError(f"duration_minutes must be positive, got {self.duration_minutes}")
        if self.frequency not in FREQUENCY_DAYS:
            raise ValueError(f"frequency must be one of {list(FREQUENCY_DAYS)}, got {self.frequency!r}")

    @property
    def label(self) -> str:
        """Display name, e.g. "Mochi: Morning walk"."""
        return f"{self.pet_name}: {self.title}" if self.pet_name else self.title

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
        return replace(self, start_date=current + timedelta(days=FREQUENCY_DAYS[self.frequency]), last_completed=None)

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

    Placement rule: tasks with a preferred_time are fixed at that time; a fixed
    task that clashes with a higher-ranked one moves to the nearest free slot;
    flexible tasks then fill the remaining gaps from day_start, highest priority first.
    """

    day: date
    owner: Owner = field(repr=False, compare=False)  # avoids Owner <-> Schedule recursion
    planned: list[tuple[time, Task]] = field(default_factory=list)  # (start time, task) pairs, sorted by time
    skipped: list[Task] = field(default_factory=list)
    reasons: dict[Task, str] = field(default_factory=dict)  # why each task was planned or skipped
    conflicts: dict[Task, list[Task]] = field(default_factory=dict)  # fixed task -> earlier-ranked tasks at its preferred time

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

    def _place(self, tasks: list[Task]) -> None:
        """Clear any previous plan and place `tasks` (already ranked): fixed ones first, then flexible ones."""
        self.planned.clear()
        self.skipped.clear()
        self.reasons.clear()
        self.conflicts.clear()
        flexible =self._place_fixed_tasks(tasks)
        self._fill_flexible_tasks(flexible)
        self.sort_by_time()

    def _collect_due_tasks(self) -> list[Task]:
        """Step 1: gather tasks from all pets that are due on `day`."""
        return [task for pet in self.owner.pets for task in pet.due_tasks(self.day)]

    def _sort_tasks(self, tasks: list[Task]) -> list[Task]:
        """Step 2: order tasks by priority (respecting owner preferences), then by duration; overdue tasks win ties."""
        if self.owner.prefer_high_priority_first:
            return sorted(tasks, key=lambda t: (-t.priority, not t.is_overdue(self.day), t.duration_minutes))
        return sorted(tasks, key=lambda t: (t.duration_minutes, -t.priority, not t.is_overdue(self.day)))

    def _place_fixed_tasks(self, tasks: list[Task]) -> list[Task]:
        """Step 3: place tasks that have a preferred_time; skip ones outside the owner's window.

        A task that clashes with an earlier-ranked fixed task is moved to the nearest free slot.
        Returns the remaining flexible tasks.
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
            if any(self._overlaps(start, task, other_start, other) for other_start, other in self.planned):
                clashing.append(task)  # relocate after every non-clashing fixed task has its slot
                continue
            self._plan(start, task, f"fixed at preferred time {fmt(start)}")
        for task in clashing:
            self._relocate(task)
        return flexible

    def _relocate(self, task: Task) -> None:
        """Move a clashing fixed task to the free slot whose start is nearest its preferred time, or skip it."""
        preferred = minutes_between(self.owner.day_start, task.preferred_time)
        self.conflicts[task] = [
            other for other_start, other in self.planned if self._overlaps(task.preferred_time, task, other_start, other)
        ]
        clashes = ", ".join(other.label for other in self.conflicts[task])
        # Within each gap that fits, the closest possible start is the preferred offset clamped into the gap.
        candidates = [
            min(max(preferred, gap_start), gap_end - task.duration_minutes)
            for gap_start, gap_end in self._free_gaps()
            if gap_end - gap_start >= task.duration_minutes
        ]
        if not candidates:
            self._skip(task, f"fixed at {fmt(task.preferred_time)}, clashes with {clashes} and no other free slot fits")
            return
        start = add_minutes(self.owner.day_start, min(candidates, key=lambda c: abs(c - preferred)))
        self._plan(start, task, f"moved from preferred time {fmt(task.preferred_time)} to avoid {clashes}")

    def _fill_flexible_tasks(self, tasks: list[Task]) -> None:
        """Step 4: pack flexible tasks into free gaps; skip ones that don't fit."""
        gaps = self._free_gaps()  # [start, end] offsets in minutes from day_start
        for task in tasks:
            gap = next((g for g in gaps if g[1] - g[0] >= task.duration_minutes), None)
            if gap is None:
                largest = max((g[1] - g[0] for g in gaps), default=0)
                self._skip(task, f"needs {task.duration_minutes} min, largest free slot is {largest} min")
                continue
            start = add_minutes(self.owner.day_start, gap[0])
            self._plan(start, task, f"{task.priority.name.lower()} priority, placed in earliest free slot")
            gap[0] += task.duration_minutes

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
