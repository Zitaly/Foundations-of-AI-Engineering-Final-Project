"""Demo: build an owner with two pets, add tasks out of order, then sort, filter, and schedule them."""

from datetime import date, time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pawpal_system import Owner, Pet, Priority, Schedule, Task, add_minutes, fmt

console = Console()
PRIORITY_STYLES = {Priority.CRITICAL: "bold red", Priority.HIGH: "dark_orange", Priority.MEDIUM: "blue", Priority.LOW: "grey50"}


def priority_text(task: Task) -> str:
    return f"[{PRIORITY_STYLES[task.priority]}]{task.priority.name.title()}[/]"


def print_tasks(heading: str, tasks: list[Task], day: date) -> None:
    """Print a table with one row per task: time, task, priority, timing rules, and completion status."""
    table = Table(title=heading, title_justify="left", header_style="bold")
    for column in ("Time", "Task", "Priority", "Timing rules", "Status"):
        table.add_column(column)
    for task in tasks:
        when = fmt(task.preferred_time) if task.preferred_time else "flexible"
        status = "[green]done[/]" if task.is_completed(day) else "not done"
        table.add_row(when, task.label, priority_text(task), task.timing_label, status)
    if not tasks:
        table.add_row("", "(none)", "", "", "")
    console.print(table)


def print_schedule(schedule: Schedule) -> None:
    """Print the plan, skipped tasks, and conflicts as tables."""
    owner = schedule.owner
    plan = Table(
        title=f"Plan for {owner.name} on {schedule.day:%A, %b %d %Y}",
        caption=f"{schedule.total_minutes()} of {owner.available_minutes} min used "
                f"(available {fmt(owner.day_start)}-{fmt(owner.day_end)})",
        title_justify="left", caption_justify="left", header_style="bold",
    )
    for column in ("Time", "Task", "Min", "Priority", "Why"):
        plan.add_column(column, overflow="fold")
    for start, task in schedule.planned:
        end = add_minutes(start, task.duration_minutes)
        plan.add_row(f"{fmt(start)}-{fmt(end)}", task.label, str(task.duration_minutes), priority_text(task), schedule.reasons[task])
    console.print(plan)

    if schedule.skipped:
        skipped = Table(title="Didn't fit", title_justify="left", header_style="bold")
        skipped.add_column("Task")
        skipped.add_column("Why", overflow="fold")
        for task in schedule.skipped:
            skipped.add_row(task.label, schedule.reasons[task])
        console.print(skipped)
    if schedule.conflicts:
        console.print(Panel("\n".join(schedule.conflict_messages()), title="Conflicts resolved", border_style="yellow"))


def main() -> None:
    today = date.today()

    owner = Owner("Jordan", available_minutes=180, day_start=time(7, 0))

    mochi = Pet("Mochi", "dog", age=3)
    biscuit = Pet("Biscuit", "cat", age=5)
    owner.add_pet(mochi)
    owner.add_pet(biscuit)

    # Added deliberately out of time order, alternating pets.
    biscuit.add_task(Task("Medication", 5, "critical", preferred_time=time(8, 15)))
    mochi.add_task(Task("Fetch in the yard", 20, "medium", after="Morning walk", after_gap_minutes=15))  # flexible
    mochi.add_task(Task("Morning walk", 30, "high", preferred_time=time(7, 30)))
    biscuit.add_task(Task("Brush coat", 15, "low", not_before=time(9, 0)))  # flexible, but not before 09:00
    mochi.add_task(Task("Evening walk", 30, "medium", preferred_time=time(9, 15)))
    mochi.add_task(Task("Breakfast", 10, "high", preferred_time=time(7, 0)))
    biscuit.add_task(Task("Litter box", 10, "medium", preferred_time=time(7, 45)))

    print_tasks("All tasks, in the order they were added", owner.all_tasks(), today)

    # Same HH:MM key as Schedule.sort_by_time; "flexible" sorts after every HH:MM string.
    by_time = sorted(owner.all_tasks(), key=lambda t: fmt(t.preferred_time) if t.preferred_time else "flexible")
    print_tasks("All tasks, sorted by preferred time", by_time, today)

    # Finish a couple of tasks; each completion adds that task's next occurrence.
    for pet in owner.pets:
        for task in list(pet.tasks):
            if task.title in ("Breakfast", "Medication"):
                next_task = pet.complete_task(task, today)
                console.print(f"Completed [bold]{task.label}[/]; next one added for {next_task.start_date:%a %b %d}.")
    console.print()

    print_tasks("Mochi's tasks", owner.filter_tasks(pet_name="Mochi", day=today), today)
    print_tasks("Biscuit's tasks", owner.filter_tasks(pet_name="Biscuit", day=today), today)
    print_tasks("Completed today", owner.filter_tasks(completed=True, day=today), today)
    print_tasks("Still to do", owner.filter_tasks(completed=False, day=today), today)
    print_tasks("Mochi, still to do", owner.filter_tasks(pet_name="Mochi", completed=False, day=today), today)

    # The schedule only includes tasks still due, placed and then sorted by start time.
    schedule = owner.create_schedule(today)
    print_schedule(schedule)
    console.print(f"Next free 20-minute slot: [bold]{fmt(slot) if (slot := schedule.next_available_slot(20)) else 'none'}[/]")


if __name__ == "__main__":
    main()
