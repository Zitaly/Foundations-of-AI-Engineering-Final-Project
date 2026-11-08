"""PawPal+ in the terminal: everything the Streamlit app does, through numbered menus.

Run `python cli.py`. It uses the same saved data (data/pawpal.json) and AI keys as the app, so you can switch
between the two. Options: --day YYYY-MM-DD (planning day), --data PATH (save file), --no-ai (built-in rules only).
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

import storage
from ai_assistant import (
    PlanningAgent,
    PlanProposal,
    TaskDraft,
    apply_changes,
    explain_schedule,
    parse_task,
    parse_time,
    suggest_tasks,
    with_fallback,
)
from export import schedule_rows, schedule_to_ics
from guardrails import (
    MAX_DESCRIPTION_CHARS,
    MAX_FEEDBACK_CHARS,
    MAX_GOAL_CHARS,
    MAX_GUIDELINES_CHARS,
    MAX_NAME_CHARS,
    MAX_NOTES_CHARS,
    MAX_TITLE_CHARS,
    VET_DISCLAIMER,
    QuotaLLM,
    SessionQuota,
    clean_text,
)
from llm import LLM, build_llm, get_api_key, get_groq_key, read_secrets
from logs import recent, setup_logging
from offline import (
    check_guidelines_rules,
    explain_schedule_rules,
    parse_task_rules,
    plan_rules,
    suggest_tasks_rules,
    unchecked_guidelines,
)
from pawpal_system import PRIORITY_NAMES, Owner, Pet, Priority, Schedule, Task, add_minutes, fmt
from rag import Retriever, build_documents, check_guidelines

PRIORITY_STYLES = {Priority.CRITICAL: "bold red", Priority.HIGH: "dark_orange", Priority.MEDIUM: "blue", Priority.LOW: "grey50"}
BACK = "b"


def priority_text(priority: Priority) -> str:
    return f"[{PRIORITY_STYLES[priority]}]{priority.name.title()}[/]"


def priority_name(value: Any) -> str:
    """"high" for Priority.HIGH or "high"; "medium" when there's no value."""
    if value is None:
        return "medium"
    return value if isinstance(value, str) else value.name.lower()


def when(task: Task) -> str:
    return fmt(task.preferred_time) if task.preferred_time else "flexible"


class PawPalCLI:
    """Menu-driven terminal front end over the same classes and AI features as app.py."""

    def __init__(self, owner: Owner, llm: LLM | None = None, console: Console | None = None,
                 path: Path = storage.DEFAULT_PATH, day: date | None = None, problems: list[str] | None = None):
        self.owner, self.path = owner, path
        self.console = console or Console()
        self.day = day or date.today()
        self.problems = problems or []
        self.quota = SessionQuota()
        self.llm = QuotaLLM(llm, self.quota) if llm else None  # same per-session request cap as the app
        self.saved_text = storage.dumps(owner)

    # ---- Input helpers ----

    def ask(self, label: str, default: str | None = None, limit: int = 200, required: bool = False) -> str:
        """A line of text, cleaned and cut to `limit` characters. Blank keeps `default`."""
        while True:
            text = clean_text(Prompt.ask(label, default=default or "", show_default=bool(default), console=self.console), limit)
            if text or not required:
                return text
            self.console.print("[red]Please enter something.[/]")

    def ask_int(self, label: str, default: int, low: int, high: int) -> int:
        while True:
            text = Prompt.ask(f"{label} ({low}-{high})", default=str(default), console=self.console).strip()
            if text.lstrip("-").isdigit() and low <= int(text) <= high:
                return int(text)
            self.console.print(f"[red]Enter a whole number from {low} to {high}.[/]")

    def ask_time(self, label: str, current: time | None, blank_means: str = "keep") -> time | None:
        """A time like 7:30, 18:30, or 6pm. Blank keeps `current`; "none" clears it."""
        hint = f"current {fmt(current)}; 'none' to clear" if current else f"blank = {blank_means}"
        while True:
            text = Prompt.ask(f"{label} [dim]({hint})[/]", default="", show_default=False, console=self.console).strip()
            if not text:
                return current
            try:
                return parse_time(text)
            except ValueError as e:
                self.console.print(f"[red]{e}[/]")

    def choose(self, label: str, options: list[str], allow_none: str | None = None) -> int | None:
        """Pick from a numbered list. Returns the index, or None for `allow_none` (option 0) or a blank answer."""
        if not options and not allow_none:
            return None
        if allow_none:
            self.console.print(f"  [bold]0[/] {allow_none}")
        for i, option in enumerate(options, 1):
            self.console.print(f"  [bold]{i}[/] {escape(option)}")
        while True:
            text = Prompt.ask(label, default="", show_default=False, console=self.console).strip()
            if not text or (allow_none and text == "0"):
                return None
            if text.isdigit() and 1 <= int(text) <= len(options):
                return int(text) - 1
            self.console.print(f"[red]Enter a number from {0 if allow_none else 1} to {len(options)}, or leave blank to cancel.[/]")

    def confirm(self, label: str, default: bool = False) -> bool:
        return Confirm.ask(label, default=default, console=self.console)

    def menu(self, title: str, options: dict[str, str]) -> str:
        """Show single-key options and return the chosen key."""
        self.console.print(f"\n[bold]{title}[/]  " + "  ".join(f"[bold cyan]{k}[/] {v}" for k, v in options.items()))
        return Prompt.ask("Choose", choices=list(options), show_choices=False, console=self.console).lower()

    def error(self, text: str) -> None:
        self.console.print(f"[red]{escape(text)}[/]")

    def pick_pet(self) -> Pet | None:
        if not self.owner.pets:
            self.error("Add a pet first (Pets menu).")
            return None
        if len(self.owner.pets) == 1:
            return self.owner.pets[0]
        index = self.choose("Pet", [f"{p.name} ({p.species})" for p in self.owner.pets])
        return None if index is None else self.owner.pets[index]

    # ---- AI ----

    def ai_or_rules(self, feature: str, ai_call: Callable[[LLM], Any], rules_call: Callable[[], Any]) -> tuple[Any, str]:
        """Run an AI feature with the built-in rules as fallback. Returns (result, line saying who answered)."""
        with self.console.status("Thinking..."):
            result, note = with_fallback(self.llm, ai_call, rules_call, feature)
        return result, note or f"Answered by {getattr(self.llm, 'last_used', None) or 'the AI'}."

    def retriever(self) -> Retriever:
        return Retriever(build_documents(self.owner, self.day))

    # ---- Output ----

    def task_status(self, task: Task) -> str:
        if task.is_completed(self.day):
            return "[green]done[/]"
        if task.is_overdue(self.day):
            return f"[yellow]overdue since {task.start_date:%b %d}[/]"
        if not task.is_due(self.day):
            return f"next due {task.start_date:%b %d}"
        return "due"

    def show_tasks(self, tasks: list[Task], title: str) -> None:
        table = Table(title=title, title_justify="left", header_style="bold")
        for column in ("#", "Time", "Task", "Min", "Priority", "Repeats", "Timing rules", "Status"):
            table.add_column(column, overflow="fold")
        for i, task in enumerate(tasks, 1):
            table.add_row(str(i), when(task), escape(task.label), str(task.duration_minutes), priority_text(task.priority),
                          task.frequency, escape(task.timing_label), self.task_status(task))
        if not tasks:
            table.add_row("", "", "(none)", "", "", "", "", "")
        self.console.print(table)

    def show_schedule(self, schedule: Schedule) -> None:
        owner = schedule.owner
        planned, skipped = len(schedule.planned), len(schedule.skipped)
        used = f"{schedule.total_minutes()} of {owner.available_minutes} min used, {fmt(owner.day_start)}-{fmt(owner.day_end)}"
        if not planned and not skipped:
            self.console.print("[dim]No tasks are due on this day.[/]")
            return
        if not skipped:
            self.console.print(f"[green]All {planned} due task(s) fit ({used}).[/]")
        else:
            self.console.print(f"[yellow]Planned {planned} of {planned + skipped} due tasks ({used}).[/]")
        critical = [t.label for t in schedule.skipped if t.priority == Priority.CRITICAL]
        if critical:
            self.error(f"Critical task not planned: {', '.join(critical)}. Free up time or change its fixed time.")
        if planned:
            table = Table(title=f"Plan for {schedule.day:%A, %b %d %Y}", title_justify="left", header_style="bold")
            for column in ("Time", "Task", "Min", "Priority", "Why"):
                table.add_column(column, overflow="fold")
            for start, task in schedule.planned:
                table.add_row(f"{fmt(start)}-{fmt(add_minutes(start, task.duration_minutes))}", escape(task.label),
                              str(task.duration_minutes), priority_text(task.priority), escape(schedule.reasons[task]))
            self.console.print(table)
        if skipped:
            table = Table(title="Didn't fit", title_justify="left", header_style="bold")
            table.add_column("Task")
            table.add_column("Priority")
            table.add_column("Why", overflow="fold")
            for task in schedule.skipped:
                table.add_row(escape(task.label), priority_text(task.priority), escape(schedule.reasons[task]))
            self.console.print(table)
        for message in schedule.conflict_messages():
            self.console.print(f"[dim]Time conflict: {escape(message)}[/]")
        if schedule.has_conflicts():
            self.error("Some planned tasks still overlap. Adjust their preferred times.")

    def header(self) -> None:
        preview = Schedule(self.day, self.owner)  # not stored, so looking doesn't add history
        preview.generate()
        due = sum(len(p.due_tasks(self.day)) for p in self.owner.pets)
        overdue = sum(t.is_overdue(self.day) for t in self.owner.all_tasks())
        ai = f"{getattr(self.llm, 'name', 'on')} → built-in rules" if self.llm else "off (built-in rules)"
        self.console.print(Panel(
            f"[bold]{self.day:%A, %B %d %Y}[/]   {escape(self.owner.name)}, {len(self.owner.pets)} pet(s)\n"
            f"Tasks due: [bold]{due}[/]   Overdue: [bold]{overdue}[/]   "
            f"Planned: [bold]{preview.total_minutes()}/{self.owner.available_minutes} min[/]   "
            f"Didn't fit: [bold]{len(preview.skipped)}[/]\nAI: {ai}",
            title="PawPal+", title_align="left", border_style="cyan",
        ))

    # ---- Main loop ----

    def save(self) -> None:
        """Autosave when something changed, like the app does after every run."""
        text = storage.dumps(self.owner)
        if text == self.saved_text:
            return
        try:
            storage.save(self.owner, self.path)
            self.saved_text = text
        except OSError as e:
            self.error(f"Couldn't save your data: {e}")

    def run(self) -> None:
        actions = {
            "1": ("Schedule", self.schedule_menu),
            "2": ("Tasks", self.tasks_menu),
            "3": ("Pets and guidelines", self.pets_menu),
            "4": ("AI assistant", self.ai_menu),
            "5": ("Settings", self.settings_menu),
            "6": ("Data (import/export)", self.data_menu),
            "7": ("AI status and activity log", self.status),
            "q": ("Quit", None),
        }
        for problem in self.problems:
            self.error(problem)
        try:
            while True:
                self.header()
                choice = self.menu("Main menu", {k: v[0] for k, v in actions.items()})
                if choice == "q":
                    break
                actions[choice][1]()
                self.save()
        except (KeyboardInterrupt, EOFError):
            self.console.print()
        self.save()
        self.console.print(f"Saved to {self.path}. Bye!")

    # ---- Schedule ----

    def schedule_menu(self) -> None:
        if not self.owner.all_tasks():
            self.error("Add pets and tasks first.")
            return
        while True:
            schedule = self.owner.create_schedule(self.day)  # stored, so skips become history for the AI
            self.show_schedule(schedule)
            choice = self.menu("Schedule", {"e": "explain (AI)", "s": "find a free slot", "r": "scheduler reasoning",
                                            "i": "export .ics", "c": "export CSV", BACK: "back"})
            if choice == BACK:
                return
            if choice == "e":
                text, note = self.ai_or_rules(
                    "Explain plan", lambda m: explain_schedule(m, schedule, self.retriever()), lambda: explain_schedule_rules(schedule),
                )
                self.console.print(Panel(escape(text), title="Explanation", title_align="left"))
                self.console.print(f"[dim]{escape(note)} The table is the source of truth. {VET_DISCLAIMER}[/]")
            elif choice == "s":
                minutes = self.ask_int("Minutes needed", 15, 5, 240)
                slot = schedule.next_available_slot(minutes)
                self.console.print(f"Next free {minutes}-minute slot: [bold]{fmt(slot)}[/]" if slot else "No free slot that long today.")
            elif choice == "r":
                self.console.print(Panel(escape(schedule.explain()), title="Step-by-step", title_align="left"))
            elif choice in ("i", "c"):
                self.export_schedule(schedule, "ics" if choice == "i" else "csv")

    def export_schedule(self, schedule: Schedule, kind: str) -> None:
        if not schedule.planned:
            self.error("Nothing is planned, so there's nothing to export.")
            return
        path = Path(self.ask("Save as", f"pawpal-{self.day}.{kind}"))
        try:
            if kind == "ics":
                path.write_text(schedule_to_ics(schedule), encoding="utf-8", newline="")
            else:
                with open(path, "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=list(schedule_rows(schedule)[0]))
                    writer.writeheader()
                    writer.writerows(schedule_rows(schedule))
            self.console.print(f"[green]Saved {path.resolve()}[/]")
        except OSError as e:
            self.error(f"Couldn't save: {e}")

    # ---- Tasks ----

    def open_tasks(self) -> list[Task]:
        return sorted(self.owner.filter_tasks(completed=False, day=self.day), key=lambda t: (when(t), t.label))

    def tasks_menu(self) -> None:
        while True:
            tasks = self.open_tasks()
            self.show_tasks(tasks, f"Tasks not yet done as of {self.day:%b %d}")
            choice = self.menu("Tasks", {"a": "add", "d": "describe in words (AI)", "e": "edit", "m": "mark done",
                                         "r": "remove", "h": "show completed", BACK: "back"})
            if choice == BACK:
                return
            if choice == "a":
                self.add_task()
            elif choice == "d":
                self.describe_task()
            elif choice == "h":
                self.show_tasks(self.owner.filter_tasks(completed=True, day=self.day), "Completed")
            elif (task := self.pick_task(tasks)) is not None:
                if choice == "e":
                    self.edit_task(task)
                elif choice == "m":
                    self.complete_task(task)
                elif choice == "r" and self.confirm(f"Remove {task.label}? This can't be undone"):
                    task.pet.remove_task(task)
                    self.console.print(f"Removed {escape(task.label)}.")

    def pick_task(self, tasks: list[Task]) -> Task | None:
        if not tasks:
            self.error("No tasks to choose from.")
            return None
        while True:
            text = Prompt.ask("Task # (blank to cancel)", default="", show_default=False, console=self.console).strip()
            if not text:
                return None
            if text.isdigit() and 1 <= int(text) <= len(tasks):
                return tasks[int(text) - 1]
            self.error(f"Enter a number from 1 to {len(tasks)}.")

    def task_form(self, pet: Pet, defaults: Task | TaskDraft | None = None, new: bool = True) -> dict:
        """Ask for task fields, offering `defaults`. Returns Task keyword arguments (no title/frequency when editing)."""
        fields: dict[str, Any] = {}
        title = getattr(defaults, "title", None)
        if new:
            title = fields["title"] = self.ask("Task name", title, MAX_TITLE_CHARS, required=True)
        fields["duration_minutes"] = self.ask_int("Minutes", min(getattr(defaults, "duration_minutes", 20), 240), 1, 240)
        fields["priority"] = Prompt.ask("Priority", choices=PRIORITY_NAMES, default=priority_name(getattr(defaults, "priority", None)),
                                        console=self.console)
        if new:
            fields["frequency"] = Prompt.ask("Repeats", choices=["daily", "weekly"], default=getattr(defaults, "frequency", "daily"),
                                             console=self.console)
            fields |= {"not_before": getattr(defaults, "not_before", None), "done_by": getattr(defaults, "done_by", None)}
        fields["preferred_time"] = self.ask_time("Fixed time", getattr(defaults, "preferred_time", None), "flexible")
        has_timing = any(getattr(defaults, name, None) for name in ("not_before", "done_by", "after"))
        if self.confirm("Set timing rules (a time window or 'comes after' another task)?", default=has_timing):
            fields["not_before"] = self.ask_time("Not before", getattr(defaults, "not_before", None), "no limit")
            fields["done_by"] = self.ask_time("Done by", getattr(defaults, "done_by", None), "no limit")
            siblings = sorted({t.title for t in pet.tasks if t.title.lower() != str(title).lower()})
            current = getattr(defaults, "after", "")
            self.console.print(f"Comes after another of {escape(pet.name)}'s tasks" + (f" [dim](current: {escape(current)})[/]" if current else "") + ":")
            index = self.choose("Choice (blank keeps current)", siblings, allow_none="(none)")
            fields["after"] = siblings[index] if index is not None else ""
            if index is None and current and not self.confirm(f"Remove the rule to come after {current}?", default=False):
                fields["after"] = current
            if fields["after"]:
                fields["after_gap_minutes"] = self.ask_int("Minutes after it", getattr(defaults, "after_gap_minutes", 0), 0, 240)
            else:
                fields["after_gap_minutes"] = 0
        return fields

    def add_task(self, draft: TaskDraft | None = None) -> None:
        if draft is not None:
            pet = next((p for p in self.owner.pets if p.name == draft.pet_name), None) or self.pick_pet()
        else:
            pet = self.pick_pet()
        if pet is None:
            return
        try:
            task = Task(**self.task_form(pet, draft), start_date=self.day)
            pet.add_task(task)
            self.console.print(f"[green]Added {escape(task.label)}.[/]")
        except ValueError as e:
            self.error(f"Couldn't add task: {e}")

    def describe_task(self) -> None:
        if not self.owner.pets:
            self.error("Add a pet first (Pets menu).")
            return
        text = self.ask("Describe the task", limit=MAX_DESCRIPTION_CHARS, required=True)
        try:
            draft, note = self.ai_or_rules("Draft task", lambda m: parse_task(m, text, self.owner), lambda: parse_task_rules(text, self.owner))
        except ValueError as e:
            self.error(f"Couldn't draft that task: {e}")
            return
        self.console.print(f"[dim]{escape(note)}[/]")
        self.show_draft(draft, "Draft")
        choice = Prompt.ask("Add it? [bold]y[/] yes, [bold]e[/] edit first, [bold]n[/] discard", choices=["y", "e", "n"],
                            default="y", show_choices=False, console=self.console)
        if choice == "y":
            try:
                pet = next(p for p in self.owner.pets if p.name == draft.pet_name)
                pet.add_task(draft.to_task(self.day))
                self.console.print(f"[green]Added {escape(draft.pet_name)}: {escape(draft.title)}.[/]")
            except (StopIteration, ValueError) as e:
                self.error(f"Couldn't add task: {e}")
        elif choice == "e":
            self.add_task(draft)

    def show_draft(self, draft: TaskDraft, title: str) -> None:
        timing = ", ".join(filter(None, [
            f"not before {fmt(draft.not_before)}" if draft.not_before else "", f"done by {fmt(draft.done_by)}" if draft.done_by else "",
        ]))
        self.console.print(Panel(
            f"[bold]{escape(draft.pet_name)}: {escape(draft.title)}[/]\n{draft.duration_minutes} min · {draft.priority} · "
            f"{fmt(draft.preferred_time) if draft.preferred_time else 'flexible'} · {draft.frequency}"
            + (f" · {timing}" if timing else "") + (f"\n{escape(draft.rationale)}" if draft.rationale else ""),
            title=title, title_align="left",
        ))

    def edit_task(self, task: Task) -> None:
        self.console.print(f"Editing [bold]{escape(task.label)}[/] (press Enter to keep each value)")
        try:
            task.update(**self.task_form(task.pet, task, new=False))
            self.console.print(f"[green]Updated {escape(task.label)}.[/]")
        except ValueError as e:
            self.error(f"Not saved: {e}")

    def complete_task(self, task: Task) -> None:
        if not task.is_due(self.day):
            self.error(f"{task.label} isn't due on {self.day:%b %d}.")
            return
        next_task = task.pet.complete_task(task, self.day)
        self.console.print(f"[green]Marked {escape(task.label)} done.[/] Next due {next_task.start_date:%a %b %d}.")

    # ---- Pets ----

    def pets_menu(self) -> None:
        while True:
            table = Table(title="Your pets", title_justify="left", header_style="bold")
            for column in ("Pet", "Species", "Age", "Notes", "Open tasks", "Guidelines"):
                table.add_column(column, overflow="fold")
            for p in self.owner.pets:
                table.add_row(escape(p.name), p.species, str(p.age), escape(p.notes), str(sum(not t.is_completed(self.day) for t in p.tasks)),
                              escape("\n".join(f"{i}. {g}" for i, g in enumerate(p.guidelines, 1))))
            if not self.owner.pets:
                table.add_row("(none)", "", "", "", "", "")
            self.console.print(table)
            choice = self.menu("Pets", {"a": "add", "g": "guidelines and notes", "r": "remove", BACK: "back"})
            if choice == BACK:
                return
            if choice == "a":
                self.add_pet()
            elif (pet := self.pick_pet()) is not None:
                if choice == "g":
                    self.edit_guidelines(pet)
                elif choice == "r" and self.confirm(f"Remove {pet.name} and all of their tasks? This can't be undone"):
                    self.owner.remove_pet(pet)
                    self.console.print(f"Removed {escape(pet.name)}.")

    def add_pet(self) -> None:
        name = self.ask("Name", limit=MAX_NAME_CHARS, required=True)
        species = Prompt.ask("Species", choices=["dog", "cat", "other"], default="dog", console=self.console)
        age = self.ask_int("Age in years", 3, 0, 40)
        notes = self.ask("Notes (optional, e.g. 'senior, sensitive stomach')", limit=MAX_NOTES_CHARS)
        try:
            self.owner.add_pet(Pet(name, species, age=age, notes=notes))
            self.console.print(f"[green]Added {escape(name)}.[/]")
        except ValueError as e:
            self.error(f"Couldn't add pet: {e}")

    def edit_guidelines(self, pet: Pet) -> None:
        self.console.print(
            "[dim]Guidelines are your own care rules, cited as e.g. \\[Mochi guideline 1]. The built-in rules understand shapes "
            "like 'At least 60 minutes of walks a day', 'Medication before 09:00', 'No food after 20:00', 'Brush coat twice a week'.[/]"
        )
        while True:
            self.console.print(f"[bold]{escape(pet.name)}[/] notes: {escape(pet.notes) or '(none)'}")
            for i, guideline in enumerate(pet.guidelines, 1):
                self.console.print(f"  {i}. {escape(guideline)}")
            if not pet.guidelines:
                self.console.print("  (no guidelines)")
            choice = self.menu("Guidelines", {"a": "add", "r": "remove", "n": "edit notes", BACK: "back"})
            if choice == BACK:
                return
            if choice == "a":
                room = MAX_GUIDELINES_CHARS - len("\n".join(pet.guidelines)) - 1
                if room <= 0:
                    self.error(f"{pet.name}'s guidelines are at the {MAX_GUIDELINES_CHARS}-character limit. Remove one first.")
                    continue
                if text := self.ask("New guideline", limit=min(room, 300)).replace("\n", " "):
                    pet.guidelines.append(text)
            elif choice == "r":
                index = self.choose("Remove which", pet.guidelines)
                if index is not None:
                    pet.guidelines.pop(index)
            elif choice == "n":
                pet.notes = self.ask("Notes", pet.notes, MAX_NOTES_CHARS)

    # ---- AI assistant ----

    def ai_menu(self) -> None:
        if not self.owner.pets:
            self.error("Add a pet first (Pets menu).")
            return
        while True:
            choice = self.menu("AI assistant", {"i": "task ideas", "c": "check tasks against guidelines", "p": "plan my day", BACK: "back"})
            if choice == BACK:
                return
            {"i": self.task_ideas, "c": self.guideline_check, "p": self.plan_day}[choice]()

    def task_ideas(self) -> None:
        pet = self.pick_pet()
        if pet is None:
            return
        retriever = self.retriever()
        ideas, note = self.ai_or_rules(
            "Task ideas", lambda m: suggest_tasks(m, self.owner, pet, retriever, self.day),
            lambda: suggest_tasks_rules(self.owner, pet, retriever, self.day),
        )
        self.console.print(f"[dim]{escape(note)} {VET_DISCLAIMER}[/]")
        if not ideas:
            self.console.print("No new ideas. This pet's tasks already look complete.")
        for i, idea in enumerate(ideas, 1):
            self.show_draft(idea, f"Idea {i} of {len(ideas)}")
            if self.confirm("Add this task?", default=False):
                try:
                    pet.add_task(idea.to_task(self.day))
                    self.console.print(f"[green]Added {escape(idea.pet_name)}: {escape(idea.title)}.[/]")
                except ValueError as e:
                    self.error(f"Couldn't add {idea.title}: {e}")

    def guideline_check(self) -> None:
        if not any(p.guidelines for p in self.owner.pets):
            self.error("Add care guidelines first (Pets menu > guidelines).")
            return
        issues, note = self.ai_or_rules(
            "Guideline check", lambda m: check_guidelines(m, self.owner, self.retriever(), self.day),
            lambda: check_guidelines_rules(self.owner, self.day),
        )
        self.console.print(f"[dim]{escape(note)}[/]")
        if not issues:
            self.console.print("[green]Your tasks follow every guideline that was checked.[/]")
        for issue in issues:
            self.console.print(Panel(
                f"{escape(issue.issue)}\nSuggestion: {escape(issue.suggestion or 'none')}\n"
                f"[dim]Source: {escape(f'[{issue.ref}] {issue.guideline}')}[/]",
                title=escape(issue.pet_name), title_align="left", border_style="yellow",
            ))
        if "built-in rules" in note and (unchecked := unchecked_guidelines(self.owner)):
            self.console.print(f"[dim]The built-in rules couldn't interpret, so didn't check: {escape(', '.join(unchecked))}.[/]")

    def plan_day(self) -> None:
        if not self.owner.all_tasks():
            self.error("Add tasks first.")
            return
        self.console.print("[dim]Describe what today needs. Nothing changes until you approve the plan.[/]")
        goal = self.ask("Goal", limit=MAX_GOAL_CHARS, required=True)
        proposal, note = self.draft_plan(goal)
        while True:
            self.show_proposal(proposal, note)
            choice = self.menu("Proposed plan", {"a": "approve", "f": "revise with feedback", "s": "show agent steps", "r": "reject"})
            if choice == "a":
                try:
                    apply_changes(self.owner, proposal.changes, proposal.day)
                    self.owner.create_schedule(proposal.day)
                    self.console.print(f"[green]Plan applied ({len(proposal.changes)} change(s)).[/]")
                except (TypeError, ValueError) as e:
                    self.error(f"Couldn't apply the plan: {e}")
                return
            if choice == "r":
                self.console.print("Plan discarded. Nothing was changed.")
                return
            if choice == "s":
                self.show_steps(proposal)
            elif choice == "f":
                feedback = self.ask("What should change", limit=MAX_FEEDBACK_CHARS, required=True)
                proposal, note = self.draft_plan(f"{proposal.goal}\n\nFeedback on your previous proposal: {feedback}")

    def draft_plan(self, goal: str) -> tuple[PlanProposal, str]:
        retriever = self.retriever()
        return self.ai_or_rules(
            "Plan my day", lambda m: PlanningAgent(m, self.owner, self.day, retriever).run(goal),
            lambda: plan_rules(self.owner, self.day, goal),
        )

    def show_proposal(self, proposal: PlanProposal, note: str) -> None:
        self.console.rule(f"Proposed plan for {proposal.day:%A, %b %d}")
        self.console.print(f"[dim]{escape(note)}[/]")
        if proposal.summary:
            self.console.print(Panel(escape(proposal.summary), title="Summary", title_align="left"))
        for warning in proposal.warnings:
            self.console.print(f"[yellow]! {escape(warning)}[/]")
        self.console.print("[bold]Proposed changes[/]")
        for change in proposal.changes:
            self.console.print(f"  - {escape(change['description'])}")
        if not proposal.changes:
            self.console.print("  (none)")
        self.show_schedule(proposal.schedule)

    def show_steps(self, proposal: PlanProposal) -> None:
        if not proposal.steps:
            self.console.print("No agent steps (the built-in rules made this plan).")
            return
        self.console.print(f"[bold]{proposal.turns} model call(s), {len(proposal.steps)} action(s)[/]")
        for step in proposal.steps:
            thought = f" [dim]{escape(step.thought)}[/]" if step.thought else ""
            self.console.print(f"Turn {step.turn}: [cyan]{escape(step.tool)}[/] {escape(str(step.args))}{thought}\n"
                               f"    -> {escape(step.result)}", highlight=False)

    # ---- Settings ----

    def settings_menu(self) -> None:
        o = self.owner
        while True:
            table = Table(title="Settings", title_justify="left", header_style="bold", show_header=False)
            table.add_column("Key", style="bold cyan")
            table.add_column("Setting")
            table.add_column("Value")
            for key, name, value in (
                ("d", "Planning day", f"{self.day:%a %b %d %Y}"),
                ("n", "Owner name", escape(o.name)),
                ("w", "Available time", f"{fmt(o.day_start)}-{fmt(o.day_end)} ({o.available_minutes} min)"),
                ("o", "Place tasks", "highest priority first" if o.prefer_high_priority_first else "shortest first"),
                ("m", "Max tasks per day", str(o.max_tasks_per_day or "no limit")),
                ("a", "Overdue boost per day", f"{o.overdue_boost_per_day:g} priority levels"),
                ("u", "Important tasks bump less important ones", "on" if o.bump_lower_priority else "off"),
                ("p", "Best-fit packing", "on" if o.best_fit_packing else "off"),
            ):
                table.add_row(key, name, value)
            self.console.print(table)
            choice = self.menu("Settings", {"d": "day", "n": "name", "w": "time", "o": "order", "m": "max tasks",
                                            "a": "boost", "u": "bumping", "p": "packing", BACK: "back"})
            if choice == BACK:
                return
            try:
                self.change_setting(choice)
            except ValueError as e:
                self.error(f"Not saved: {e}")

    def change_setting(self, key: str) -> None:
        o = self.owner
        if key == "d":
            text = self.ask("Day (YYYY-MM-DD, 'today', or 'tomorrow')", "today").lower()
            self.day = {"today": date.today(), "tomorrow": date.today() + timedelta(days=1)}.get(text) or date.fromisoformat(text)
        elif key == "n":
            o.name = self.ask("Owner name", o.name, MAX_NAME_CHARS) or o.name
        elif key == "w":
            start = self.ask_time("Day starts", o.day_start)
            o.set_window(start or o.day_start, self.ask_int("Free minutes", o.available_minutes, 0, 24 * 60 - 1))
        elif key == "o":
            o.prefer_high_priority_first = not o.prefer_high_priority_first
        elif key == "m":
            o.max_tasks_per_day = self.ask_int("Max tasks per day (0 = no limit)", o.max_tasks_per_day or 0, 0, 50) or None
        elif key == "a":
            boost = float(self.ask("Priority levels an overdue task gains per day (0-1)", f"{o.overdue_boost_per_day:g}"))
            if not 0 <= boost <= 1:
                raise ValueError("the boost must be between 0 and 1")
            o.overdue_boost_per_day = boost
        elif key == "u":
            o.bump_lower_priority = not o.bump_lower_priority
        elif key == "p":
            o.best_fit_packing = not o.best_fit_packing

    # ---- Data ----

    def data_menu(self) -> None:
        self.console.print(f"Changes save automatically to [bold]{self.path}[/].")
        choice = self.menu("Data", {"e": "export JSON", "i": "import JSON (replaces everything)", BACK: "back"})
        if choice == "e":
            path = Path(self.ask("Save as", "pawpal-data.json"))
            try:
                path.write_text(storage.dumps(self.owner), encoding="utf-8")
                self.console.print(f"[green]Saved {path.resolve()}[/]")
            except OSError as e:
                self.error(f"Couldn't save: {e}")
        elif choice == "i":
            path = Path(self.ask("File to import", required=True))
            try:
                owner = storage.loads(path.read_text(encoding="utf-8"))
            except (OSError, storage.StorageError, UnicodeDecodeError) as e:
                self.error(f"Couldn't import that file: {e}")
                return
            if self.confirm(f"Replace all current data with {len(owner.pets)} pet(s) from {path.name}?"):
                self.owner = owner
                self.console.print("[green]Data imported.[/]")

    # ---- Status ----

    def status(self) -> None:
        for problem in self.problems:
            self.error(problem)
        if self.llm is None:
            self.console.print("AI is off, so AI features use PawPal+'s built-in rules. "
                               "Set GEMINI_API_KEY or GROQ_API_KEY (or add them to .streamlit/secrets.toml) to use AI.")
        else:
            self.console.print(f"[green]Connected:[/] {getattr(self.llm, 'name', 'AI')} → built-in rules")
            for provider in getattr(self.llm, "providers", []):
                self.console.print(f"  {provider.provider} ({provider.model}): {provider.request_count} request(s), "
                                   f"{provider.cache_hits} from cache, backups: {', '.join(provider.models[1:]) or 'none'}")
            self.console.print(f"AI requests this session: {self.quota.used} of {self.quota.limit}")
        self.console.print(Panel("\n".join(recent.lines) or "Nothing logged yet.", title="Recent activity", title_align="left"),
                           markup=False, highlight=False)


def load_owner(path: Path, console: Console) -> Owner:
    """The saved owner, or a new one. An unreadable save file is moved aside instead of being overwritten."""
    try:
        return storage.load(path) or Owner("Jordan")
    except storage.StorageError as e:
        backup = storage.quarantine(path)
        console.print(f"[yellow]Your saved data couldn't be read ({e}). It was moved to {backup.name}; starting fresh.[/]")
        return Owner("Jordan")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PawPal+ in the terminal.")
    parser.add_argument("--day", type=date.fromisoformat, help="planning day, YYYY-MM-DD (default: today)")
    parser.add_argument("--data", type=Path, default=storage.DEFAULT_PATH, help=f"save file (default: {storage.DEFAULT_PATH})")
    parser.add_argument("--no-ai", action="store_true", help="use only the built-in rules")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # consoles without Unicode show "?" instead of crashing
    setup_logging(console=False)  # warnings would interrupt prompts; see menu 7 or data/logs instead
    console = Console()
    llm, problems = None, []
    if not args.no_ai:
        secrets = read_secrets()
        llm, problems = build_llm(get_api_key() or secrets.get("GEMINI_API_KEY"), get_groq_key() or secrets.get("GROQ_API_KEY"))
    PawPalCLI(load_owner(args.data, console), llm, console, args.data, args.day, problems).run()


if __name__ == "__main__":
    main()
