"""PawPal+: plan daily pet care around your time, your priorities, and your care guidelines."""

import json
from datetime import date, time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import storage
from ai_assistant import PlanningAgent, TaskDraft, apply_changes, explain_schedule, parse_task, suggest_tasks, with_fallback
from export import schedule_rows, schedule_to_ics
from llm import FallbackLLM, GeminiLLM, GroqLLM, LLMError, get_api_key, get_groq_key
from offline import (
    check_guidelines_rules,
    explain_schedule_rules,
    parse_task_rules,
    plan_rules,
    suggest_tasks_rules,
    unchecked_guidelines,
)
from pawpal_system import PRIORITY_NAMES, Owner, Pet, Priority, Schedule, Task, add_minutes, fmt, minutes_between
from rag import Retriever, build_documents, check_guidelines

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="wide")

PRIORITY_COLORS = {"Critical": "#c92a2a", "High": "#e8590c", "Medium": "#1c7ed6", "Low": "#868e96"}
PRIORITY_BADGES = {"Critical": "🔴 Critical", "High": "🟠 High", "Medium": "🔵 Medium", "Low": "⚪ Low"}
MINUTES = st.column_config.NumberColumn("Minutes", format="%d min")
MIDNIGHT = time(0, 0)


def badge(task: Task) -> str:
    return PRIORITY_BADGES[task.priority.name.title()]


# ---- Data: load once per session, save after every run that changed something ----

def load_owner() -> Owner:
    """The saved owner, or a new one. An unreadable save file is moved aside instead of being overwritten."""
    try:
        return storage.load() or Owner("Jordan")
    except storage.StorageError as e:
        backup = storage.quarantine()
        st.session_state.load_warning = (
            f"Your saved data couldn't be read ({e}). It was moved to {backup.name} and a new profile was started."
        )
        return Owner("Jordan")


if "owner" not in st.session_state:
    st.session_state.owner = load_owner()
owner: Owner = st.session_state.owner


def clear_ai_state() -> None:
    """Forget drafts, ideas, and proposals that may refer to pets or tasks that no longer exist."""
    for key in ("task_draft", "ideas", "proposal", "ai_explanation"):
        st.session_state.pop(key, None)


def import_data() -> None:
    """File uploader callback: replace all data with the uploaded JSON."""
    upload = st.session_state.get("import_file")
    if upload is None:
        return
    try:
        st.session_state.owner = storage.loads(upload.getvalue().decode("utf-8"))
        clear_ai_state()
        st.session_state.flash = "Data imported."
    except (storage.StorageError, UnicodeDecodeError) as e:
        st.session_state.flash_error = f"Couldn't import that file: {e}"


# ---- AI ----

def secret(name: str, from_env: str | None) -> str | None:
    """A key from the environment, else from .streamlit/secrets.toml."""
    if from_env:
        return from_env
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets.toml
        return None


@st.cache_resource
def make_llm(gemini_key: str | None, groq_key: str | None) -> tuple[FallbackLLM | None, list[str]]:
    """Gemini, then Groq, shared by every session so rate pacing and the reply cache cover the whole quota.

    Returns the chain (None if no provider is available) and any setup problems to show.
    """
    providers, problems = [], []
    for name, key, cls in (("Gemini", gemini_key, GeminiLLM), ("Groq", groq_key, GroqLLM)):
        if not key:
            continue
        try:
            providers.append(cls(api_key=key))
        except (LLMError, ImportError) as e:
            problems.append(f"{name} unavailable: {e}")
    for provider in providers[:-1]:
        provider.max_wait = 0  # don't wait for a busy provider when the next one can answer now
    return (FallbackLLM(providers) if providers else None), problems


llm, llm_problems = make_llm(secret("GEMINI_API_KEY", get_api_key()), secret("GROQ_API_KEY", get_groq_key()))


def ai_or_rules(ai_call, rules_call) -> tuple:
    """Run an AI feature with the built-in rules as fallback. Returns (result, caption saying who answered)."""
    result, note = with_fallback(llm, ai_call, rules_call)
    return result, note or f"Answered by {llm.last_used}."

# ---- Sidebar: the day and the owner's settings ----

with st.sidebar:
    st.header("Your day")
    view_day = st.date_input("Planning day", value=date.today(), key="view_day")
    owner_name = st.text_input("Owner name", value=owner.name)
    col1, col2 = st.columns(2)
    day_start = col1.time_input("Day starts", value=owner.day_start, step=900)
    available_minutes = col2.number_input("Free minutes", min_value=0, max_value=24 * 60, value=owner.available_minutes, step=15)
    try:
        owner.set_window(day_start, int(available_minutes))  # validates before applying either value
        owner.name = owner_name.strip() or owner.name
    except ValueError as e:
        st.error(f"Settings not saved: {e}")

    with st.expander("Scheduling rules"):
        order = st.radio(
            "Place tasks", ["Highest priority first", "Shortest first"],
            index=0 if owner.prefer_high_priority_first else 1,
            help="Shortest first fits as many tasks as possible.",
        )
        owner.prefer_high_priority_first = order == "Highest priority first"
        limit = st.number_input("Max tasks per day (0 = no limit)", min_value=0, max_value=50, value=owner.max_tasks_per_day or 0)
        owner.max_tasks_per_day = int(limit) or None
        owner.overdue_boost_per_day = st.slider(
            "Overdue boost per day", 0.0, 1.0, float(owner.overdue_boost_per_day), 0.25,
            help="Priority levels an overdue task gains for each day it's late, up to High. 0.5 turns Low into High after 4 days.",
        )
        owner.bump_lower_priority = st.toggle(
            "Important tasks can bump less important ones", value=owner.bump_lower_priority,
            help="If a task doesn't fit, strictly lower-priority tasks are moved or dropped to make room. "
                 "Only applies when placing highest priority first.",
        )
        owner.best_fit_packing = st.toggle(
            "Best-fit packing", value=owner.best_fit_packing,
            help="Arrange flexible tasks so the most important ones fit, instead of filling the earliest gap first.",
        )

    st.divider()
    st.subheader("AI assistant")
    for problem in llm_problems:
        st.error(problem)
    if llm is not None:
        st.success(f"Connected: {llm.name} → built-in rules")
        for provider in llm.providers:
            st.caption(
                f"**{provider.provider}** ({provider.model}): {provider.request_count} request(s), "
                f"{provider.cache_hits} from cache, paced to {provider.rpm}/min per model. "
                f"Backup models: {', '.join(provider.models[1:]) or 'none'}."
            )
    else:
        st.info("No AI keys found, so AI features use PawPal+'s built-in rules. Set GEMINI_API_KEY or GROQ_API_KEY to use AI.")
    st.caption("If every AI provider is unavailable, the built-in rules take over. Every suggestion waits for your review.")

# ---- Header and day summary ----

st.title("🐾 PawPal+")
st.caption("Daily pet care planning that fits your time, your priorities, and your care rules.")

if "load_warning" in st.session_state:
    st.warning(st.session_state.pop("load_warning"))
if "flash_error" in st.session_state:
    st.error(st.session_state.pop("flash_error"))
if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"))

preview = Schedule(view_day, owner)  # not stored, so previewing doesn't add history
preview.generate()
metric_cols = st.columns(4)
metric_cols[0].metric("Tasks due", sum(len(p.due_tasks(view_day)) for p in owner.pets), border=True)
metric_cols[1].metric("Overdue", sum(t.is_overdue(view_day) for t in owner.all_tasks()), border=True)
metric_cols[2].metric("Time planned", f"{preview.total_minutes()} / {owner.available_minutes} min", border=True)
metric_cols[3].metric("Didn't fit", len(preview.skipped), border=True)

schedule_tab, tasks_tab, pets_tab, planner_tab = st.tabs(["Schedule", "Tasks", "Pets", "AI planner"], key="main_tab")
pets_by_name = {p.name: p for p in owner.pets}


def build_retriever() -> Retriever:
    """Search index over every pet's guidelines and the task history up to view_day."""
    return Retriever(build_documents(owner, view_day))


def task_status(task: Task) -> str:
    """Short status of `task` as of view_day."""
    if task.is_completed(view_day):
        return "✅ Done"
    if task.is_overdue(view_day):
        return f"⚠️ Overdue since {task.start_date:%b %d}"
    if not task.is_due(view_day):
        return f"🗓️ Next due {task.start_date:%b %d}"
    return "⏳ Due"


def when(task: Task) -> str:
    """The task's fixed time, or "flexible"."""
    return fmt(task.preferred_time) if task.preferred_time else "flexible"


def timeline_chart(schedule: Schedule) -> alt.Chart:
    """Gantt-style bars for the planned tasks across the owner's available window, colored by priority."""
    window_start = minutes_between(MIDNIGHT, schedule.owner.day_start)
    window_end = window_start + schedule.owner.available_minutes
    rows = []
    for start, task in schedule.planned:
        begin = minutes_between(MIDNIGHT, start)
        rows.append({
            "Task": task.label,
            "Start": begin,
            "End": begin + task.duration_minutes,
            "Priority": task.priority.name.title(),
            "Time": f"{fmt(start)}–{fmt(add_minutes(start, task.duration_minutes))}",
        })
    span = window_end - window_start
    step = 15 if span <= 90 else 30 if span <= 240 else 60
    ticks = list(range(window_start - window_start % step + step * (window_start % step > 0), window_end + 1, step))
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar(cornerRadius=4)
        .encode(
            # Minutes since midnight, labelled as HH:MM (utcFormat avoids shifting by the viewer's time zone).
            x=alt.X("Start:Q", scale=alt.Scale(domain=[window_start, window_end]),
                    axis=alt.Axis(title=None, values=ticks, labelExpr="utcFormat(datum.value * 60000, '%H:%M')")),
            x2="End:Q",
            y=alt.Y("Task:N", sort=None, title=None),
            color=alt.Color("Priority:N", scale=alt.Scale(domain=list(PRIORITY_COLORS), range=list(PRIORITY_COLORS.values())),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["Task", "Time", "Priority"],
        )
        .properties(height=max(90, 34 * len(rows)))
    )


def show_plan(schedule: Schedule, key: str) -> None:
    """Summary banner, timeline, plan table, skipped table, and conflict notes for `schedule`."""
    planned, skipped = len(schedule.planned), len(schedule.skipped)
    used = f"{schedule.total_minutes()} of {schedule.owner.available_minutes} min used"
    if not planned and not skipped:
        st.info("No tasks are due on this day.")
    elif not skipped:
        st.success(f"All {planned} due task{'s' if planned != 1 else ''} fit in the day ({used}).")
    else:
        st.warning(f"Planned {planned} of {planned + skipped} due tasks ({used}). {skipped} didn't fit; see below.")
    critical = [t.label for t in schedule.skipped if t.priority == Priority.CRITICAL]
    if critical:
        st.error(f"Critical task not planned: {', '.join(critical)}. Free up time or change its fixed time.")

    if planned:
        st.altair_chart(timeline_chart(schedule), width="stretch", key=f"{key}_timeline")
        table = pd.DataFrame(schedule_rows(schedule))
        table["Priority"] = table["Priority"].map(PRIORITY_BADGES)
        st.dataframe(table, hide_index=True, key=f"{key}_plan", column_config={"Minutes": MINUTES, "Why": st.column_config.TextColumn(width="large")})

    if skipped:
        st.markdown("**Didn't fit**")
        st.dataframe(
            pd.DataFrame({"Task": t.label, "Minutes": t.duration_minutes, "Priority": badge(t), "Why": schedule.reasons[t]} for t in schedule.skipped),
            hide_index=True, key=f"{key}_skipped", column_config={"Minutes": MINUTES},
        )
    for message in schedule.conflict_messages():
        st.caption(f"Time conflict: {message}")
    if schedule.has_conflicts():
        st.error("Some planned tasks still overlap. Adjust their preferred times.")


# ---- Pets ----

with pets_tab:
    add_col, list_col = st.columns([2, 3], gap="large")
    with add_col:
        st.subheader("Add a pet")
        with st.form("add_pet", clear_on_submit=True):
            pet_name = st.text_input("Name", placeholder="Mochi")
            col1, col2 = st.columns(2)
            species = col1.selectbox("Species", ["dog", "cat", "other"])
            pet_age = col2.number_input("Age (years)", min_value=0, max_value=40, value=3)
            pet_notes = st.text_input("Notes (optional)", placeholder="Senior, sensitive stomach")
            if st.form_submit_button("Add pet", type="primary", width="stretch"):
                try:
                    if not pet_name.strip():
                        raise ValueError("give your pet a name")
                    owner.add_pet(Pet(pet_name.strip(), species, age=int(pet_age), notes=pet_notes.strip()))
                    st.session_state.flash = f"Added {pet_name.strip()}."
                    st.rerun()
                except ValueError as e:
                    st.error(f"Couldn't add pet: {e}")

    with list_col:
        st.subheader("Your pets")
        if not owner.pets:
            st.info("No pets yet. Add your first pet to start planning.")
        else:
            st.dataframe(
                pd.DataFrame(
                    {"Pet": p.name, "Species": p.species.capitalize(), "Age": p.age, "Notes": p.notes,
                     "Tasks": sum(not t.is_completed(view_day) for t in p.tasks), "Guidelines": len(p.guidelines)}
                    for p in owner.pets
                ),
                hide_index=True,
            )

            st.subheader("Care guidelines and notes")
            st.caption(
                "Write your own care rules, one per line. They're used to check tasks, suggest ideas, and plan, "
                "and cited as e.g. [Mochi guideline 1]. The built-in rules understand shapes like "
                "\"At least 60 minutes of walks a day\", \"Medication before 09:00\", \"No food after 20:00\", "
                "and \"Brush coat twice a week\"."
            )
            edit_pet = pets_by_name[st.selectbox("Pet", list(pets_by_name), key="guideline_pet")]
            notes_text = st.text_input("Notes", value=edit_pet.notes, key=f"notes_{edit_pet.name}")
            guideline_text = st.text_area(
                "Guidelines (one per line)",
                value="\n".join(edit_pet.guidelines),
                key=f"guidelines_{edit_pet.name}",
                placeholder="At least 60 minutes of walks a day\nMedication must be given before 09:00",
            )
            save_col, remove_col = st.columns([1, 1])
            if save_col.button("Save", type="primary"):
                edit_pet.notes = notes_text.strip()
                edit_pet.guidelines = [line.strip() for line in guideline_text.splitlines() if line.strip()]
                st.session_state.flash = f"Saved {len(edit_pet.guidelines)} guideline(s) for {edit_pet.name}."
                st.rerun()
            with remove_col.popover(f"Remove {edit_pet.name}"):
                st.write(f"Remove **{edit_pet.name}** and all of their tasks? This can't be undone.")
                if st.button("Remove pet", type="primary", key="confirm_remove_pet"):
                    owner.remove_pet(edit_pet)
                    clear_ai_state()
                    st.session_state.flash = f"Removed {edit_pet.name}."
                    st.rerun()

# ---- Tasks ----


def complete(task: Task) -> None:
    """Button callback: complete `task` on view_day and leave a message for the next run."""
    next_task = task.pet.complete_task(task, view_day)  # also adds the next occurrence
    st.session_state.flash = f"Marked {task.label} done. Next due {next_task.start_date:%a %b %d}."


def remove(task: Task) -> None:
    """Button callback: remove `task` from its pet and leave a message for the next run."""
    label = task.label
    task.pet.remove_task(task)
    st.session_state.flash = f"Removed {label}."


def accept_idea(index: int) -> None:
    """Button callback: add an accepted idea as a real task."""
    idea = st.session_state.ideas.pop(index)
    try:
        pets_by_name[idea.pet_name].add_task(idea.to_task(view_day))
        st.session_state.flash = f"Added {idea.pet_name}: {idea.title}."
    except (KeyError, ValueError) as e:
        st.session_state.flash_error = f"Couldn't add {idea.title}: {e}"


def reject_idea(index: int) -> None:
    """Button callback: drop an idea."""
    st.session_state.ideas.pop(index)


def timing_inputs(key: str, task: Task | TaskDraft | None = None, after_options: list[str] | None = None) -> dict:
    """Optional time-window (and ordering-rule) widgets for use inside a form. Returns Task keyword arguments."""
    not_before, done_by = getattr(task, "not_before", None), getattr(task, "done_by", None)
    col1, col2 = st.columns(2)
    use_not_before = col1.checkbox("Not before", value=not_before is not None, key=f"{key}_use_nb")
    not_before_time = col1.time_input("Not before", value=not_before or owner.day_start, step=300,
                                      key=f"{key}_nb", label_visibility="collapsed")
    use_done_by = col2.checkbox("Done by", value=done_by is not None, key=f"{key}_use_db")
    done_by_time = col2.time_input("Done by", value=done_by or owner.day_end, step=300,
                                   key=f"{key}_db", label_visibility="collapsed")
    result = {"not_before": not_before_time if use_not_before else None, "done_by": done_by_time if use_done_by else None}
    if after_options is not None:
        after = getattr(task, "after", "")
        options = ["(none)", *sorted(set(after_options) | ({after} if after else set()))]
        col1, col2 = st.columns(2)
        chosen = col1.selectbox("Comes after", options, index=options.index(after) if after else 0, key=f"{key}_after",
                                help="Another task of the same pet that must finish first.")
        gap = col2.number_input("Minutes after it", min_value=0, max_value=240, step=5,
                                value=getattr(task, "after_gap_minutes", 0), key=f"{key}_gap")
        result |= {"after": "" if chosen == "(none)" else chosen, "after_gap_minutes": int(gap)}
    return result


def check_predecessor(pet: Pet, after: str) -> None:
    """Raise ValueError if `after` names a task this pet doesn't have."""
    if after and not any(t.title.lower() == after.lower() for t in pet.tasks):
        raise ValueError(f"{pet.name} has no task called {after!r} to come after")


# "flexible" sorts after every HH:MM string, so flexible tasks come last when sorting by time.
SORT_KEYS = {
    "Time": when,
    "Pet": lambda t: (t.pet_name, t.title),
    "Priority": lambda t: -t.priority,
    "Duration": lambda t: t.duration_minutes,
}
# Completed occurrences are kept as history, so hide them by default.
STATUS_FILTERS = {"Not completed": False, "Completed": True, "All": None}

with tasks_tab:
    if not owner.pets:
        st.info("Add a pet in the Pets tab first.")
    else:
        st.subheader("Add a task")
        describe_col, form_col = st.columns(2, gap="large")
        with describe_col:
            st.markdown("**Describe it in your own words**")
            nl_text = st.text_input(
                "Description", label_visibility="collapsed",
                placeholder="Walk Mochi for 30 minutes every morning at 7:30, it's important",
            )
            if st.button("Draft task", disabled=not nl_text.strip()):
                try:
                    with st.spinner("Drafting..."):
                        st.session_state.task_draft = ai_or_rules(
                            lambda m: parse_task(m, nl_text, owner), lambda: parse_task_rules(nl_text, owner)
                        )
                except ValueError as e:
                    st.error(f"Couldn't draft that task: {e}")

            if "task_draft" in st.session_state:
                draft, note = st.session_state.task_draft
                draft: TaskDraft
                st.caption(note)
                with st.form("review_draft", border=True):
                    st.caption("Check the draft, fix anything that's wrong, then confirm.")
                    pet_names = list(pets_by_name)
                    col1, col2, col3 = st.columns(3)
                    draft_pet = col1.selectbox("Pet", pet_names, index=pet_names.index(draft.pet_name) if draft.pet_name in pet_names else 0)
                    draft_title = col2.text_input("Task", value=draft.title)
                    draft_duration = col3.number_input("Minutes", min_value=1, max_value=240, value=min(draft.duration_minutes, 240))
                    col1, col2, col3 = st.columns(3)
                    draft_priority = col1.selectbox("Priority", PRIORITY_NAMES, index=PRIORITY_NAMES.index(draft.priority))
                    draft_frequency = col2.selectbox("Repeats", ["daily", "weekly"], index=["daily", "weekly"].index(draft.frequency))
                    draft_time = col3.time_input("Fixed time", value=draft.preferred_time or owner.day_start, step=300)
                    draft_fixed = st.checkbox("Use a fixed time", value=draft.preferred_time is not None)
                    draft_timing = timing_inputs("draft", draft)
                    confirm_col, discard_col = st.columns(2)
                    confirmed = confirm_col.form_submit_button("Confirm and add", type="primary", width="stretch")
                    discarded = discard_col.form_submit_button("Discard", width="stretch")
                if confirmed:
                    try:
                        pets_by_name[draft_pet].add_task(Task(
                            draft_title.strip(), int(draft_duration), draft_priority,
                            preferred_time=draft_time if draft_fixed else None, frequency=draft_frequency, start_date=view_day,
                            **draft_timing,
                        ))
                        st.session_state.pop("task_draft")
                        st.session_state.flash = f"Added {draft_title} for {draft_pet}."
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                if discarded:
                    st.session_state.pop("task_draft")
                    st.rerun()

        with form_col:
            st.markdown("**Or fill in the details**")
            with st.form("add_task", clear_on_submit=True, border=True):
                col1, col2 = st.columns(2)
                task_pet = col1.selectbox("Pet", list(pets_by_name))
                task_title = col2.text_input("Task", placeholder="Morning walk")
                col1, col2, col3 = st.columns(3)
                duration = col1.number_input("Minutes", min_value=1, max_value=240, value=20)
                priority = col2.selectbox("Priority", PRIORITY_NAMES, index=PRIORITY_NAMES.index("medium"))
                frequency = col3.selectbox("Repeats", ["daily", "weekly"])
                col1, col2 = st.columns(2)
                preferred_time = col1.time_input("Fixed time", value=owner.day_start, step=300)
                has_fixed_time = col2.checkbox("Use a fixed time")
                with st.expander("Timing rules (optional)"):
                    timing = timing_inputs("new", after_options=sorted({t.title for t in owner.all_tasks()}))
                if st.form_submit_button("Add task", type="primary", width="stretch"):
                    try:
                        if not task_title.strip():
                            raise ValueError("give the task a name")
                        check_predecessor(pets_by_name[task_pet], timing["after"])
                        pets_by_name[task_pet].add_task(Task(
                            task_title.strip(), int(duration), priority,
                            preferred_time=preferred_time if has_fixed_time else None, frequency=frequency, start_date=view_day,
                            **timing,
                        ))
                        st.session_state.flash = f"Added {task_title.strip()} for {task_pet}."
                        st.rerun()
                    except ValueError as e:
                        st.error(f"Couldn't add task: {e}")

        st.divider()
        st.subheader("Task list")
        if not owner.all_tasks():
            st.info("No tasks yet. Add one above.")
        else:
            col1, col2, col3 = st.columns(3)
            pet_filter = col1.selectbox("Pet", ["All", *pets_by_name], key="pet_filter")
            status_filter = col2.selectbox("Status", list(STATUS_FILTERS))
            sort_by = col3.selectbox("Sort by", list(SORT_KEYS))
            filtered = owner.filter_tasks(
                pet_name=None if pet_filter == "All" else pet_filter, completed=STATUS_FILTERS[status_filter], day=view_day,
            )
            filtered.sort(key=SORT_KEYS[sort_by])

            if not filtered:
                st.info("No tasks match these filters.")
            else:
                st.dataframe(
                    pd.DataFrame(
                        {"Time": when(t), "Pet": t.pet_name, "Task": t.title, "Minutes": t.duration_minutes,
                         "Priority": badge(t), "Repeats": t.frequency, "Timing rules": t.timing_label, "Status": task_status(t)}
                        for t in filtered
                    ),
                    hide_index=True, column_config={"Minutes": MINUTES},
                )
                overdue = sum(t.is_overdue(view_day) for t in filtered)
                if overdue:
                    count = "1 task is" if overdue == 1 else f"{overdue} tasks are"
                    st.warning(f"{count} overdue. Overdue tasks gain priority each day until they're done.")
                else:
                    # Judge "caught up" on every task for the chosen pet(s), not just the rows the status filter shows.
                    pet_tasks = owner.filter_tasks(pet_name=None if pet_filter == "All" else pet_filter, day=view_day)
                    if not any(t.is_due(view_day) for t in pet_tasks) and any(t.is_completed(view_day) for t in pet_tasks):
                        st.success(f"All caught up for {view_day:%b %d}.")

                # Buttons use on_click callbacks so the change is applied before the page redraws.
                picked_col, done_col, edit_col, remove_col = st.columns([4, 1, 1, 1], vertical_alignment="bottom")
                # Pick by position: Streamlit deep-copies widget values, so selecting the Task itself would return a copy.
                picked = filtered[picked_col.selectbox(
                    "Selected task", range(len(filtered)),
                    format_func=lambda i: f"{filtered[i].label} · {when(filtered[i])} · {task_status(filtered[i])}",
                )]
                done_col.button("Mark done", disabled=not picked.is_due(view_day), on_click=complete, args=(picked,), width="stretch")
                with edit_col.popover("Edit", width="stretch"):
                    with st.form(f"edit_{picked.id}", border=False):  # keyed by task, so defaults follow the selection
                        st.markdown(f"**{picked.label}**")
                        col1, col2 = st.columns(2)
                        edit_minutes = col1.number_input("Minutes", min_value=1, max_value=240, value=min(picked.duration_minutes, 240))
                        edit_priority = col2.selectbox("Priority", PRIORITY_NAMES, index=PRIORITY_NAMES.index(picked.priority.name.lower()))
                        col1, col2 = st.columns(2)
                        edit_time = col1.time_input("Fixed time", value=picked.preferred_time or owner.day_start, step=300)
                        edit_fixed = col2.checkbox("Use a fixed time", value=picked.preferred_time is not None)
                        siblings = sorted({t.title for t in picked.pet.tasks if t.title.lower() != picked.title.lower()})
                        edit_timing = timing_inputs(f"edit_{picked.id}", picked, after_options=siblings)
                        if st.form_submit_button("Save changes", type="primary", width="stretch"):
                            try:
                                picked.update(
                                    duration_minutes=int(edit_minutes), priority=edit_priority,
                                    preferred_time=edit_time if edit_fixed else None, **edit_timing,
                                )
                                st.session_state.flash = f"Updated {picked.label}."
                                st.rerun()
                            except ValueError as e:
                                st.error(f"Not saved: {e}")
                with remove_col.popover("Remove", width="stretch"):
                    st.write(f"Remove **{picked.label}**? This can't be undone.")
                    st.button("Remove task", type="primary", on_click=remove, args=(picked,), key="confirm_remove_task")

        st.divider()
        ideas_col, check_col = st.columns(2, gap="large")
        with ideas_col:
            st.subheader("Task ideas")
            st.caption("Suggestions based on the pet's species, age, notes, your guidelines, and recent history.")
            idea_pet = st.selectbox("Ideas for", list(pets_by_name), key="idea_pet")
            if st.button("Suggest tasks"):
                pet, retriever = pets_by_name[idea_pet], build_retriever()
                with st.spinner("Thinking of ideas..."):
                    st.session_state.ideas, st.session_state.ideas_note = ai_or_rules(
                        lambda m: suggest_tasks(m, owner, pet, retriever, view_day),
                        lambda: suggest_tasks_rules(owner, pet, retriever, view_day),
                    )
                if not st.session_state.ideas:
                    st.info("No new ideas. This pet's tasks already look complete.")
            if st.session_state.get("ideas"):
                st.caption(st.session_state.get("ideas_note", ""))
            for i, idea in enumerate(st.session_state.get("ideas", [])):
                with st.container(border=True):
                    text_col, accept_col, reject_col = st.columns([5, 1, 1], vertical_alignment="center")
                    idea_time = fmt(idea.preferred_time) if idea.preferred_time else "flexible"
                    text_col.markdown(
                        f"**{idea.pet_name}: {idea.title}** · {idea.duration_minutes} min · {idea.priority} · "
                        f"{idea_time} · {idea.frequency}  \n{idea.rationale}"
                    )
                    accept_col.button("Add", key=f"accept_idea_{i}", on_click=accept_idea, args=(i,), type="primary")
                    reject_col.button("Skip", key=f"reject_idea_{i}", on_click=reject_idea, args=(i,))

        with check_col:
            st.subheader("Guideline check")
            st.caption("Compares every pet's tasks with the care guidelines you wrote.")
            has_guidelines = any(p.guidelines for p in owner.pets)
            if not has_guidelines:
                st.caption("Add care guidelines in the Pets tab to use this check.")
            if st.button("Check tasks", disabled=not has_guidelines):
                with st.spinner("Checking..."):
                    issues, note = ai_or_rules(
                        lambda m: check_guidelines(m, owner, build_retriever(), view_day),
                        lambda: check_guidelines_rules(owner, view_day),
                    )
                st.caption(note)
                if not issues:
                    st.success("Your tasks follow every guideline that was checked.")
                for issue in issues:
                    st.warning(
                        f"**{issue.pet_name}:** {issue.issue}  \n"
                        f"Suggestion: {issue.suggestion or 'none'}  \n"
                        f"Source: [{issue.ref}] _{issue.guideline}_"
                    )
                if "built-in rules" in note and (unchecked := unchecked_guidelines(owner)):
                    st.caption(f"The built-in rules couldn't interpret, so didn't check: {', '.join(unchecked)}.")

# ---- Schedule ----

with schedule_tab:
    if not owner.all_tasks():
        st.info("Add pets and tasks, then come back here to build the day's plan.")
    else:
        title_col, button_col = st.columns([4, 1], vertical_alignment="bottom")
        title_col.subheader(f"Plan for {view_day:%A, %B %d}")
        generate = button_col.button("Generate schedule", type="primary", width="stretch")

        # Once generated, the day's schedule is rebuilt on every run so it reflects later task edits.
        if generate or view_day in owner.schedules:
            schedule = owner.create_schedule(view_day)
            show_plan(schedule, key="day")

            if schedule.planned:
                ics_col, csv_col, slot_col = st.columns([1, 1, 2], vertical_alignment="bottom")
                ics_col.download_button(
                    "Add to calendar (.ics)", schedule_to_ics(schedule), file_name=f"pawpal-{view_day}.ics",
                    mime="text/calendar", width="stretch",
                )
                csv_col.download_button(
                    "Download CSV", pd.DataFrame(schedule_rows(schedule)).to_csv(index=False),
                    file_name=f"pawpal-{view_day}.csv", mime="text/csv", width="stretch",
                )
                with slot_col:
                    slot_minutes = st.number_input("Find a free slot for (minutes)", min_value=5, max_value=240, value=15, step=5)
                    slot = schedule.next_available_slot(int(slot_minutes))
                    st.caption(f"Next free slot: **{fmt(slot)}**" if slot else "No free slot that long today.")

            st.divider()
            if st.button("Explain this plan"):
                with st.spinner("Explaining..."):
                    text, note = ai_or_rules(
                        lambda m: explain_schedule(m, schedule, build_retriever()), lambda: explain_schedule_rules(schedule)
                    )
                st.session_state.ai_explanation = (view_day, text, note)
            explanation = st.session_state.get("ai_explanation")
            if explanation and explanation[0] == view_day:
                st.info(explanation[1])
                st.caption(f"{explanation[2]} The table above is the source of truth.")
            with st.expander("Scheduler's step-by-step reasoning"):
                st.code(schedule.explain(), language=None)
        else:
            st.info("Click **Generate schedule** to plan this day. The numbers above preview it.")

# ---- AI planner ----


def run_planner(goal_text: str) -> None:
    """Plan view_day for the goal (Gemini agent, or the built-in rules) and keep the proposal for review."""
    retriever = build_retriever()
    with st.spinner("Planning..."):
        st.session_state.proposal, st.session_state.proposal_note = ai_or_rules(
            lambda m: PlanningAgent(m, owner, view_day, retriever).run(goal_text),
            lambda: plan_rules(owner, view_day, goal_text),
        )


with planner_tab:
    st.subheader("Plan my day")
    st.caption(
        "Describe what today needs. The planner adjusts task settings and reruns the scheduler on a copy of your "
        "data. Nothing changes until you approve."
    )
    goal = st.text_area(
        "What should the plan achieve?",
        placeholder="I only have 60 minutes today. Make sure medication and feeding still happen.",
        key="agent_goal",
    )
    if st.button("Draft a plan", type="primary", disabled=not goal.strip() or not owner.all_tasks()):
        run_planner(goal)

    proposal = st.session_state.get("proposal")
    if proposal:
        st.divider()
        st.markdown(f"#### Proposed plan for {proposal.day:%A, %b %d}")
        st.caption(st.session_state.get("proposal_note", ""))
        if proposal.day != view_day:
            st.caption(f"This proposal is for {proposal.day:%b %d}, not the day selected in the sidebar.")
        if proposal.summary:
            st.info(proposal.summary)
        for warning in proposal.warnings:
            st.warning(warning)

        st.markdown("**Proposed changes**")
        if proposal.changes:
            st.markdown("\n".join(f"- {change['description']}" for change in proposal.changes))
        else:
            st.caption("No changes.")

        show_plan(proposal.schedule, key="proposal")

        if proposal.steps:
            with st.expander(f"Agent steps ({proposal.turns} model call(s), {len(proposal.steps)} action(s))"):
                shown_turn = None
                for step in proposal.steps:
                    if step.turn != shown_turn:
                        shown_turn = step.turn
                        st.markdown(f"**Turn {step.turn}**" + (f": _{step.thought}_" if step.thought else ""))
                    st.markdown(f"`{step.tool}` `{json.dumps(step.args)}`")
                    st.code(step.result, language=None)

        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve plan", type="primary", width="stretch"):
            try:
                apply_changes(owner, proposal.changes, proposal.day)
                owner.create_schedule(proposal.day)
                st.session_state.pop("proposal")
                st.session_state.flash = f"Plan applied ({len(proposal.changes)} change(s))."
                st.rerun()
            except (TypeError, ValueError) as e:
                st.error(f"Couldn't apply the plan (your tasks may have changed since it was drafted): {e}")
        if reject_col.button("Reject", width="stretch"):
            st.session_state.pop("proposal")
            st.rerun()

        feedback = st.text_input("Or tell the planner what to change", placeholder="Keep Fetch, defer brushing instead")
        if st.button("Revise with feedback", disabled=not feedback.strip()):
            run_planner(f"{proposal.goal}\n\nFeedback on your previous proposal: {feedback}")
            st.rerun()

# ---- Autosave, then the data controls (last, so the export includes this run's changes) ----

saved_text = storage.dumps(owner)
if st.session_state.get("saved_text") != saved_text:
    try:
        storage.save(owner)
        st.session_state.saved_text = saved_text
    except OSError as e:
        st.sidebar.error(f"Couldn't save your data: {e}")

with st.sidebar:
    st.divider()
    st.subheader("Your data")
    try:
        shown_path = storage.DEFAULT_PATH.relative_to(Path.cwd())
    except ValueError:
        shown_path = storage.DEFAULT_PATH
    st.caption(f"Changes save automatically to `{shown_path}`.")
    st.download_button("Export (JSON)", saved_text, file_name="pawpal-data.json", mime="application/json", width="stretch")
    st.file_uploader("Import (JSON)", type="json", key="import_file", on_change=import_data, help="Replaces all current data.")
