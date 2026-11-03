import json

import pandas as pd
import streamlit as st
from ai_assistant import (
    PlanningAgent,
    TaskDraft,
    apply_changes,
    explain_schedule,
    parse_task,
    suggest_tasks,
)
from llm import GeminiLLM, LLMError, get_api_key
from pawpal_system import Owner, Pet, Priority, Schedule, Task, add_minutes, fmt
from rag import Retriever, build_documents, check_guidelines

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")

st.markdown(
    """
Welcome to the PawPal+ starter app.

This file is intentionally thin. It gives you a working Streamlit app so you can start quickly,
but **it does not implement the project logic**. Your job is to design the system and build it.

Use this app as your interactive demo once your backend classes/functions exist.
"""
)

with st.expander("Scenario", expanded=True):
    st.markdown(
        """
**PawPal+** is a pet care planning assistant. It helps a pet owner plan care tasks
for their pet(s) based on constraints like time, priority, and preferences.

You will design and implement the scheduling logic and connect it to this Streamlit UI.
"""
    )

with st.expander("What you need to build", expanded=True):
    st.markdown(
        """
At minimum, your system should:
- Represent pet care tasks (what needs to happen, how long it takes, priority)
- Represent the pet and the owner (basic info and preferences)
- Build a plan/schedule for a day that chooses and orders tasks based on constraints
- Explain the plan (why each task was chosen and when it happens)
"""
    )

st.divider()

# The Owner object holds all pets and tasks; keeping it in session_state lets it survive reruns.
if "owner" not in st.session_state:
    st.session_state.owner = Owner("Jordan")
owner: Owner = st.session_state.owner


def get_llm() -> GeminiLLM | None:
    """The Gemini client, or None if there's no API key. A successful client is kept for the session."""
    if "llm" not in st.session_state:
        key = get_api_key()
        if not key:
            try:
                key = st.secrets.get("GEMINI_API_KEY")
            except Exception:  # no secrets.toml
                key = None
        if not key:
            return None
        try:
            st.session_state.llm = GeminiLLM(api_key=key)
        except (LLMError, ImportError) as e:
            st.sidebar.error(f"Gemini unavailable: {e}")
            return None
    return st.session_state.llm


llm = get_llm()
ai_ready = llm is not None
st.sidebar.markdown("### ✨ AI features")
if ai_ready:
    st.sidebar.success(f"Gemini connected ({llm.model}).")
else:
    st.sidebar.info("AI features are off. Set GEMINI_API_KEY (environment or .streamlit/secrets.toml) and restart.")
st.sidebar.caption("Every AI suggestion waits for your review before it changes anything.")

st.subheader("Owner")
col1, col2, col3 = st.columns(3)
with col1:
    owner_name = st.text_input("Owner name", value=owner.name)
with col2:
    day_start = st.time_input("Day starts at", value=owner.day_start)
with col3:
    available_minutes = st.number_input(
        "Available minutes", min_value=0, max_value=24 * 60, value=owner.available_minutes, step=15
    )

try:
    owner.set_window(day_start, int(available_minutes))  # validates before applying either value
    owner.name = owner_name
except ValueError as e:
    st.error(f"Owner settings not saved: {e}")

st.markdown("### Pets")

col1, col2, col3 = st.columns(3)
with col1:
    pet_name = st.text_input("Pet name", value="Mochi")
with col2:
    species = st.selectbox("Species", ["dog", "cat", "other"])
with col3:
    pet_age = st.number_input("Age (years)", min_value=0, max_value=40, value=3)

if st.button("Add pet"):
    try:
        owner.add_pet(Pet(pet_name.strip(), species, age=int(pet_age)))
        st.success(f"Added {pet_name}.")
    except ValueError as e:
        st.error(str(e))

if owner.pets:
    st.table(
        pd.DataFrame(
            {"Pet": p.name, "Species": p.species.capitalize(), "Age": p.age, "Tasks": p.task_count,
             "Guidelines": len(p.guidelines)}
            for p in owner.pets
        ).set_index("Pet")
    )

    with st.expander("📋 Care guidelines"):
        st.caption(
            "Write your own care rules for each pet, one per line. The AI checks tasks against them, "
            "uses them for task ideas and plans, and cites them as e.g. [Mochi guideline 1]."
        )
        guideline_pets = {p.name: p for p in owner.pets}
        guideline_pet = guideline_pets[st.selectbox("Guidelines for", list(guideline_pets), key="guideline_pet")]
        guideline_text = st.text_area(
            "One guideline per line",
            value="\n".join(guideline_pet.guidelines),
            key=f"guidelines_{guideline_pet.name}",
            placeholder="At least 60 minutes of walks a day\nMedication must be given before 09:00",
        )
        if st.button("Save guidelines"):
            guideline_pet.guidelines = [line.strip() for line in guideline_text.splitlines() if line.strip()]
            st.success(f"Saved {len(guideline_pet.guidelines)} guideline(s) for {guideline_pet.name}.")
else:
    st.info("No pets yet. Add one above.")

st.markdown("### Tasks")

pets_by_name = {p.name: p for p in owner.pets}
if not owner.pets:
    st.caption("Add a pet before adding tasks.")
else:
    st.markdown("#### ✨ Describe a task")
    nl_text = st.text_input(
        "Describe the task in your own words",
        placeholder="Walk Mochi for 30 minutes every morning at 7:30, it's important",
        disabled=not ai_ready,
    )
    if st.button("Draft task with AI", disabled=not ai_ready or not nl_text.strip()):
        try:
            with st.spinner("Drafting..."):
                st.session_state.task_draft = parse_task(llm, nl_text, owner)
        except (LLMError, ValueError) as e:
            st.error(f"Couldn't draft that task: {e}")

    draft: TaskDraft | None = st.session_state.get("task_draft")
    if draft:
        st.caption("Review the AI's draft. Fix anything that's wrong, then confirm.")
        with st.form("review_draft"):
            col1, col2, col3 = st.columns(3)
            with col1:
                draft_pet = st.selectbox("For pet", list(pets_by_name), index=list(pets_by_name).index(draft.pet_name))
            with col2:
                draft_title = st.text_input("Task title", value=draft.title)
            with col3:
                draft_duration = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=draft.duration_minutes)
            col1, col2, col3 = st.columns(3)
            with col1:
                draft_priority = st.selectbox("Priority", ["low", "medium", "high"], index=["low", "medium", "high"].index(draft.priority))
            with col2:
                draft_frequency = st.selectbox("Frequency", ["daily", "weekly"], index=["daily", "weekly"].index(draft.frequency))
            with col3:
                draft_fixed = st.checkbox("Fixed time?", value=draft.preferred_time is not None)
                draft_time = st.time_input("Preferred time", value=draft.preferred_time or owner.day_start)
            confirm_col, discard_col = st.columns(2)
            confirmed = confirm_col.form_submit_button("Confirm and add", type="primary")
            discarded = discard_col.form_submit_button("Discard")
        if confirmed:
            try:
                pets_by_name[draft_pet].add_task(
                    Task(
                        draft_title.strip(),
                        int(draft_duration),
                        draft_priority,
                        preferred_time=draft_time if draft_fixed else None,
                        frequency=draft_frequency,
                    )
                )
                st.session_state.pop("task_draft")
                st.session_state.flash = f"Added {draft_title} for {draft_pet}."
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        if discarded:
            st.session_state.pop("task_draft")
            st.rerun()

    st.markdown("#### Add a task manually")
    task_pet = st.selectbox("For pet", list(pets_by_name))

    col1, col2, col3 = st.columns(3)
    with col1:
        task_title = st.text_input("Task title", value="Morning walk")
    with col2:
        duration = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=20)
    with col3:
        priority = st.selectbox("Priority", ["low", "medium", "high"], index=2)

    col1, col2, col3 = st.columns(3)
    with col1:
        frequency = st.selectbox("Frequency", ["daily", "weekly"])
    with col2:
        has_fixed_time = st.checkbox("Fixed time?")
    with col3:
        preferred_time = st.time_input("Preferred time", value=owner.day_start, disabled=not has_fixed_time)

    if st.button("Add task"):
        try:
            pets_by_name[task_pet].add_task(
                Task(
                    task_title.strip(),
                    int(duration),
                    priority,
                    preferred_time=preferred_time if has_fixed_time else None,
                    frequency=frequency,
                )
            )
            st.success(f"Added {task_title} for {task_pet}.")
        except ValueError as e:
            st.error(str(e))

st.markdown("#### Task list")
view_day = st.date_input("Day", help="Task status, Done buttons, AI features, and the schedule below all use this day.")


def build_retriever() -> Retriever:
    """Search index over every pet's guidelines and the task history up to view_day."""
    return Retriever(build_documents(owner, view_day))


PRIORITY_BADGES = {Priority.HIGH: "🔴 High", Priority.MEDIUM: "🟡 Medium", Priority.LOW: "🟢 Low"}


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


def complete(task: Task) -> None:
    """Button callback: complete `task` on view_day and leave a message for the next run."""
    next_task = task.pet.complete_task(task, view_day)  # also adds the next occurrence
    st.session_state.flash = f"Marked {task.label} done. Next due {next_task.start_date:%a %b %d}."


def remove(task: Task) -> None:
    """Button callback: remove `task` from its pet and leave a message for the next run."""
    label = task.label
    task.pet.remove_task(task)
    st.session_state.flash = f"Removed {label}."


# "flexible" sorts after every HH:MM string, so flexible tasks come last when sorting by time.
SORT_KEYS = {
    "Time": when,
    "Pet": lambda t: (t.pet_name, t.title),
    "Priority": lambda t: -t.priority,
    "Duration": lambda t: t.duration_minutes,
}
# Completed occurrences are kept as history, so hide them by default.
STATUS_FILTERS = {"Not completed": False, "Completed": True, "All": None}

if not owner.all_tasks():
    st.info("No tasks yet. Add one above.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        pet_filter = st.selectbox("Show pet", ["All", *pets_by_name])
    with col2:
        status_filter = st.selectbox("Show status", list(STATUS_FILTERS))
    with col3:
        sort_by = st.selectbox("Sort by", list(SORT_KEYS))

    filtered = owner.filter_tasks(
        pet_name=None if pet_filter == "All" else pet_filter,
        completed=STATUS_FILTERS[status_filter],
        day=view_day,
    )
    filtered.sort(key=SORT_KEYS[sort_by])

    if not filtered:
        st.info("No tasks match these filters.")
    else:
        st.table(
            pd.DataFrame(
                {
                    "Time": when(task),
                    "Task": task.label,
                    "Minutes": task.duration_minutes,
                    "Priority": PRIORITY_BADGES[task.priority],
                    "Repeats": task.frequency,
                    "Status": task_status(task),
                }
                for task in filtered
            ).set_index("Time")
        )

        overdue = sum(task.is_overdue(view_day) for task in filtered)
        if overdue:
            count = "1 task is" if overdue == 1 else f"{overdue} tasks are"
            st.warning(f"{count} overdue. Overdue tasks win priority ties in the schedule until marked done.")
        else:
            # Judge "caught up" on every task for the chosen pet(s), not just the rows the status filter shows.
            pet_tasks = owner.filter_tasks(pet_name=None if pet_filter == "All" else pet_filter, day=view_day)
            if not any(t.is_due(view_day) for t in pet_tasks) and any(t.is_completed(view_day) for t in pet_tasks):
                st.success(f"All caught up for {view_day:%b %d}.")

        # Buttons use on_click callbacks so the change is applied before the page redraws.
        picked_col, done_col, remove_col = st.columns([4, 1, 1], vertical_alignment="bottom")
        with picked_col:
            # Pick by position: Streamlit deep-copies widget values, so selecting the Task itself would return a copy.
            picked = filtered[
                st.selectbox(
                    "Task",
                    range(len(filtered)),
                    format_func=lambda i: f"{filtered[i].label} · {when(filtered[i])} · {task_status(filtered[i])}",
                )
            ]
        done_col.button(
            "Mark done",
            disabled=not picked.is_due(view_day),
            on_click=complete,
            args=(picked,),
        )
        remove_col.button("Remove", on_click=remove, args=(picked,))

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

if owner.pets:
    st.markdown("#### ✨ Task ideas")
    st.caption("The AI suggests tasks based on the pet's species, age, notes, your guidelines, and recent history.")


    def accept_idea(index: int) -> None:
        """Button callback: add an accepted idea as a real task."""
        idea = st.session_state.ideas.pop(index)
        try:
            pet = next(p for p in owner.pets if p.name == idea.pet_name)
            pet.add_task(idea.to_task(view_day))
            st.session_state.flash = f"Added {idea.pet_name}: {idea.title}."
        except (StopIteration, ValueError) as e:
            st.session_state.flash = f"Couldn't add {idea.title}: {e}"


    def reject_idea(index: int) -> None:
        """Button callback: drop an idea."""
        st.session_state.ideas.pop(index)


    idea_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
    idea_pet = idea_col.selectbox("Ideas for", list(pets_by_name), key="idea_pet")
    if button_col.button("Suggest tasks", disabled=not ai_ready):
        try:
            with st.spinner("Thinking of ideas..."):
                st.session_state.ideas = suggest_tasks(llm, owner, pets_by_name[idea_pet], build_retriever(), view_day)
            if not st.session_state.ideas:
                st.info("No new ideas. This pet's tasks already look complete.")
        except LLMError as e:
            st.error(f"Couldn't get ideas: {e}")

    for i, idea in enumerate(st.session_state.get("ideas", [])):
        text_col, accept_col, reject_col = st.columns([6, 1, 1], vertical_alignment="center")
        idea_time = fmt(idea.preferred_time) if idea.preferred_time else "flexible"
        text_col.markdown(
            f"**{idea.pet_name}: {idea.title}** · {idea.duration_minutes} min · {idea.priority} · "
            f"{idea_time} · {idea.frequency}  \n_{idea.rationale}_"
        )
        accept_col.button("Accept", key=f"accept_idea_{i}", on_click=accept_idea, args=(i,))
        reject_col.button("Reject", key=f"reject_idea_{i}", on_click=reject_idea, args=(i,))

    st.markdown("#### ✨ Check tasks against guidelines")
    has_guidelines = any(p.guidelines for p in owner.pets)
    if not has_guidelines:
        st.caption("Add care guidelines under Pets to use this check.")
    if st.button("Check tasks", disabled=not ai_ready or not has_guidelines):
        try:
            with st.spinner("Checking..."):
                issues = check_guidelines(llm, owner, build_retriever(), view_day)
            if not issues:
                st.success("Your tasks follow every guideline.")
            for issue in issues:
                st.warning(
                    f"**{issue.pet_name}:** {issue.issue}  \n"
                    f"Suggestion: {issue.suggestion or 'none'}  \n"
                    f"Source: [{issue.ref}] _{issue.guideline}_"
                )
        except LLMError as e:
            st.error(f"Couldn't check guidelines: {e}")

st.divider()


def show_plan(schedule: Schedule) -> None:
    """Summary banner, plan table, skipped table, and conflict warnings for `schedule`."""
    planned, skipped = len(schedule.planned), len(schedule.skipped)
    used = f"{schedule.total_minutes()} of {schedule.owner.available_minutes} min used"

    if not planned and not skipped:
        st.info("No tasks are due on this day.")
    elif not skipped:
        st.success(f"All {planned} due task{'s' if planned != 1 else ''} fit in the day ({used}).")
    else:
        st.warning(f"Planned {planned} of {planned + skipped} due tasks ({used}). {skipped} didn't fit; see below.")

    if planned:
        st.table(
            pd.DataFrame(
                {
                    "Time": f"{fmt(start)}–{fmt(add_minutes(start, task.duration_minutes))}",
                    "Task": task.label,
                    "Minutes": task.duration_minutes,
                    "Priority": PRIORITY_BADGES[task.priority],
                    "Why": schedule.reasons[task],
                }
                for start, task in schedule.planned
            ).set_index("Time")
        )

    if skipped:
        st.markdown("**Skipped**")
        st.table(
            pd.DataFrame(
                {"Task": t.label, "Priority": PRIORITY_BADGES[t.priority], "Why": schedule.reasons[t]}
                for t in schedule.skipped
            ).set_index("Task")
        )

    for message in schedule.conflict_messages():
        st.warning(f"Time conflict: {message}")

    if schedule.has_conflicts():
        st.error("Some planned tasks still overlap. Adjust their preferred times.")


st.subheader("Build Schedule")
st.caption(f"For {view_day:%A, %b %d %Y}. Change the day in the task list above.")

# Once generated, the day's schedule is rebuilt on every run so it reflects later task edits.
if st.button("Generate schedule") or view_day in owner.schedules:
    schedule = owner.create_schedule(view_day)
    show_plan(schedule)

    with st.expander("Full explanation"):
        st.text(schedule.explain())

    if st.button("✨ Explain with AI", disabled=not ai_ready):
        try:
            with st.spinner("Explaining..."):
                st.session_state.ai_explanation = (view_day, explain_schedule(llm, schedule, build_retriever()))
        except LLMError as e:
            st.error(f"Couldn't explain the schedule: {e}")
    explanation = st.session_state.get("ai_explanation")
    if explanation and explanation[0] == view_day:
        st.info(explanation[1])
        st.caption("AI summary of the plan. The table above is the source of truth.")

st.divider()

st.subheader("✨ Plan for me")
st.caption(
    "The planning agent adjusts task settings and runs the scheduler until your goal is met, "
    "working on a copy of your data. Nothing changes until you approve its proposal."
)
goal = st.text_area(
    "What should the plan achieve?",
    placeholder="I only have 60 minutes today. Make sure medication and feeding still happen.",
    key="agent_goal",
)


def run_agent(goal_text: str) -> None:
    """Run the planning agent for view_day and keep its proposal for review."""
    try:
        with st.spinner("Planning..."):
            st.session_state.proposal = PlanningAgent(llm, owner, view_day, build_retriever()).run(goal_text)
    except LLMError as e:
        st.error(f"The planning agent failed: {e}")


if st.button("Draft a plan", disabled=not ai_ready or not goal.strip() or not owner.all_tasks()):
    run_agent(goal)

proposal = st.session_state.get("proposal")
if proposal:
    st.markdown(f"**Proposed plan for {proposal.day:%A, %b %d}**")
    if proposal.day != view_day:
        st.caption(f"This proposal is for {proposal.day:%b %d}, not the day selected above.")
    if proposal.summary:
        st.info(proposal.summary)
    for warning in proposal.warnings:
        st.warning(warning)

    st.markdown("**Proposed changes**")
    if proposal.changes:
        st.markdown("\n".join(f"- {change['description']}" for change in proposal.changes))
    else:
        st.caption("No changes. Your current settings already meet the goal.")

    show_plan(proposal.schedule)

    with st.expander(f"Agent steps ({len(proposal.steps)})"):
        for i, step in enumerate(proposal.steps, 1):
            st.markdown(f"**{i}. {step.tool}** `{json.dumps(step.args)}`" + (f"  \n_{step.thought}_" if step.thought else ""))
            st.code(step.result, language=None)

    approve_col, reject_col = st.columns(2)
    if approve_col.button("Approve plan", type="primary"):
        try:
            apply_changes(owner, proposal.changes, proposal.day)
            owner.create_schedule(proposal.day)
            st.session_state.pop("proposal")
            st.toast(f"Plan applied ({len(proposal.changes)} change(s)).")
            st.rerun()
        except (TypeError, ValueError) as e:
            st.error(f"Couldn't apply the plan (your tasks may have changed since it was drafted): {e}")
    if reject_col.button("Reject"):
        st.session_state.pop("proposal")
        st.rerun()

    feedback = st.text_input("Or tell the agent what to change", placeholder="Keep Fetch, defer brushing instead")
    if st.button("Revise with feedback", disabled=not ai_ready or not feedback.strip()):
        run_agent(f"{proposal.goal}\n\nFeedback on your previous proposal: {feedback}")
        st.rerun()
