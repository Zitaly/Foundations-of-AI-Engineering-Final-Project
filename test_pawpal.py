from datetime import date, time, timedelta

import pytest

from pawpal_system import Owner, Pet, Priority, Task

DAY = date(2026, 10, 5)  # a Monday


def make_owner(**kwargs) -> tuple[Owner, Pet]:
    owner = Owner("Jordan", available_minutes=kwargs.pop("available_minutes", 120), day_start=time(8, 0), **kwargs)
    pet = Pet("Mochi", "dog")
    owner.add_pet(pet)
    return owner, pet


def starts(schedule) -> dict[str, str]:
    return {task.title: f"{start:%H:%M}" for start, task in schedule.planned}


def test_mark_complete_changes_task_status():
    day = date(2026, 10, 5)
    task = Task("Morning walk", 30, "high", start_date=day)
    assert task.last_completed is None
    assert task.is_due(day)

    task.mark_complete(day)

    assert task.last_completed == day
    assert not task.is_due(day)


def test_add_task_increases_pet_task_count():
    pet = Pet("Mochi", "dog")
    assert pet.task_count == 0

    pet.add_task(Task("Breakfast", 10, "high"))

    assert pet.task_count == 1


# 1. Daily limit is applied after placement


def test_limit_ignores_tasks_that_could_not_be_placed():
    owner, pet = make_owner(max_tasks_per_day=2)
    pet.add_task(Task("Too early", 10, "high", preferred_time=time(6, 0), start_date=DAY))
    pet.add_task(Task("Walk", 20, "medium", start_date=DAY))
    pet.add_task(Task("Brush", 10, "low", start_date=DAY))

    schedule = owner.create_schedule(DAY)

    assert set(starts(schedule)) == {"Walk", "Brush"}
    assert [t.title for t in schedule.skipped] == ["Too early"]


def test_limit_cuts_lowest_ranked_placed_tasks():
    owner, pet = make_owner(max_tasks_per_day=1)
    pet.add_task(Task("Walk", 20, "high", start_date=DAY))
    brush = Task("Brush", 10, "low", start_date=DAY)
    pet.add_task(brush)

    schedule = owner.create_schedule(DAY)

    assert list(starts(schedule)) == ["Walk"]
    assert "daily limit" in schedule.reasons[brush]


# 2. Clashing fixed tasks are moved instead of overlapping


def test_clashing_fixed_task_moves_to_nearest_free_slot():
    owner, pet = make_owner()
    pet.add_task(Task("Breakfast", 30, "high", preferred_time=time(9, 0), start_date=DAY))
    meds = Task("Meds", 10, "low", preferred_time=time(9, 20), start_date=DAY)
    pet.add_task(meds)

    schedule = owner.create_schedule(DAY)

    assert starts(schedule) == {"Breakfast": "09:00", "Meds": "09:30"}
    assert not schedule.has_conflicts()
    assert "moved from preferred time 09:20" in schedule.reasons[meds]


def test_clashing_fixed_task_skipped_when_no_slot_fits():
    owner, pet = make_owner(available_minutes=30)
    pet.add_task(Task("Walk", 30, "high", preferred_time=time(8, 0), start_date=DAY))
    meds = Task("Meds", 10, "low", preferred_time=time(8, 10), start_date=DAY)
    pet.add_task(meds)

    schedule = owner.create_schedule(DAY)

    assert schedule.skipped == [meds]
    assert "no other free slot fits" in schedule.reasons[meds]


# 7. Missed occurrences carry over


def test_missed_weekly_task_stays_due_until_completed():
    task = Task("Bath", 30, frequency="weekly", start_date=DAY)
    wednesday = DAY + timedelta(days=2)

    assert task.is_due(wednesday)
    assert task.is_overdue(wednesday)

    task.mark_complete(wednesday)

    assert not task.is_due(wednesday)
    assert not task.is_due(DAY + timedelta(days=7))  # a completed occurrence never comes back


# Completing a recurring task creates the next occurrence


def test_completing_daily_task_creates_tomorrows_instance():
    _, pet = make_owner()
    walk = Task("Walk", 20, "high", preferred_time=time(9, 0), start_date=DAY)
    pet.add_task(walk)

    next_walk = pet.complete_task(walk, DAY)

    assert next_walk is not walk
    assert pet.tasks == [walk, next_walk]
    assert walk.is_completed(DAY)
    assert next_walk.start_date == DAY + timedelta(days=1)
    assert next_walk.last_completed is None
    assert (next_walk.title, next_walk.duration_minutes, next_walk.priority, next_walk.preferred_time, next_walk.pet_name) == (
        "Walk", 20, walk.priority, time(9, 0), "Mochi"
    )
    assert not next_walk.is_due(DAY)
    assert next_walk.is_due(DAY + timedelta(days=1))


def test_completing_weekly_task_late_keeps_its_weekday():
    _, pet = make_owner()
    bath = Task("Bath", 30, frequency="weekly", start_date=DAY)  # Monday
    pet.add_task(bath)

    next_bath = pet.complete_task(bath, DAY + timedelta(days=2))  # done Wednesday

    assert next_bath.start_date == DAY + timedelta(days=7)  # next Monday, not next Wednesday
    assert [t for t in pet.tasks if t.is_due(DAY + timedelta(days=7))] == [next_bath]


def test_completing_a_task_twice_is_rejected():
    _, pet = make_owner()
    walk = Task("Walk", 20, start_date=DAY)
    pet.add_task(walk)
    pet.complete_task(walk, DAY)

    with pytest.raises(ValueError, match="already completed"):
        pet.complete_task(walk, DAY)
    assert pet.task_count == 2


def test_completed_task_drops_out_of_schedule_and_next_one_appears():
    owner, pet = make_owner()
    walk = Task("Walk", 20, start_date=DAY)
    pet.add_task(walk)
    next_walk = pet.complete_task(walk, DAY)

    assert owner.create_schedule(DAY).planned == []
    assert [task for _, task in owner.create_schedule(DAY + timedelta(days=1)).planned] == [next_walk]


# Conflict detection


def test_conflict_between_pets_is_detected_and_reported():
    owner, mochi = make_owner()
    biscuit = Pet("Biscuit", "cat")
    owner.add_pet(biscuit)
    walk = Task("Walk", 30, "high", preferred_time=time(9, 0), start_date=DAY)
    meds = Task("Meds", 10, "low", preferred_time=time(9, 0), start_date=DAY)  # same start time
    mochi.add_task(walk)
    biscuit.add_task(meds)

    schedule = owner.create_schedule(DAY)

    assert schedule.conflicts == {meds: [walk]}
    assert schedule.conflict_messages() == ["Biscuit: Meds at 09:00 overlaps Mochi: Walk - moved to 08:50"]  # 10 min early beats 30 min late
    assert "Conflicts detected:" in schedule.explain()
    assert not schedule.has_conflicts()  # detected and resolved, so the final plan has no overlap


def test_conflict_reported_as_skipped_when_task_cannot_move():
    owner, pet = make_owner(available_minutes=30)
    walk = Task("Walk", 30, "high", preferred_time=time(8, 0), start_date=DAY)
    meds = Task("Meds", 10, "low", preferred_time=time(8, 10), start_date=DAY)
    pet.add_task(walk)
    pet.add_task(meds)

    schedule = owner.create_schedule(DAY)

    assert schedule.conflict_messages() == ["Mochi: Meds at 08:10 overlaps Mochi: Walk - skipped"]


def test_back_to_back_tasks_are_not_a_conflict():
    owner, pet = make_owner()
    pet.add_task(Task("Walk", 30, "high", preferred_time=time(9, 0), start_date=DAY))
    pet.add_task(Task("Meds", 10, "low", preferred_time=time(9, 30), start_date=DAY))

    schedule = owner.create_schedule(DAY)

    assert schedule.conflicts == {}
    assert list(starts(schedule).values()) == ["09:00", "09:30"]


def test_overdue_task_is_tagged_and_wins_priority_ties():
    owner, pet = make_owner(available_minutes=30)
    wednesday = DAY + timedelta(days=2)
    pet.add_task(Task("Walk", 30, "medium", start_date=wednesday))
    bath = Task("Bath", 30, "medium", frequency="weekly", start_date=DAY)
    pet.add_task(bath)

    schedule = owner.create_schedule(wednesday)

    assert list(starts(schedule)) == ["Bath"]
    assert "overdue since Mon Oct 05" in schedule.reasons[bath]


# Sorting by time


def test_planned_tasks_are_sorted_by_start_time():
    owner, pet = make_owner()
    pet.add_task(Task("Late", 10, "high", preferred_time=time(9, 30), start_date=DAY))
    pet.add_task(Task("Early", 10, "low", preferred_time=time(8, 5), start_date=DAY))
    pet.add_task(Task("Flexible", 5, "medium", start_date=DAY))

    schedule = owner.create_schedule(DAY)

    assert list(starts(schedule).values()) == ["08:00", "08:05", "09:30"]


# Filtering


def test_filter_tasks_by_pet_and_completion():
    owner, mochi = make_owner()
    biscuit = Pet("Biscuit", "cat")
    owner.add_pet(biscuit)
    walk = Task("Walk", 20, start_date=DAY)
    feed = Task("Feed", 5, start_date=DAY)
    brush = Task("Brush", 10, start_date=DAY)
    mochi.add_task(walk)
    mochi.add_task(feed)
    biscuit.add_task(brush)
    walk.mark_complete(DAY)

    assert owner.filter_tasks(pet_name="Mochi", day=DAY) == [walk, feed]
    assert owner.filter_tasks(completed=True, day=DAY) == [walk]
    assert owner.filter_tasks(completed=False, day=DAY) == [feed, brush]
    assert owner.filter_tasks(pet_name="Mochi", completed=False, day=DAY) == [feed]
    assert owner.filter_tasks(completed=True, day=DAY - timedelta(days=1)) == []  # not done yet as of the day before


# Sorting by priority


def test_flexible_tasks_placed_highest_priority_first():
    owner, pet = make_owner()
    pet.add_task(Task("Brush", 10, "low", start_date=DAY))
    pet.add_task(Task("Walk", 20, "high", start_date=DAY))
    pet.add_task(Task("Play", 15, "medium", start_date=DAY))

    schedule = owner.create_schedule(DAY)

    assert starts(schedule) == {"Walk": "08:00", "Play": "08:20", "Brush": "08:35"}


def test_equal_priority_places_shorter_task_first():
    owner, pet = make_owner()
    pet.add_task(Task("Long walk", 40, "high", start_date=DAY))
    pet.add_task(Task("Meds", 5, "high", start_date=DAY))

    schedule = owner.create_schedule(DAY)

    assert starts(schedule) == {"Meds": "08:00", "Long walk": "08:05"}


def test_shortest_first_preference_ignores_priority_order():
    owner, pet = make_owner(prefer_high_priority_first=False)
    pet.add_task(Task("Walk", 30, "high", start_date=DAY))
    pet.add_task(Task("Brush", 10, "low", start_date=DAY))
    pet.add_task(Task("Feed", 10, "medium", start_date=DAY))

    schedule = owner.create_schedule(DAY)

    # Same duration: higher priority breaks the tie.
    assert starts(schedule) == {"Feed": "08:00", "Brush": "08:10", "Walk": "08:20"}


def test_high_priority_flexible_task_bumps_low_priority_fixed_task():
    owner, pet = make_owner(available_minutes=30)
    toy = Task("Toy", 20, "low", preferred_time=time(8, 5), start_date=DAY)
    meds = Task("Meds", 15, "high", start_date=DAY)
    pet.add_task(toy)
    pet.add_task(meds)

    schedule = owner.create_schedule(DAY)

    assert starts(schedule) == {"Meds": "08:00"}
    assert schedule.reasons[meds] == "high priority, made room by bumping Mochi: Toy"
    assert schedule.reasons[toy] == "bumped by higher-priority Mochi: Meds"
    assert schedule.conflict_messages() == ["Mochi: Toy at 08:05 overlaps Mochi: Meds - skipped"]


def test_bumped_task_is_moved_instead_of_skipped_when_it_still_fits():
    owner, pet = make_owner(available_minutes=60)
    pet.add_task(Task("Toy", 20, "low", preferred_time=time(8, 20), start_date=DAY))
    pet.add_task(Task("Walk", 40, "high", start_date=DAY))  # needs 08:00-08:40, which Toy blocks

    schedule = owner.create_schedule(DAY)

    assert starts(schedule) == {"Walk": "08:00", "Toy": "08:40"}
    assert not schedule.has_conflicts()


def test_bumping_never_displaces_equal_priority_tasks():
    owner, pet = make_owner(available_minutes=30)
    pet.add_task(Task("Toy", 20, "high", preferred_time=time(8, 5), start_date=DAY))
    meds = Task("Meds", 15, "high", start_date=DAY)
    pet.add_task(meds)

    schedule = owner.create_schedule(DAY)

    assert list(starts(schedule)) == ["Toy"]
    assert schedule.reasons[meds] == "needs 15 min, largest free slot is 5 min"


def test_bumping_can_be_turned_off():
    owner, pet = make_owner(available_minutes=30, bump_lower_priority=False)
    pet.add_task(Task("Toy", 20, "low", preferred_time=time(8, 5), start_date=DAY))
    meds = Task("Meds", 15, "high", start_date=DAY)
    pet.add_task(meds)

    schedule = owner.create_schedule(DAY)

    assert list(starts(schedule)) == ["Toy"]
    assert schedule.reasons[meds] == "needs 15 min, largest free slot is 5 min"


def test_critical_task_outranks_high_and_bumps_it():
    owner, pet = make_owner(available_minutes=20)
    pet.add_task(Task("Walk", 20, "high", preferred_time=time(8, 0), start_date=DAY))
    meds = Task("Meds", 5, "critical", start_date=DAY)
    pet.add_task(meds)

    schedule = owner.create_schedule(DAY)

    assert starts(schedule)["Meds"] == "08:00"
    assert "Walk" not in starts(schedule)


# Weighted priority: overdue tasks gain priority each day


def test_overdue_low_task_rises_above_medium_after_enough_days():
    owner, pet = make_owner(available_minutes=20)
    friday = DAY + timedelta(days=4)
    brush = Task("Brush", 20, "low", start_date=DAY)  # 4 days overdue: 1 + 4 * 0.5 = 3 (high)
    pet.add_task(brush)
    pet.add_task(Task("Play", 20, "medium", start_date=friday))

    schedule = owner.create_schedule(friday)

    assert list(starts(schedule)) == ["Brush"]
    assert schedule.reasons[brush].startswith("low priority, raised to high after 4 days overdue")


def test_overdue_boost_is_capped_below_critical():
    owner, _ = make_owner()
    old = Task("Brush", 20, "low", start_date=DAY - timedelta(days=30))
    schedule = owner.create_schedule(DAY)

    assert schedule.effective_priority(old) == Priority.HIGH
    assert schedule.effective_priority(Task("Meds", 5, "critical", start_date=DAY)) == Priority.CRITICAL


def test_overdue_boost_can_be_turned_off():
    owner, pet = make_owner(available_minutes=20, overdue_boost_per_day=0)
    friday = DAY + timedelta(days=4)
    pet.add_task(Task("Brush", 20, "low", start_date=DAY))
    pet.add_task(Task("Play", 20, "medium", start_date=friday))

    assert list(starts(owner.create_schedule(friday))) == ["Play"]


# Next available slot


def test_next_available_slot_finds_first_gap_that_fits():
    owner, pet = make_owner()
    pet.add_task(Task("Feed", 10, "high", preferred_time=time(8, 0), start_date=DAY))
    pet.add_task(Task("Walk", 30, "high", preferred_time=time(8, 15), start_date=DAY))
    schedule = owner.create_schedule(DAY)

    assert schedule.next_available_slot(5) == time(8, 10)
    assert schedule.next_available_slot(20) == time(8, 45)
    assert schedule.next_available_slot(20, after=time(9, 0)) == time(9, 0)
    assert schedule.next_available_slot(200) is None


# Task ids


def test_each_occurrence_gets_its_own_id():
    _, pet = make_owner()
    walk = Task("Walk", 20, start_date=DAY)
    pet.add_task(walk)

    next_walk = pet.complete_task(walk, DAY)

    assert walk.id and next_walk.id and walk.id != next_walk.id


# Recurrence


def test_task_is_not_due_before_its_start_date():
    task = Task("Walk", 20, start_date=DAY)

    assert not task.is_due(DAY - timedelta(days=1))
    assert task.last_occurrence(DAY - timedelta(days=1)) is None


def test_overdue_daily_task_next_occurrence_is_day_after_completion():
    _, pet = make_owner()
    walk = Task("Walk", 20, start_date=DAY)
    pet.add_task(walk)

    next_walk = pet.complete_task(walk, DAY + timedelta(days=2))

    assert next_walk.start_date == DAY + timedelta(days=3)


def test_weekly_task_missed_for_weeks_creates_only_one_next_occurrence():
    _, pet = make_owner()
    bath = Task("Bath", 30, frequency="weekly", start_date=DAY)  # Monday Oct 05
    pet.add_task(bath)

    next_bath = pet.complete_task(bath, DAY + timedelta(days=15))  # Tuesday Oct 20

    assert next_bath.start_date == date(2026, 10, 26)  # the following Monday
    assert pet.task_count == 2
    assert pet.due_tasks(date(2026, 10, 21)) == []  # nothing piles up from the missed weeks


def test_completing_task_before_its_start_date_schedules_next_from_start_date():
    _, pet = make_owner()
    walk = Task("Walk", 20, start_date=DAY)
    pet.add_task(walk)

    next_walk = pet.complete_task(walk, DAY - timedelta(days=1))

    assert walk.is_completed(DAY)
    assert next_walk.start_date == DAY + timedelta(days=1)


def test_completing_task_from_another_pet_is_rejected():
    _, pet = make_owner()
    stranger = Task("Walk", 20, start_date=DAY)

    with pytest.raises(ValueError, match="isn't one of Mochi's tasks"):
        pet.complete_task(stranger, DAY)
    assert stranger.last_completed is None


def test_stored_schedule_is_not_updated_until_regenerated():
    owner, pet = make_owner()
    walk = Task("Walk", 20, start_date=DAY)
    pet.add_task(walk)
    first = owner.create_schedule(DAY)

    pet.complete_task(walk, DAY)

    assert owner.schedules[DAY] is first
    assert [task for _, task in first.planned] == [walk]  # stale until create_schedule runs again
    assert owner.create_schedule(DAY).planned == []
    assert owner.schedules[DAY] is not first


# Conflict detection and relocation


def test_relocation_tie_prefers_earlier_slot():
    owner, pet = make_owner()
    pet.add_task(Task("Walk", 30, "high", preferred_time=time(9, 0), start_date=DAY))
    meds = Task("Meds", 20, "low", preferred_time=time(9, 5), start_date=DAY)
    pet.add_task(meds)

    schedule = owner.create_schedule(DAY)

    # 08:40 and 09:30 are both 25 min from 09:05.
    assert starts(schedule)["Meds"] == "08:40"
    assert not schedule.has_conflicts()


def test_several_tasks_clashing_with_same_fixed_task_are_relocated_in_rank_order():
    owner, pet = make_owner()
    walk = Task("Walk", 30, "high", preferred_time=time(9, 0), start_date=DAY)
    feed = Task("Feed", 10, "medium", preferred_time=time(9, 10), start_date=DAY)
    meds = Task("Meds", 10, "low", preferred_time=time(9, 10), start_date=DAY)
    for task in (walk, feed, meds):
        pet.add_task(task)

    schedule = owner.create_schedule(DAY)

    assert starts(schedule) == {"Feed": "08:50", "Walk": "09:00", "Meds": "09:30"}
    assert schedule.conflicts == {feed: [walk], meds: [walk]}
    assert not schedule.has_conflicts()


def test_flexible_tasks_fill_gaps_around_fixed_tasks():
    owner, pet = make_owner()
    pet.add_task(Task("Walk", 30, "high", preferred_time=time(8, 10), start_date=DAY))
    pet.add_task(Task("Feed", 10, "medium", start_date=DAY))
    pet.add_task(Task("Brush", 15, "low", start_date=DAY))

    schedule = owner.create_schedule(DAY)

    assert starts(schedule) == {"Feed": "08:00", "Walk": "08:10", "Brush": "08:40"}
    assert schedule.conflicts == {}
    assert not schedule.has_conflicts()


# Window boundaries


def test_fixed_task_ending_exactly_at_day_end_is_planned():
    owner, pet = make_owner()
    pet.add_task(Task("Walk", 30, preferred_time=time(9, 30), start_date=DAY))

    assert starts(owner.create_schedule(DAY)) == {"Walk": "09:30"}


def test_fixed_task_before_day_start_is_skipped():
    owner, pet = make_owner()
    walk = Task("Walk", 30, preferred_time=time(7, 59), start_date=DAY)
    pet.add_task(walk)

    schedule = owner.create_schedule(DAY)

    assert schedule.skipped == [walk]
    assert schedule.reasons[walk] == "fixed at 07:59, outside available time 08:00-10:00 (starts before 08:00)"


def test_task_longer_than_window_is_skipped():
    owner, pet = make_owner()
    hike = Task("Hike", 200, start_date=DAY)
    pet.add_task(hike)

    schedule = owner.create_schedule(DAY)

    assert schedule.skipped == [hike]
    assert schedule.reasons[hike] == "needs 200 min, largest free slot is 120 min"


def test_zero_available_minutes_skips_everything():
    owner, pet = make_owner(available_minutes=0)
    pet.add_task(Task("Walk", 10, start_date=DAY))
    pet.add_task(Task("Meds", 5, preferred_time=time(8, 0), start_date=DAY))

    schedule = owner.create_schedule(DAY)

    assert schedule.planned == []
    assert len(schedule.skipped) == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"available_minutes": -1}, "can't be negative"),
        ({"day_start": time(23, 0), "available_minutes": 60}, "before midnight"),
    ],
)
def test_invalid_owner_window_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Owner("Jordan", **kwargs)


# Explanation output


def test_explain_lists_plan_and_total():
    owner, pet = make_owner()
    pet.add_task(Task("Walk", 30, "high", start_date=DAY))

    lines = owner.create_schedule(DAY).explain().splitlines()

    assert lines == [
        "Plan for Jordan on Monday, Oct 05 2026 (available 08:00-10:00, 120 min):",
        "  08:00-08:30  Mochi: Walk (30 min) [high] - high priority, placed in earliest free slot",
        "Total: 30 of 120 min used.",
    ]


def test_explain_with_no_due_tasks():
    owner, _ = make_owner()

    assert owner.create_schedule(DAY).explain().endswith("No tasks are due today.")


# Validation and ownership


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"duration_minutes": 0}, "duration_minutes must be positive"),
        ({"frequency": "Daily"}, "frequency must be one of"),
        ({"priority": "urgent"}, "priority must be one of low, medium, high, critical"),
    ],
)
def test_invalid_task_is_rejected(kwargs, message):
    fields = {"title": "Walk", "duration_minutes": 20, **kwargs}
    with pytest.raises(ValueError, match=message):
        Task(**fields)


def test_string_priority_is_case_insensitive():
    assert Task("Walk", 20, "HIGH").priority == Priority.HIGH


def test_adding_same_task_twice_to_a_pet_is_ignored():
    pet = Pet("Mochi", "dog")
    walk = Task("Walk", 20)

    pet.add_task(walk)
    pet.add_task(walk)

    assert pet.task_count == 1


def test_duplicate_pet_name_is_rejected_but_same_pet_is_ignored():
    owner, mochi = make_owner()

    owner.add_pet(mochi)
    with pytest.raises(ValueError, match="already has a pet named 'Mochi'"):
        owner.add_pet(Pet("Mochi", "cat"))
    assert owner.pets == [mochi]


# Moving tasks between pets


def test_task_moved_to_another_pet_is_scheduled_once():
    owner, mochi = make_owner()
    biscuit = Pet("Biscuit", "cat")
    owner.add_pet(biscuit)
    walk = Task("Walk", 20, start_date=DAY)
    mochi.add_task(walk)

    biscuit.add_task(walk)

    assert mochi.tasks == []
    assert biscuit.tasks == [walk]
    assert (walk.pet, walk.pet_name) == (biscuit, "Biscuit")
    assert [task for _, task in owner.create_schedule(DAY).planned] == [walk]


def test_removed_task_can_be_added_to_another_pet():
    _, mochi = make_owner()
    biscuit = Pet("Biscuit", "cat")
    walk = Task("Walk", 20, start_date=DAY)
    mochi.add_task(walk)

    mochi.remove_task(walk)
    assert walk.pet is None
    biscuit.add_task(walk)

    assert (mochi.tasks, biscuit.tasks) == ([], [walk])


def test_next_occurrence_belongs_to_same_pet():
    _, pet = make_owner()
    walk = Task("Walk", 20, start_date=DAY)
    pet.add_task(walk)

    next_walk = pet.complete_task(walk, DAY)

    assert next_walk.pet is pet
    assert walk.pet is pet  # the completed occurrence stays in the pet's history


# Integer priorities


def test_integer_priority_is_converted():
    owner, pet = make_owner()
    pet.add_task(Task("Walk", 20, 3, start_date=DAY))

    assert "[high]" in owner.create_schedule(DAY).explain()


def test_out_of_range_integer_priority_is_rejected():
    with pytest.raises(ValueError, match="priority must be one of"):
        Task("Walk", 20, 5)


# Changing the owner's window


def test_changing_window_past_midnight_is_rejected():
    owner = Owner("Jordan", day_start=time(22, 0), available_minutes=60)

    with pytest.raises(ValueError, match="before midnight"):
        owner.available_minutes = 200
    with pytest.raises(ValueError, match="before midnight"):
        owner.day_start = time(23, 30)
    assert (owner.day_start, owner.available_minutes) == (time(22, 0), 60)  # rejected values aren't applied


def test_set_window_accepts_pair_that_is_only_valid_together():
    owner = Owner("Jordan", day_start=time(8, 0), available_minutes=600)

    owner.set_window(time(20, 0), 60)  # 20:00 + 600 min would pass midnight if day_start were set first

    assert (owner.day_start, owner.available_minutes, owner.day_end) == (time(20, 0), 60, time(21, 0))


def test_set_window_rejects_invalid_pair_without_changing_anything():
    owner = Owner("Jordan", day_start=time(8, 0), available_minutes=120)

    with pytest.raises(ValueError, match="before midnight"):
        owner.set_window(time(23, 0), 120)

    assert (owner.day_start, owner.available_minutes) == (time(8, 0), 120)


# Time windows


def test_flexible_task_waits_for_its_not_before_time():
    owner, pet = make_owner()
    walk = Task("Walk", 30, not_before=time(9, 0), start_date=DAY)
    pet.add_task(walk)

    schedule = owner.create_schedule(DAY)

    assert starts(schedule) == {"Walk": "09:00"}
    assert schedule.reasons[walk].endswith("within its window (not before 09:00)")


def test_done_by_task_is_placed_before_its_deadline_even_when_low_priority():
    owner, pet = make_owner()
    pet.add_task(Task("Play", 60, "high", start_date=DAY))
    pet.add_task(Task("Meds", 10, "low", done_by=time(8, 30), start_date=DAY))

    schedule = owner.create_schedule(DAY)

    # Play takes 08:00-09:00 first; Meds can't fit before 08:30 and Play outranks it, so it's skipped with a clear reason.
    assert "Meds" not in starts(schedule)
    assert "within 08:00-08:30" in schedule.reasons[schedule.skipped[0]]


def test_done_by_task_bumps_lower_priority_task_out_of_its_window():
    owner, pet = make_owner()
    pet.add_task(Task("Play", 60, "low", start_date=DAY))
    pet.add_task(Task("Meds", 10, "high", done_by=time(8, 30), start_date=DAY))

    schedule = owner.create_schedule(DAY)

    assert starts(schedule)["Meds"] == "08:00"
    assert starts(schedule)["Play"] == "08:10"


@pytest.mark.parametrize("kwargs, message", [
    ({"not_before": time(9, 0), "done_by": time(9, 10)}, "shorter than the task"),
    ({"preferred_time": time(8, 0), "not_before": time(9, 0)}, "before its not-before time"),
    ({"preferred_time": time(8, 50), "done_by": time(9, 0)}, "ends after its done-by time"),
    ({"after": "walk"}, "can't come after itself"),
    ({"after": "Feed", "after_gap_minutes": -5}, "can't be negative"),
])
def test_invalid_timing_rules_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Task("Walk", 20, **kwargs)


# Ordering rules


def test_task_is_placed_after_its_predecessor_with_gap():
    owner, pet = make_owner()
    pet.add_task(Task("Meds", 5, "high", after="Breakfast", after_gap_minutes=15, start_date=DAY))
    pet.add_task(Task("Breakfast", 10, "low", start_date=DAY))

    schedule = owner.create_schedule(DAY)

    assert starts(schedule) == {"Breakfast": "08:00", "Meds": "08:25"}
    meds = next(t for _, t in schedule.planned if t.title == "Meds")
    assert "after Mochi: Breakfast +15 min" in schedule.reasons[meds]


def test_fixed_task_that_breaks_its_order_is_moved():
    owner, pet = make_owner()
    pet.add_task(Task("Breakfast", 10, "high", preferred_time=time(9, 0), start_date=DAY))
    meds = Task("Meds", 5, "high", preferred_time=time(8, 30), after="Breakfast", start_date=DAY)
    pet.add_task(meds)

    schedule = owner.create_schedule(DAY)

    assert starts(schedule) == {"Breakfast": "09:00", "Meds": "09:10"}
    assert "to avoid the order with Mochi: Breakfast" in schedule.reasons[meds]
    assert meds not in schedule.conflicts  # not an overlap, so not reported as a time conflict


def test_predecessor_must_finish_before_a_fixed_dependent():
    owner, pet = make_owner()
    pet.add_task(Task("Walk", 30, "high", preferred_time=time(8, 30), after="Harness on", start_date=DAY))
    pet.add_task(Task("Harness on", 5, "low", start_date=DAY))
    pet.add_task(Task("Brush", 30, "medium", start_date=DAY))

    schedule = owner.create_schedule(DAY)

    assert starts(schedule)["Harness on"] < "08:30"
    assert starts(schedule)["Walk"] == "08:30"


def test_ordering_rule_only_applies_within_the_same_pet():
    owner, mochi = make_owner()
    biscuit = Pet("Biscuit", "cat")
    owner.add_pet(biscuit)
    biscuit.add_task(Task("Breakfast", 10, "high", preferred_time=time(9, 0), start_date=DAY))
    mochi.add_task(Task("Meds", 5, "high", after="Breakfast", start_date=DAY))

    assert starts(owner.create_schedule(DAY))["Meds"] == "08:00"


def test_rule_with_missing_predecessor_is_ignored():
    owner, pet = make_owner()
    pet.add_task(Task("Meds", 5, after="Breakfast", start_date=DAY))
    assert starts(owner.create_schedule(DAY)) == {"Meds": "08:00"}


# Best-fit packing


def make_fragmented_owner(**kwargs) -> tuple[Owner, Pet]:
    """Free gaps of 25 and 30 minutes around fixed tasks; flexible tasks of 20, 25, and 10 minutes."""
    owner, pet = make_owner(available_minutes=60, **kwargs)
    pet.add_task(Task("Feed", 5, "critical", preferred_time=time(8, 25), start_date=DAY))
    pet.add_task(Task("Short walk", 20, "high", start_date=DAY))
    pet.add_task(Task("Long walk", 25, "high", start_date=DAY))
    pet.add_task(Task("Brush", 10, "medium", start_date=DAY))
    return owner, pet


def test_first_fit_wastes_space_and_skips_a_task():
    owner, _ = make_fragmented_owner(best_fit_packing=False)
    schedule = owner.create_schedule(DAY)
    assert [t.title for t in schedule.skipped] == ["Brush"]


def test_best_fit_packing_fits_every_task():
    owner, _ = make_fragmented_owner()

    schedule = owner.create_schedule(DAY)

    assert schedule.skipped == []
    assert starts(schedule) == {"Long walk": "08:00", "Feed": "08:25", "Short walk": "08:30", "Brush": "08:50"}
    long_walk = next(t for _, t in schedule.planned if t.title == "Long walk")
    assert "so more important tasks fit" in schedule.reasons[long_walk]
    assert not schedule.has_conflicts()


def test_best_fit_never_trades_a_higher_priority_task_for_lower_ones():
    owner, pet = make_owner(available_minutes=30)
    pet.add_task(Task("Walk", 30, "high", start_date=DAY))
    pet.add_task(Task("Brush", 15, "low", start_date=DAY))
    pet.add_task(Task("Play", 15, "low", start_date=DAY))

    assert list(starts(owner.create_schedule(DAY))) == ["Walk"]


def test_best_fit_in_shortest_first_mode_fits_the_most_tasks():
    owner, pet = make_owner(available_minutes=30, prefer_high_priority_first=False)
    pet.add_task(Task("Walk", 30, "high", start_date=DAY))
    pet.add_task(Task("Brush", 15, "low", start_date=DAY))
    pet.add_task(Task("Play", 15, "low", start_date=DAY))

    assert set(starts(owner.create_schedule(DAY))) == {"Brush", "Play"}


# Editing tasks


def test_update_validates_before_changing_anything():
    walk = Task("Walk", 30, "low")

    with pytest.raises(ValueError):
        walk.update(priority="high", duration_minutes=0)
    assert (walk.priority, walk.duration_minutes) == (Priority.LOW, 30)

    walk.update(priority="high", not_before=time(9, 0))
    assert (walk.priority, walk.not_before) == (Priority.HIGH, time(9, 0))


def test_skip_reasons_say_why_a_task_cannot_fit_the_day():
    owner = Owner("Jordan", available_minutes=45, day_start=time(7, 0))
    mochi = Pet("Mochi", "dog")
    owner.add_pet(mochi)
    walk = Task("Walk", 30, "high", preferred_time=time(7, 20), start_date=date(2026, 10, 5))
    evening = Task("Evening walk", 30, "medium", not_before=time(17, 0), start_date=date(2026, 10, 5))
    early = Task("Early meds", 5, "high", done_by=time(6, 30), start_date=date(2026, 10, 5))
    for task in (walk, evening, early):
        mochi.add_task(task)

    reasons = owner.create_schedule(date(2026, 10, 5)).reasons

    assert reasons[walk] == "fixed at 07:20, outside available time 07:00-07:45 (would end at 07:50)"
    assert reasons[evening] == "can't start before 17:00, but the available time ends at 07:45"
    assert reasons[early] == "must be done by 06:30, but the available time starts at 07:00"
