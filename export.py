"""Schedule exports: table rows (for display and CSV) and an iCalendar file for calendar apps."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pawpal_system import Schedule, add_minutes, fmt


def schedule_rows(schedule: Schedule) -> list[dict]:
    """One row per planned task, in time order."""
    return [
        {
            "Start": fmt(start),
            "End": fmt(add_minutes(start, task.duration_minutes)),
            "Pet": task.pet_name,
            "Task": task.title,
            "Minutes": task.duration_minutes,
            "Priority": task.priority.name.title(),
            "Why": schedule.reasons[task],
        }
        for start, task in schedule.planned
    ]


def _escape(text: str) -> str:
    """Escape text for an iCalendar property value (RFC 5545)."""
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def schedule_to_ics(schedule: Schedule) -> str:
    """The planned tasks as calendar events in local (floating) time, importable into Google Calendar, Outlook, etc."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//PawPal+//Daily plan//EN", "CALSCALE:GREGORIAN"]
    for start, task in schedule.planned:
        begin = datetime.combine(schedule.day, start)
        end = begin + timedelta(minutes=task.duration_minutes)
        lines += [
            "BEGIN:VEVENT",
            f"UID:{task.id}-{schedule.day:%Y%m%d}@pawpal",
            f"DTSTAMP:{stamp}",
            f"DTSTART:{begin:%Y%m%dT%H%M%S}",
            f"DTEND:{end:%Y%m%dT%H%M%S}",
            f"SUMMARY:{_escape(task.label)}",
            f"DESCRIPTION:{_escape(f'{task.priority.name.title()} priority. {schedule.reasons[task]}')}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"
