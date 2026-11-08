"""Retrieval over two sources: each pet's owner-written guidelines and the owner's task history.

Uses BM25 keyword scoring, so it needs no embedding model or extra API calls.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

from guardrails import DATA_RULE, as_data
from llm import LLM
from pawpal_system import Owner, Task, fmt

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it",
    "of", "on", "or", "the", "to", "was", "with", "should", "must", "every", "each", "per",
}


def tokenize(text: str) -> list[str]:
    """Lowercase words minus stopwords, with a plural "s" stripped so "walks" matches "walk"."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w for w in words if w not in STOPWORDS]


@dataclass(frozen=True)
class Document:
    """One retrievable piece of text and where it came from."""

    text: str
    source: str  # "guideline" or "history"
    pet_name: str
    ref: str  # citation shown to the user, e.g. "Mochi guideline 2"


def describe_task(task: Task, day: date) -> str:
    """One-line summary of a task and its status on `day`, for prompts."""
    when = f"fixed at {fmt(task.preferred_time)}" if task.preferred_time else "flexible time"
    if task.timing_label:
        when += f" ({task.timing_label})"  # e.g. "(done by 09:00, after Breakfast +15 min)"
    if task.is_completed(day):
        status = "done"
    elif task.is_overdue(day):
        status = f"overdue since {task.start_date:%b %d}"
    elif task.is_due(day):
        status = "due"
    else:
        status = f"next due {task.start_date:%b %d}"
    return (
        f"{task.label}: {task.duration_minutes} min, {task.priority.name.lower()} priority, "
        f"{when}, repeats {task.frequency}, {status}"
    )


def build_documents(owner: Owner, day: date, history_days: int = 14) -> list[Document]:
    """Collect every pet's guidelines plus task history from the last `history_days` days up to `day`."""
    since = day - timedelta(days=history_days)
    docs = []
    for pet in owner.pets:
        for i, text in enumerate(pet.guidelines, 1):
            docs.append(Document(text, "guideline", pet.name, f"{pet.name} guideline {i}"))
        for task in pet.tasks:
            if task.last_completed and since <= task.last_completed <= day:
                docs.append(Document(
                    f"{task.label} ({task.duration_minutes} min) was completed on {task.last_completed:%a %b %d}.",
                    "history", pet.name, f"history {task.last_completed:%b %d}",
                ))
            elif task.is_overdue(day):
                docs.append(Document(
                    f"{task.label} has been overdue since {task.start_date:%a %b %d}.",
                    "history", pet.name, f"history {day:%b %d}",
                ))
    for schedule_day, schedule in owner.schedules.items():
        if since <= schedule_day <= day:
            for task in schedule.skipped:
                docs.append(Document(
                    f"On {schedule_day:%a %b %d}, {task.label} was skipped: {schedule.reasons[task]}.",
                    "history", task.pet_name, f"schedule {schedule_day:%b %d}",
                ))
    return docs


class Retriever:
    """BM25 search over a fixed list of documents."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1, self.b = k1, b
        self._tokens = [Counter(tokenize(doc.text)) for doc in documents]
        self._lengths = [sum(tokens.values()) for tokens in self._tokens]
        self._avg_length = sum(self._lengths) / len(documents) if documents else 0
        doc_freq = Counter(term for tokens in self._tokens for term in tokens)
        n = len(documents)
        self._idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in doc_freq.items()}

    def search(self, query: str, pet_name: str | None = None, source: str | None = None, k: int = 4) -> list[Document]:
        """Return up to `k` documents matching `query`, best first. Documents with no matching words are left out."""
        terms = tokenize(query)
        scored = []
        for doc, tokens, length in zip(self.documents, self._tokens, self._lengths):
            if pet_name and doc.pet_name.lower() != pet_name.lower():
                continue
            if source and doc.source != source:
                continue
            score = sum(
                self._idf[t] * tokens[t] * (self.k1 + 1)
                / (tokens[t] + self.k1 * (1 - self.b + self.b * length / self._avg_length))
                for t in terms
                if t in tokens
            )
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in scored[:k]]


def format_hits(docs: list[Document]) -> str:
    """Format retrieved documents as cited bullet lines for a prompt."""
    return "\n".join(f"- [{doc.ref}] {doc.text}" for doc in docs) or "- (nothing relevant found)"


@dataclass
class GuidelineIssue:
    """A task (or missing task) that breaks one of the owner's guidelines."""

    pet_name: str
    guideline: str
    issue: str
    suggestion: str
    ref: str


CHECK_SYSTEM = """You audit pets' care tasks against the owner's own numbered guidelines for each pet.
Only report a problem when a pet's current tasks or history clearly break one of that pet's guidelines.
A guideline also counts as broken when it calls for care the pet has no task for (e.g. "brush coat weekly" with no brushing task).
A task's time window counts: "flexible time (done by 09:00)" always finishes by 09:00.
Do not invent guidelines or facts.
Reply with JSON: {"issues": [{"pet": "<pet name>", "guideline": <guideline number>, "issue": "<what is wrong>", "suggestion": "<one concrete fix>"}]}
Use {"issues": []} if the tasks follow every guideline.
""" + DATA_RULE


def check_guidelines(llm: LLM, owner: Owner, retriever: Retriever, day: date) -> list[GuidelineIssue]:
    """Check every pet's tasks against its guidelines in one request. Issues citing a guideline that doesn't exist are dropped."""
    pets = {pet.name.lower(): pet for pet in owner.pets if pet.guidelines}
    if not pets:
        return []
    sections = [f"Today: {day:%A %b %d %Y}"]
    for pet in pets.values():
        guidelines = "\n".join(f"{i}. {g}" for i, g in enumerate(pet.guidelines, 1))
        tasks = "\n".join(f"- {describe_task(t, day)}" for t in pet.tasks if not t.is_completed(day)) or "- (no tasks)"
        history = retriever.search(" ".join(pet.guidelines), pet_name=pet.name, source="history", k=6)
        sections.append(
            f"## {pet.name} ({pet.species}, {pet.age} years). Notes: {pet.notes or 'none'}\n"
            f"Guidelines:\n{guidelines}\nCurrent tasks:\n{tasks}\nRecent history:\n{format_hits(history)}"
        )
    reply = llm.generate_json(CHECK_SYSTEM, as_data("\n\n".join(sections)))
    issues = []
    for item in reply.get("issues", []) if isinstance(reply, dict) else []:
        if not isinstance(item, dict):
            continue
        pet = pets.get(str(item.get("pet") or "").strip().lower())
        number, issue = item.get("guideline"), str(item.get("issue") or "").strip()
        if pet is None or not isinstance(number, int) or not 1 <= number <= len(pet.guidelines) or not issue:
            continue  # can't be traced to a real pet's guideline, so don't show it
        issues.append(GuidelineIssue(
            pet.name, pet.guidelines[number - 1], issue,
            str(item.get("suggestion") or "").strip(), f"{pet.name} guideline {number}",
        ))
    return issues
