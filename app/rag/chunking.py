"""
Chunking strategy for turning a generated roadmap into retrievable units.

Design decision: rather than splitting the roadmap's JSON by raw character
count (which would cut tasks/subtasks in half and destroy meaning), we chunk
*semantically* along the roadmap's natural structure:

  - One chunk per top-level task, containing the goal it serves, its
    position in the sequence, a link back to the previous task (so a
    "what comes before X" question retrieves something coherent even from
    a single chunk), estimated hours, subtask titles, and curated resource
    labels.
  - One extra "overview" chunk containing the goal, overall skill list, and
    total estimated hours, so questions about the roadmap as a whole
    ("how long will this take?") retrieve something reasonable.

This keeps each chunk small (a handful of sentences), self-contained, and
topically coherent - which matters more for retrieval quality here than
hitting a fixed token count, since the source document (a roadmap) is
already short and structured rather than free-flowing prose.

Richer metadata (task_index, total_tasks, task_title) is stored alongside
each chunk so the retriever can filter, deduplicate, and label chunks with
a citeable source (e.g. "Task 2/5") without re-parsing the chunk text.
"""
from __future__ import annotations

from typing import TypedDict

from app.resources import resources_for_topic


class Chunk(TypedDict):
    id: str
    text: str
    metadata: dict


def chunk_roadmap(
    roadmap_id: str,
    goal_title: str,
    estimated_hours: int,
    skills: list[str],
    tasks: list[dict],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    total_tasks = len(tasks)

    overview_text = (
        f"Roadmap overview for goal '{goal_title}'. "
        f"Total estimated time: {estimated_hours} hours. "
        f"Skills to learn: {', '.join(skills)}. "
        f"This roadmap has {total_tasks} tasks in total, "
        f"sequenced from foundational to advanced."
    )
    chunks.append(
        {
            "id": f"{roadmap_id}::overview",
            "text": overview_text,
            "metadata": {
                "roadmap_id": roadmap_id,
                "type": "overview",
                "task_index": -1,
                "task_title": "Overview",
                "source_label": "Roadmap Overview",
            },
        }
    )

    for idx, task in enumerate(tasks):
        subtask_titles = [st["title"] for st in task.get("subtasks", [])]
        subtasks_str = "; ".join(subtask_titles) if subtask_titles else "No subtasks listed."

        previous_title = tasks[idx - 1]["title"] if idx > 0 else None
        previous_str = (
            f"Comes after task '{previous_title}'." if previous_title else "This is the first task."
        )

        resource_labels = [r.label for r in resources_for_topic(task["title"])]
        resources_str = ", ".join(resource_labels) if resource_labels else "None"

        task_text = (
            f"Goal: {goal_title}. "
            f"Task {idx + 1} of {total_tasks}: '{task['title']}' "
            f"(estimated {task.get('estimated_hours', 'unknown')} hours). "
            f"{previous_str} "
            f"Subtasks: {subtasks_str}. "
            f"Suggested resources: {resources_str}."
        )
        chunks.append(
            {
                "id": f"{roadmap_id}::task::{idx}",
                "text": task_text,
                "metadata": {
                    "roadmap_id": roadmap_id,
                    "type": "task",
                    "task_index": idx,
                    "total_tasks": total_tasks,
                    "task_title": task["title"],
                    "source_label": f"Task {idx + 1}/{total_tasks}: {task['title']}",
                },
            }
        )

    return chunks
