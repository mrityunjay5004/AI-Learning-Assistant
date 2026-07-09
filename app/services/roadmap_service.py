"""
Business logic for POST /roadmap.

Flow: build prompt -> call LLM for structured JSON -> attach curated
resources deterministically (never LLM-generated, see app/resources.py) ->
persist to DB -> index the roadmap into the vector store so /chat can
retrieve from it later.
"""
import logging
import time
from datetime import datetime, timezone

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.exceptions import DatabaseError, RoadmapNotFoundError
from app.models.db_models import Roadmap
from app.models.schemas import (
    LLMRoadmapOutput,
    ResourceLink,
    RoadmapRequest,
    RoadmapResponse,
    Task,
)
from app.prompts import ROADMAP_SYSTEM_INSTRUCTION, build_roadmap_prompt
from app.rag.retriever import index_roadmap
from app.resources import resources_for_topic
from app.services.llm_client import generate_structured

logger = logging.getLogger(__name__)


def _attach_resources(tasks: list[Task]) -> list[Task]:
    """Attach curated, non-hallucinated resource links to each task by
    matching its title against app/resources.py's lookup table. Kept as a
    deterministic post-processing step rather than an LLM instruction, since
    asking a model to produce URLs risks plausible-but-dead/incorrect links."""
    for task in tasks:
        curated = resources_for_topic(task.title)
        task.resources = [ResourceLink(label=r.label, url=r.url) for r in curated]
    return tasks


def _roadmap_to_markdown(roadmap: Roadmap) -> str:
    """Render a stored roadmap as a readable Markdown document. Kept as
    simple string templating (no external Markdown library needed) since the
    structure is small and fixed."""
    lines = [
        f"# Learning Roadmap: {roadmap.goal_title}",
        "",
        f"- **Experience level:** {roadmap.experience}",
        f"- **Learning style:** {roadmap.learning_style}",
        f"- **Weekly hours:** {roadmap.weekly_hours}",
        f"- **Total estimated hours:** {roadmap.estimated_hours}",
        f"- **Generated:** {roadmap.created_at.isoformat() if roadmap.created_at else 'unknown'}",
        "",
        "## Skills covered",
        "",
    ]
    lines += [f"- {skill}" for skill in roadmap.skills]
    lines += ["", "## Tasks", ""]

    for idx, task in enumerate(roadmap.tasks, start=1):
        lines.append(f"### {idx}. {task['title']} ({task.get('estimated_hours', '?')}h)")
        lines.append("")
        for subtask in task.get("subtasks", []):
            lines.append(f"- [ ] {subtask['title']}")
        resources = task.get("resources", [])
        if resources:
            lines.append("")
            lines.append("**Resources:**")
            for r in resources:
                lines.append(f"- [{r['label']}]({r['url']})")
        lines.append("")

    return "\n".join(lines)


def get_roadmap_markdown(db: Session, roadmap_id: str) -> str:
    """Fetch a stored roadmap and render it as Markdown, for the
    GET /roadmap/{id}/markdown export endpoint."""
    roadmap = db.get(Roadmap, roadmap_id)
    if roadmap is None:
        raise RoadmapNotFoundError(f"No roadmap found with id '{roadmap_id}'.")
    return _roadmap_to_markdown(roadmap)


def generate_roadmap(db: Session, req: RoadmapRequest) -> RoadmapResponse:
    prompt = build_roadmap_prompt(
        goal_title=req.goal_title,
        experience=req.experience,
        known_skills=req.known_skills,
        learning_style=req.learning_style.value,
        weekly_hours=req.weekly_hours,
    )

    llm_start = time.monotonic()
    llm_output: LLMRoadmapOutput = generate_structured(
        system_instruction=ROADMAP_SYSTEM_INSTRUCTION,
        user_prompt=prompt,
        output_schema=LLMRoadmapOutput,
    )
    llm_latency_ms = (time.monotonic() - llm_start) * 1000
    logger.info(
        "Roadmap LLM call for goal='%s' completed in %.0fms (%d tasks)",
        req.goal_title,
        llm_latency_ms,
        len(llm_output.tasks),
    )

    enriched_tasks = _attach_resources(list(llm_output.tasks))

    try:
        roadmap = Roadmap(
            goal_title=req.goal_title,
            experience=req.experience,
            known_skills=req.known_skills,
            learning_style=req.learning_style.value,
            weekly_hours=req.weekly_hours,
            estimated_hours=llm_output.estimated_hours,
            skills=llm_output.skills,
            tasks=[t.model_dump() for t in enriched_tasks],
        )
        db.add(roadmap)
        db.commit()
        db.refresh(roadmap)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to persist roadmap for goal='%s'", req.goal_title)
        raise DatabaseError("Failed to save the generated roadmap.") from exc

    # Index into the vector store for RAG chat (best-effort: log but don't
    # fail roadmap creation if indexing has a transient issue - the roadmap
    # itself is still usable via /project, and /chat will simply retrieve no
    # context and fall back to general-knowledge answers until re-indexed).
    try:
        index_roadmap(
            roadmap_id=roadmap.id,
            goal_title=roadmap.goal_title,
            estimated_hours=roadmap.estimated_hours,
            skills=roadmap.skills,
            tasks=roadmap.tasks,
        )
    except Exception:
        logger.exception(
            "Failed to index roadmap_id=%s for RAG; chat will run without context until fixed",
            roadmap.id,
        )

    return RoadmapResponse(
        roadmap_id=roadmap.id,
        goal_title=roadmap.goal_title,
        estimated_hours=roadmap.estimated_hours,
        skills=roadmap.skills,
        tasks=[Task.model_validate(t) for t in roadmap.tasks],
        generated_at=roadmap.created_at.replace(tzinfo=timezone.utc)
        if roadmap.created_at
        else datetime.now(timezone.utc),
    )
