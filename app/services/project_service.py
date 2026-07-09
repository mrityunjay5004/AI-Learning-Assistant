"""
Business logic for POST /project.

Accepts either a roadmap_id (in which case we pull goal/skills/context from
the stored roadmap) or a standalone goal_title + skills payload.
"""
import logging
import time

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.exceptions import DatabaseError, RoadmapNotFoundError
from app.models.db_models import Roadmap
from app.models.schemas import LLMProjectOutput, ProjectRequest, ProjectResponse
from app.prompts import PROJECT_SYSTEM_INSTRUCTION, build_project_prompt
from app.services.llm_client import generate_structured

logger = logging.getLogger(__name__)


def recommend_project(db: Session, req: ProjectRequest) -> ProjectResponse:
    req.validate_one_source_present()

    roadmap_context = None

    if req.roadmap_id:
        try:
            roadmap = db.get(Roadmap, req.roadmap_id)
        except SQLAlchemyError as exc:
            logger.exception("Failed to look up roadmap_id=%s", req.roadmap_id)
            raise DatabaseError("Failed to look up the roadmap.") from exc

        if roadmap is None:
            raise RoadmapNotFoundError(f"No roadmap found with id '{req.roadmap_id}'.")
        goal_title = roadmap.goal_title
        skills = roadmap.skills
        task_titles = ", ".join(t["title"] for t in roadmap.tasks)
        roadmap_context = (
            f"Total estimated hours: {roadmap.estimated_hours}. Tasks: {task_titles}."
        )
    else:
        goal_title = req.goal_title
        skills = req.skills

    prompt = build_project_prompt(goal_title, skills, roadmap_context)

    llm_start = time.monotonic()
    llm_output: LLMProjectOutput = generate_structured(
        system_instruction=PROJECT_SYSTEM_INSTRUCTION,
        user_prompt=prompt,
        output_schema=LLMProjectOutput,
    )
    llm_latency_ms = (time.monotonic() - llm_start) * 1000
    logger.info(
        "Project LLM call for goal='%s' (roadmap_id=%s) completed in %.0fms",
        goal_title,
        req.roadmap_id,
        llm_latency_ms,
    )

    return ProjectResponse(**llm_output.model_dump())
