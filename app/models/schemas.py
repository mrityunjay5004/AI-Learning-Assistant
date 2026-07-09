"""
Pydantic models for all request/response payloads.

These serve two purposes:
1. FastAPI request validation (query/body shape, types, constraints).
2. A strict schema that the LLM's raw JSON output is parsed into, so a
   malformed or partially-hallucinated response is caught immediately
   rather than silently propagated to the client.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------
class LearningStyle(str, Enum):
    project_based = "Project Based"
    theory_based = "Theory Based"
    video_based = "Video Based"
    mixed = "Mixed"


class Difficulty(str, Enum):
    beginner = "Beginner"
    intermediate = "Intermediate"
    advanced = "Advanced"


# ---------------------------------------------------------------------------
# /roadmap
# ---------------------------------------------------------------------------
class RoadmapRequest(BaseModel):
    goal_title: str = Field(..., min_length=2, max_length=120, examples=["Backend Developer"])
    experience: str = Field(..., min_length=2, max_length=60, examples=["Less than 1 year"])
    known_skills: List[str] = Field(default_factory=list, max_length=50)
    learning_style: LearningStyle = LearningStyle.mixed
    weekly_hours: int = Field(..., gt=0, le=80, description="Hours per week the user can commit")

    @field_validator("known_skills")
    @classmethod
    def strip_and_dedupe_skills(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip() for s in v if s.strip()]
        # de-dupe while preserving order
        seen = set()
        result = []
        for skill in cleaned:
            key = skill.lower()
            if key not in seen:
                seen.add(key)
                result.append(skill)
        return result


class Subtask(BaseModel):
    title: str


class ResourceLink(BaseModel):
    """A curated (never LLM-generated) learning resource for a task, to
    avoid the URL-hallucination risk of asking the model for links directly.
    See app/resources.py for the lookup table."""

    label: str
    url: str


class Task(BaseModel):
    title: str
    estimated_hours: int = Field(..., ge=0)
    subtasks: List[Subtask] = Field(default_factory=list)
    resources: List[ResourceLink] = Field(
        default_factory=list,
        description="Curated documentation/learning links for this task (deterministically matched, not LLM-generated).",
    )


class RoadmapResponse(BaseModel):
    roadmap_id: str = Field(default_factory=lambda: str(uuid4()))
    goal_title: str
    estimated_hours: int
    skills: List[str]
    tasks: List[Task]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# /project
# ---------------------------------------------------------------------------
class ProjectRequest(BaseModel):
    roadmap_id: Optional[str] = None
    goal_title: Optional[str] = Field(None, min_length=2, max_length=120)
    skills: Optional[List[str]] = Field(None, max_length=50)

    @field_validator("skills")
    @classmethod
    def clean_skills(cls, v):
        if v is None:
            return v
        return [s.strip() for s in v if s.strip()]

    def validate_one_source_present(self) -> None:
        """Either roadmap_id, or (goal_title + skills), must be provided."""
        if self.roadmap_id:
            return
        if self.goal_title and self.skills:
            return
        raise ValueError(
            "Provide either 'roadmap_id' OR both 'goal_title' and 'skills'."
        )


class ProjectResponse(BaseModel):
    title: str
    difficulty: Difficulty
    estimated_hours: int
    tech_stack: List[str]
    features: List[str]
    why_this_project: str


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    roadmap_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    response: str
    follow_up_questions: List[str] = Field(default_factory=list)
    retrieved_context: Optional[List[str]] = Field(
        default=None,
        description="Roadmap chunks retrieved and used as grounding context (debug/transparency).",
    )


# ---------------------------------------------------------------------------
# Internal: shape we require directly from the LLM (used for validation)
# ---------------------------------------------------------------------------
class LLMRoadmapOutput(BaseModel):
    estimated_hours: int = Field(..., ge=1)
    skills: List[str] = Field(..., min_length=1)
    tasks: List[Task] = Field(..., min_length=1)


class LLMProjectOutput(BaseModel):
    title: str
    difficulty: Difficulty
    estimated_hours: int = Field(..., ge=1)
    tech_stack: List[str] = Field(..., min_length=1)
    features: List[str] = Field(..., min_length=1)
    why_this_project: str


class LLMChatOutput(BaseModel):
    response: str
    follow_up_questions: List[str] = Field(default_factory=list, max_length=5)
