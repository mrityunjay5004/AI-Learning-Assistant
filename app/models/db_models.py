"""
SQLAlchemy ORM models.

Roadmap: stores the generated roadmap (as JSON) plus the original request
inputs, so /project and /chat can look it up by roadmap_id later.

ChatMessage: stores conversation turns per roadmap, giving the RAG chat
endpoint real multi-turn memory (this is the "conversation history"
product-thinking feature described in the README).
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Roadmap(Base):
    __tablename__ = "roadmaps"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    goal_title: Mapped[str] = mapped_column(String, nullable=False)
    experience: Mapped[str] = mapped_column(String, nullable=False)
    known_skills: Mapped[list] = mapped_column(JSON, default=list)
    learning_style: Mapped[str] = mapped_column(String, nullable=False)
    weekly_hours: Mapped[int] = mapped_column(Integer, nullable=False)

    estimated_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    skills: Mapped[list] = mapped_column(JSON, default=list)
    tasks: Mapped[list] = mapped_column(JSON, default=list)  # list of task dicts

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    chat_messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="roadmap", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    roadmap_id: Mapped[str] = mapped_column(ForeignKey("roadmaps.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    roadmap: Mapped["Roadmap"] = relationship(back_populates="chat_messages")
