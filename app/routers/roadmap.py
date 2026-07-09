import logging

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.schemas import RoadmapRequest, RoadmapResponse
from app.services.roadmap_service import generate_roadmap, get_roadmap_markdown

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Roadmap"])


@router.post("/roadmap", response_model=RoadmapResponse, status_code=201)
def create_roadmap(req: RoadmapRequest, db: Session = Depends(get_db)) -> RoadmapResponse:
    logger.info("Generating roadmap for goal='%s'", req.goal_title)
    return generate_roadmap(db, req)


@router.get(
    "/roadmap/{roadmap_id}/markdown",
    response_class=PlainTextResponse,
    responses={200: {"content": {"text/markdown": {}}}},
)
def export_roadmap_markdown(roadmap_id: str, db: Session = Depends(get_db)) -> PlainTextResponse:
    """Export a previously generated roadmap as a Markdown document -
    useful for saving to a notes app or committing alongside a repo."""
    logger.info("Exporting roadmap_id=%s as markdown", roadmap_id)
    markdown = get_roadmap_markdown(db, roadmap_id)
    return PlainTextResponse(content=markdown, media_type="text/markdown")
