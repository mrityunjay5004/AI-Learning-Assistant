import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.exceptions import InvalidRequestError
from app.models.schemas import ProjectRequest, ProjectResponse
from app.services.project_service import recommend_project

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Project"])


@router.post("/project", response_model=ProjectResponse, status_code=200)
def create_project(req: ProjectRequest, db: Session = Depends(get_db)) -> ProjectResponse:
    try:
        req.validate_one_source_present()
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc

    logger.info("Recommending project for roadmap_id=%s goal=%s", req.roadmap_id, req.goal_title)
    return recommend_project(db, req)
