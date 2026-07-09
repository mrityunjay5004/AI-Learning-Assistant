import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import handle_chat

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse, status_code=200)
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    logger.info("Chat message for roadmap_id=%s", req.roadmap_id)
    return handle_chat(db, req)
