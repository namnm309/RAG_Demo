from fastapi import APIRouter

from api.v1 import chat, health, interview_plans, interview_questions, knowledge

router = APIRouter()
router.include_router(health.router)
router.include_router(knowledge.router)
router.include_router(chat.router)
router.include_router(interview_plans.router)
router.include_router(interview_questions.router)
