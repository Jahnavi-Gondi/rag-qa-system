from fastapi import APIRouter
from backend.app.answer import answer_query

router = APIRouter()

@router.get("/ask")
def ask(query: str):
    return answer_query(query)
