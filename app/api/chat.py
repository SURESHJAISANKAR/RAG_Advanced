from fastapi import APIRouter
from app.services.rag_service import ask_question
from app.models.schema import QueryRequest, QueryResponse

router = APIRouter()

@router.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    answer = ask_question(request.query, request.session_id)
    return QueryResponse(response=answer)
