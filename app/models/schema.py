from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    query: str
    session_id: str

class QueryResponse(BaseModel):
    response: str
    sources: List[str]   # ✅ NEW