from fastapi import FastAPI
from app.api.chat import router as chat_router

# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Production Ready RAG System",
    version="1.0"
)

# Register routes
app.include_router(chat_router, prefix="/api")