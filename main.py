"""
SHL Assessment Recommender - FastAPI Service
Exposes /health and /chat endpoints for the SHL evaluator.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models import ChatRequest, ChatResponse
from agent import SHLAgent
from retriever import HybridRetriever

# Load environment variables
load_dotenv()

# Global references
agent: SHLAgent = None
retriever: HybridRetriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global agent, retriever
    
    print("=" * 60)
    print("SHL Assessment Recommender - Starting up...")
    print("=" * 60)
    
    # Initialize retriever (loads catalog, builds indexes)
    catalog_path = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
    retriever = HybridRetriever(catalog_path)
    
    # Initialize agent (auto-selects provider: Gemini or Groq via env vars)
    agent = SHLAgent(retriever=retriever)
    
    print("=" * 60)
    print("SHL Assessment Recommender - Ready!")
    print(f"Catalog loaded: {len(retriever.catalog)} assessments")
    print("=" * 60)
    
    yield
    
    # Cleanup
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment recommendation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint. Returns {"status": "ok"} with HTTP 200."""
    if agent is None or not agent.ready:
        raise HTTPException(status_code=503, detail="LLM provider not configured")
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint. Takes stateless conversation history, returns agent reply.
    
    Request body:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }
    
    Response:
    {
        "reply": "Agent's text response",
        "recommendations": null | [{name, url, test_type}, ...],
        "end_of_conversation": false
    }
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
    
    try:
        response = agent.chat(request.messages)
        if response.recommendations is None:
            response.recommendations = []
        return response
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        # Return a safe fallback rather than 500
        return ChatResponse(
            reply="I apologize for the inconvenience. Could you please rephrase your question about SHL assessments?",
            recommendations=[],
            end_of_conversation=False,
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
