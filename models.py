"""
Pydantic models for the SHL Assessment Recommender API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatMessage(BaseModel):
    """A single message in the conversation."""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message content")


class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    messages: list[ChatMessage] = Field(..., description="Full conversation history")


class Recommendation(BaseModel):
    """A single assessment recommendation."""
    name: str = Field(..., description="Assessment name from the SHL catalog")
    url: str = Field(..., description="Full URL to the assessment in the SHL catalog")
    test_type: str = Field(..., description="Test type code(s), e.g. 'K', 'P', 'A,S'")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    reply: str = Field(..., description="The agent's text reply")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="List of 1-10 recommended assessments, or empty list if still gathering context",
    )
    end_of_conversation: bool = Field(
        False, description="True only when the agent considers the task complete"
    )


class CatalogItem(BaseModel):
    """A single assessment from the SHL catalog."""
    name: str
    url: str
    remote_testing: bool = False
    adaptive_irt: bool = False
    test_type: list[str] = Field(default_factory=list)
    test_type_labels: list[str] = Field(default_factory=list)
    description: str = ""
    duration: str = ""
    languages_raw: str = ""
