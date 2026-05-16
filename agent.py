"""
SHL Assessment Recommender Agent.
Single LLM call architecture with hybrid retrieval and post-generation validation.
Provider order: Groq (primary) -> OpenRouter (fallback) -> Gemini (final fallback).
"""

import json
import os
import time
import requests
from typing import Optional

from pydantic import BaseModel, Field

from models import ChatMessage, ChatResponse, Recommendation
from prompts import (
    SYSTEM_PROMPT,
    build_turn_cap_instruction,
    format_catalog_context,
    format_conversation_history,
)
from retriever import HybridRetriever


# Schema for structured output
class LLMRecommendation(BaseModel):
    """A single assessment recommendation from the LLM."""
    name: str = Field(description="Exact assessment name from the SHL catalog")
    url: str = Field(description="Exact URL from the SHL catalog")
    test_type: str = Field(description="Test type code(s), e.g. 'K' or 'A,S'")


class LLMResponse(BaseModel):
    """Structured response from the LLM."""
    reply: str = Field(description="The agent's conversational text reply")
    recommendations: Optional[list[LLMRecommendation]] = Field(
        None, description="List of 1-10 recommended assessments, or null if still gathering context"
    )
    end_of_conversation: bool = Field(
        False, description="True only when the user explicitly confirms the final shortlist"
    )


class SHLAgent:
    """Conversational agent for SHL assessment recommendation."""

    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 2  # seconds

    def __init__(self, retriever: HybridRetriever, api_key: str = None):
        self.retriever = retriever
        self.provider = None  # Will be set during initialization
        self.gemini_client = None
        self.groq_client = None
        self.gemini_types = None
        self.model_name = None
        self.ready = False
        self.openrouter_api_key = None

        # Determine which LLM provider to use
        gemini_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        groq_key = os.environ.get("GROQ_API_KEY", "")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        preferred_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

        # Initialize Gemini if available
        if gemini_key:
            try:
                from google import genai
                from google.genai import types
                self.gemini_client = genai.Client(api_key=gemini_key)
                self.gemini_types = types
            except ImportError:
                print("WARNING: google-genai not installed.")

        # Initialize Groq if available
        if groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
            except ImportError:
                print("WARNING: groq not installed.")

        # Initialize OpenRouter if available
        if openrouter_key:
            self.openrouter_api_key = openrouter_key

        # Select provider based on preference and availability
        if preferred_provider == "groq" and self.groq_client:
            self.provider = "groq"
            self.model_name = "llama-3.3-70b-versatile"
        elif preferred_provider == "openrouter" and self.openrouter_api_key:
            self.provider = "openrouter"
            self.model_name = "deepseek/deepseek-chat"
        elif preferred_provider == "gemini" and self.gemini_client:
            self.provider = "gemini"
            self.model_name = "gemini-2.5-flash"
        elif self.groq_client:
            self.provider = "groq"
            self.model_name = "llama-3.3-70b-versatile"
        elif self.openrouter_api_key:
            self.provider = "openrouter"
            self.model_name = "deepseek/deepseek-chat"
        elif self.gemini_client:
            self.provider = "gemini"
            self.model_name = "gemini-2.5-flash"

        if self.provider == "gemini":
            print(f"LLM Provider: Gemini ({self.model_name})")
            self.ready = True
        elif self.provider == "groq":
            print(f"LLM Provider: Groq ({self.model_name})")
            self.ready = True
        elif self.provider == "openrouter":
            print(f"LLM Provider: OpenRouter ({self.model_name})")
            self.ready = True
        else:
            print("ERROR: No LLM provider available! Set GEMINI_API_KEY or GROQ_API_KEY.")

    def _extract_search_query(self, messages: list[ChatMessage]) -> str:
        """Extract a search query from the conversation for retrieval."""
        user_messages = [m.content for m in messages if m.role == "user"]

        if not user_messages:
            return ""

        # Use only recent user messages to avoid assistant echo noise.
        return " ".join(user_messages[-3:])

    def _count_turns(self, messages: list[ChatMessage]) -> int:
        """Count total turns (user + assistant messages). The next response will be turn_count + 1."""
        return len(messages) + 1

    def _validate_recommendations(
        self, recommendations: list | None
    ) -> list[Recommendation]:
        """Validate and fix recommendations against the catalog."""
        if recommendations is None:
            return []

        if not isinstance(recommendations, list):
            return []

        validated = []
        for rec in recommendations:
            if isinstance(rec, LLMRecommendation):
                name, url, test_type = rec.name, rec.url, rec.test_type
            elif isinstance(rec, dict):
                name = rec.get("name", "")
                url = rec.get("url", "")
                test_type = rec.get("test_type", "")
            else:
                continue

            # Validate URL against catalog
            if self.retriever.validate_url(url):
                validated.append(
                    Recommendation(name=name, url=url, test_type=test_type)
                )
            else:
                # Try to find by name
                item = self.retriever.find_by_name(name)
                if item:
                    validated.append(
                        Recommendation(
                            name=item["name"],
                            url=item["url"],
                            test_type=",".join(item.get("test_type", [])),
                        )
                    )
                else:
                    print(f"  WARNING: Dropping invalid recommendation: {name} ({url})")

        return validated

    def _call_gemini(self, prompt: str) -> LLMResponse:
        """Call Google Gemini with structured output."""
        from google.genai import types

        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.95,
                max_output_tokens=4096,
                response_mime_type="application/json",
                response_schema=LLMResponse,
            ),
        )

        if response.parsed:
            return response.parsed

        # Fallback: manual JSON parsing
        response_text = response.text.strip()
        parsed = json.loads(response_text)
        return LLMResponse(**parsed)

    def _call_groq(self, prompt: str) -> LLMResponse:
        """Call Groq with JSON mode."""
        response = self.groq_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON matching the requested schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content.strip()
        parsed = json.loads(response_text)
        return LLMResponse(**parsed)

    def _call_openrouter(self, prompt: str) -> LLMResponse:
        """Call OpenRouter with JSON mode via HTTP API."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON matching the requested schema."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        response_text = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(response_text)
        return LLMResponse(**parsed)

    def _call_llm(self, prompt: str) -> LLMResponse:
        """Call the LLM with retry logic."""
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                if self.provider == "gemini":
                    return self._call_gemini(prompt)
                elif self.provider == "groq":
                    return self._call_groq(prompt)
                elif self.provider == "openrouter":
                    return self._call_openrouter(prompt)
                else:
                    raise RuntimeError("No LLM provider configured")

            except Exception as e:
                last_error = e
                error_str = str(e)
                error_lower = error_str.lower()

                # Rate limit or temporary unavailability — retry with backoff or fallback
                if (
                    "429" in error_str
                    or "quota" in error_lower
                    or "rate" in error_lower
                    or "503" in error_str
                    or "unavailable" in error_lower
                ):
                    if self.provider == "groq" and self.openrouter_api_key:
                        print("  Groq limited. Falling back to OpenRouter...")
                        self.provider = "openrouter"
                        self.model_name = "deepseek/deepseek-chat"
                        return self._call_openrouter(prompt)

                    if self.provider == "openrouter" and self.gemini_client:
                        print("  OpenRouter limited. Falling back to Gemini...")
                        self.provider = "gemini"
                        self.model_name = "gemini-2.5-flash"
                        return self._call_gemini(prompt)

                    wait_time = self.RETRY_DELAY_BASE * (2 ** attempt)
                    print(f"  Rate limited (attempt {attempt + 1}/{self.MAX_RETRIES}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  LLM error (attempt {attempt + 1}): {e}")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(1)
                        continue
                    raise

        raise last_error

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        """
        Process a conversation and return the agent's response.
        Single LLM call: retrieval -> prompt building -> generation -> validation.
        """
        # 0. Rule-based refusal for legal/compliance requests
        if messages:
            last_user = next((m for m in reversed(messages) if m.role == "user"), None)
            if last_user:
                user_text = last_user.content.lower()
                if any(term in user_text for term in [
                    "legal",
                    "compliance",
                    "hipaa",
                    "eeoc",
                    "gdpr",
                    "law",
                    "regulation",
                    "regulatory",
                    "lawsuit",
                ]):
                    return ChatResponse(
                        reply=(
                            "I can only assist with selecting SHL assessments. "
                            "For legal or compliance questions, please consult your legal or HR team."
                        ),
                        recommendations=[],
                        end_of_conversation=False,
                    )
        # 1. Extract search query from conversation
        search_query = self._extract_search_query(messages)

        # 2. Retrieve relevant catalog items
        retrieved_items = self.retriever.retrieve(search_query, top_k=12)

        # 3. Count turns for cap awareness
        turn_count = self._count_turns(messages)
        turn_cap_instruction = build_turn_cap_instruction(turn_count)

        # 4. Build the full prompt
        catalog_context = format_catalog_context(retrieved_items)
        conversation_history = format_conversation_history(
            [{"role": m.role, "content": m.content} for m in messages]
        )

        prompt = SYSTEM_PROMPT.format(
            turn_cap_instruction=turn_cap_instruction,
            catalog_context=catalog_context,
            conversation_history=conversation_history,
        )

        # 5. Call LLM with retries
        try:
            llm_response = self._call_llm(prompt)

            # 6. Validate recommendations against catalog
            validated_recommendations = self._validate_recommendations(
                llm_response.recommendations
            )

            # 7. Build response
            return ChatResponse(
                reply=llm_response.reply,
                recommendations=validated_recommendations,
                end_of_conversation=llm_response.end_of_conversation,
            )

        except Exception as e:
            print(f"  ERROR in LLM call: {e}")
            return ChatResponse(
                reply=(
                    "I can help with SHL assessment selection. "
                    "Please confirm the role title, seniority, and key skills, or tell me which assessments you want to add or remove."
                ),
                recommendations=[],
                end_of_conversation=False,
            )
