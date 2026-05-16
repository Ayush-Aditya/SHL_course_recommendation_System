# Deployment Guide

This project is ready for Docker-based deployment on Render or Railway. Secrets must be provided as environment variables in the platform dashboard.

## Render (Docker)

1. Push this repository to GitHub.
2. In Render, create a new Web Service and connect the repository.
3. Render reads render.yaml and Dockerfile automatically.
4. Set environment variables in the Render dashboard:
   - GROQ_API_KEY
   - OPENROUTER_API_KEY
   - GEMINI_API_KEY
   - LLM_PROVIDER (recommended: groq)
5. Deploy and verify:
   - GET /health returns {"status":"ok"}

## Railway (Docker)

1. Push this repository to GitHub.
2. In Railway, create a new project and connect the repository.
3. Railway detects the Dockerfile and builds the service.
4. Set environment variables:
   - GROQ_API_KEY
   - OPENROUTER_API_KEY
   - GEMINI_API_KEY
   - LLM_PROVIDER (recommended: groq)
5. Deploy and verify:
   - GET /health returns {"status":"ok"}

## Secret Handling

- Do not commit .env to Git.
- Store secrets only in the deployment platform environment variables.
- If any key was exposed, rotate it immediately.
