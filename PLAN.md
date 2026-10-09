# RAG Project Plan

## Completed

- Created the initial project structure
- Added Flask application entry point
- Added configuration management in `config.py`
- Added Qdrant vector database service
- Added LLM / embedding service integration
- Added basic text chunking utility
- Added Gradio UI entry point
- Added Docker and Docker Compose setup for Qdrant
- Added English project README
- Added `.env.example` template
- Added document upload and automatic indexing support
- Added multi-file upload support
- Added source-based document listing
- Added source-specific deletion
- Added knowledge base clearing function

## In Progress

- No major task is currently blocked

## Next Tasks

1. Completed: add folder-level upload support for entire directories
2. Improve the document management interface in Gradio
3. Improve PDF extraction quality for Chinese and complex layouts
4. Add more robust error handling and retry logic for Qdrant/OpenAI calls
5. Add lightweight automated tests for indexing and query flow
6. Add a production deployment guide and environment hardening notes

## Notes

- The project is designed as a lightweight RAG starter using Flask, Gradio, Qdrant, and OpenAI-compatible models.
- Qdrant is run through Docker locally at `http://localhost:6333`.
- The Flask API and Gradio UI are kept separate so the backend and frontend responsibilities remain clear.
- This file should be updated whenever major features are added or completed.
