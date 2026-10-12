# RAG Project

A local Retrieval-Augmented Generation (RAG) application built with Python, Flask, React, and Qdrant. It supports document upload, folder indexing, vector search, and context-based answer generation using either OpenAI-compatible models, DashScope, or a local sentence-transformers embedding backend.

## Overview

This project has a split architecture:

- Flask backend: handles file parsing, chunking, embedding generation, Qdrant storage, and retrieval/query APIs
- React frontend: provides the user interface in the `frontend/` directory
- Qdrant: stores vector embeddings and metadata for document retrieval
- Local embedding backend: default option for development without an external API key

The app is designed for local development and experimentation, while keeping a clean separation between backend logic and frontend UI.

## Features

- Upload one or multiple files for indexing
- Index a whole folder of documents
- Extract text and split it into manageable chunks
- Generate embeddings through a pluggable backend
- Store and search vectors in Qdrant
- Retrieve relevant context for a question
- Generate answers using the configured LLM backend
- Support local and cloud embedding providers
- Use a React interface under `frontend/` with API proxying to Flask
- Centralize prompt templates under a dedicated `prompts/` module for maintainability
- Render prompt variables through a consistent template registry and renderer

## Project Structure

```text
RAG/
├── app.py                         # Flask application and API routes
├── config.py                     # Environment loading and app config
├── demo.py                       # Demo / local test script
├── Dockerfile                    # Container definition for runtime image
├── docker-compose.yml            # Qdrant container setup
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── PLAN.md                      # Progress tracking notes
├── .env                         # Base environment defaults
├── .env.dev                     # Development overrides
├── .env.prod                    # Production overrides
├── .gitignore
├── doc/                         # Project docs and artifacts
├── frontend/                    # React + Vite frontend app
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   ├── public/
│   └── .env.example
├── prompts/
│   ├── __init__.py
│   ├── registry.py               # Prompt registry and metadata
│   ├── render.py                 # Jinja-based prompt rendering
│   ├── templates/
│   │   ├── chat/
│   │   │   └── system.j2
│   │   └── rag/
│   │       ├── answer_with_context.j2
│   │       ├── answer_without_context.j2
│   │       └── rewrite_query.j2
├── services/
│   ├── __init__.py
│   ├── embeddings.py            # Embedding backend implementations
│   ├── llm_service.py          # LLM orchestration + prompt rendering
│   └── qdrant_service.py       # Qdrant collection / search logic
├── static/
├── templates/
├── tests/
├── utils/
│   ├── document_loader.py      # Text extraction and indexing utilities
│   └── text_splitter.py        # Chunking logic
├── env/                        # Local Python virtual environment
├── gradio_app.py               # Gradio version (legacy/alternate UI)
└── __pycache__/
```

## Environment Configuration

The project uses environment files at the project root:

- `.env` — base local settings
- `.env.dev` — development override values
- `.env.prod` — production override values

`config.py` loads the base `.env` and then overlays `.env.<APP_ENV>`, where `APP_ENV` is read from the environment. This means your development configuration can override defaults without changing the base file.

### Typical values

```env
APP_ENV=development
DEBUG=true
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_EMBEDDING_DEVICE=cpu
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents
QDRANT_VECTOR_SIZE=384
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50
```

### Local embedding backend

Use this when you want to run fully locally without an external API key:

```env
APP_ENV=development
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCAL_EMBEDDING_DEVICE=cpu
QDRANT_URL=http://localhost:6333
QDRANT_VECTOR_SIZE=384
```

### OpenAI-compatible backend

```env
APP_ENV=production
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
QDRANT_URL=http://qdrant:6333
QDRANT_VECTOR_SIZE=1536
```

### DashScope backend

```env
EMBEDDING_PROVIDER=dashscope
DASHSCOPE_API_KEY=your_key_here
DASHSCOPE_EMBEDDING_MODEL=qwen3.7-text-embedding-flash
```

## Frontend environment

The React app has its own `.env.example` file in the `frontend/` directory:

```env
VITE_API_BASE_URL=http://localhost:5000
```

In the current setup, Vite is configured to proxy `/api` requests to the Flask backend, so the frontend usually just calls `/api/...` without needing a custom base URL.

## Prompt Template Management

The project now includes a dedicated `prompts/` module for managing large numbers of prompt templates in a maintainable way.

### Why this is needed

When a project grows, prompt strings often become scattered across Python code, which creates several issues:

- prompt logic is hard to review and modify
- variables are mixed into different functions
- it becomes difficult to reuse or compare versions
- prompt maintenance becomes fragile as the project scales

### Current design

The implementation is organized as follows:

- `prompts/registry.py` stores prompt metadata and registration
- `prompts/render.py` renders templates using variable injection
- `prompts/templates/` stores the actual prompt text files
- `services/llm_service.py` uses the prompt renderer instead of embedding raw prompt strings directly in code

This gives a clean separation between business logic and prompt definition.

### Example: using a template with variables

Template file:

```jinja2
Use the following context:
{{ context }}

Question:
{{ question }}

Answer in {{ language }}.
```

Render code:

```python
from prompts.render import render_prompt

prompt = render_prompt(
    "rag.answer_with_context",
    question="What is RAG?",
    context="RAG combines retrieval and generation.",
    language="English",
)
```

### Best practices for large projects

- Keep prompt files in dedicated folders by scenario, such as `rag/`, `chat/`, `summarize/`, and `classification/`
- Register prompt names in a central registry
- Use structured variables instead of ad hoc string concatenation
- Validate required fields before rendering
- Version prompts when changing behavior or evaluation strategy
- Use a consistent language field for output control

### Prompt versioning strategy

For a growing project, versioning is recommended:

- keep the prompt template in a file-based system
- store template metadata in the registry
- if needed, version files like `answer_v2.j2` or `answer_with_context.j2`
- compare prompt behavior before and after changes during evaluation

This makes the system extensible and predictable as the project grows.

## Requirements

Before starting the project, make sure these are installed:

- Python 3.10+
- Node.js + npm for the frontend
- Docker + Docker Compose for Qdrant
- Optional: OpenAI API key or DashScope API key
- Optional: local embedding support via `sentence-transformers`

## Quick Start

### 1) Create or activate the virtual environment

This project already includes a local environment under `env/`, but you can also create your own.

```bash
cd D:\Projects\RAG
D:\Projects\RAG\env\Scripts\python.exe -m pip install -r requirements.txt
```

### 2) Start Qdrant

From the project root:

```bash
docker compose up -d
```

This starts a local Qdrant instance at:

```text
http://localhost:6333
```

If you want to verify it:

```bash
curl http://localhost:6333/collections
```

### 3) Start the Flask backend

```bash
cd D:\Projects\RAG
D:\Projects\RAG\env\Scripts\python.exe app.py
```

The Flask API will run at:

```text
http://localhost:5000
```

Health check:

```text
http://localhost:5000/health
```

### 4) Start the frontend

```bash
cd D:\Projects\RAG\frontend
npm install
npm run dev -- --host 0.0.0.0
```

Then open:

```text
http://localhost:5173
```

The frontend is configured to proxy `/api` traffic to the Flask backend automatically.

## Backend API

The Flask app exposes these main endpoints:

- `GET /` — default page
- `GET /health` — health check
- `POST /api/index` — index raw text
- `POST /api/upload` — upload one or more files
- `POST /api/upload-folder` — index files from a folder path
- `GET /api/documents` — list indexed document sources
- `POST /api/documents/delete` — delete a document source
- `POST /api/clear` — clear the collection
- `POST /api/query` — ask a question and get an answer with sources

## Typical Usage Flow

1. Start Qdrant
2. Start the Flask backend
3. Start the React frontend
4. Upload files or point to a folder to index
5. Ask a question in the UI
6. The app retrieves relevant context from Qdrant
7. The configured LLM generates the final answer

## Important Notes

### Vector size must match the embedding model

This is a common issue when switching backends.

- Local embedding model: `sentence-transformers/all-MiniLM-L6-v2` → vector size is usually `384`
- OpenAI embeddings: often `1536` for `text-embedding-3-small`

If Qdrant was created with the wrong vector size, indexing and retrieval may fail. Keep `QDRANT_VECTOR_SIZE` aligned with your chosen embedding model.

### Local mode is default

The project is set to local embedding mode by default in development, so you can run the app without an API key if you do not want to use OpenAI or DashScope.

### Legacy Gradio UI

There is also a Gradio app in `gradio_app.py`, but the active modern UI lives in the React frontend under `frontend/`.

### Prompt management is now centralized

The prompt system is no longer embedded in the LLM service directly. Prompt files are managed under the `prompts/` module, and the rendering path is standardized, which is much easier to scale as the app grows.

## Troubleshooting

### Flask starts but vector indexing fails

Check:

- `EMBEDDING_PROVIDER` in `.env` or `.env.dev`
- `QDRANT_VECTOR_SIZE`
- whether Qdrant is reachable at `QDRANT_URL`

### Local embedding download warnings

The first time you run the model, it will download from Hugging Face. If you see a warning about unauthenticated requests, it is usually harmless for local usage.

### React frontend cannot call backend

Ensure:

- backend is running on `http://localhost:5000`
- frontend is running on `http://localhost:5173`
- Vite proxy is configured in `frontend/vite.config.js`

## License

This project is intended for local development, experimentation, and learning. Use it as a base for your own RAG or document-search workflow.
